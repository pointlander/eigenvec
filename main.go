// Copyright 2026 The eigenvec Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command eigenvec builds an order-6 Markov graph of the characters in a
// text, computes its PageRank (the principal eigenvector of the damped
// transition matrix), and generates text from that ranking.
//
// A prompt is walked as a path of overlapping 6-character nodes. At every
// node the count-weighted PageRank of the successors is normalized into a
// distribution over the next character. A transformer is pretrained on the
// text, one token at a time. Each token is its PageRank embedding: the
// node's rank on the characters it contains, and the PageRank it sends to
// each successor. Attention reads the last thousand of those vectors and
// predicts the next rune. Each new character is chosen by Monte Carlo tree
// search. The tempered token distribution is tilted by that prediction.
// Closing a frequent word such as "the" is discounted by how often the
// corpus uses it. A rollout scores a path by the geometric mean of those
// probabilities, and the character is drawn from the visit counts. A
// rollout only reads the network.
package main

import (
	"archive/tar"
	"archive/zip"
	"bufio"
	"cmp"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"simd"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

const (
	// damp is the PageRank damping factor.
	damp = 0.85
	// tol is the L1 convergence tolerance for power iteration.
	tol = 1e-12
	// maxRankIter caps power iteration.
	maxRankIter = 10000
	// order is the number of characters in a Markov token.
	order = 6
	// learnRate is the step size of one pretraining update.
	learnRate = 0.05
	// dModel is the width of a token vector inside the transformer.
	dModel = 32
	// ffnDim is the inner width of the position-wise feed-forward layer.
	ffnDim = 128
	// ctxLen is how many token vectors attention can see.
	ctxLen = 1000
	// clipNorm bounds the output gradient on each step.
	clipNorm = 1
	// tilt is the share of a draw taken from the transformer.
	// The rest follows the current token.
	tilt = 0.15
	// temp softens the token distribution. 1 leaves it unchanged.
	temp = 1.5
	// mctsSims is the number of simulations behind each generated character.
	mctsSims = 32
	// mctsDepth is how far a simulation looks ahead, in characters.
	mctsDepth = 16
	// mctsC is the PUCT exploration constant.
	mctsC = 1.5
	// freq discounts closing a word in proportion to how often the corpus
	// uses it. 0 leaves frequent words at their token probability.
	freq = 0.5
)

// token is an order-6 Markov state: six successive characters.
type token [order]rune

// edge is a weighted transition to the shifted token.
type edge struct {
	to int
	w  float64
	ch rune
	// p is the count-weighted successor PageRank, normalized over this node's successors.
	p float64
}

// Graph is the order-6 character graph and its PageRank vector.
type Graph struct {
	outs  [][]edge
	token []token
	id    map[token]int
	rank  []float64
	// words counts lowercased words in the text. A word is a run of letters.
	words map[string]int
	// src is the text the graph was built from.
	src []rune
	// runeID is the coordinate of each rune in a PageRank embedding.
	runeID map[rune]int
}

func main() {
	file := flag.String("file", "pg100.txt", "text to model")
	prompt := flag.String("prompt", "To be, or not to be", "prompt that starts the walk")
	n := flag.Int("n", 400, "characters to generate after the prompt")
	seed := flag.Uint64("seed", 1, "sampling seed")
	top := flag.Int("top", 12, "how many PageRank tokens to print")
	gutenberg := flag.Int("gutenberg", 0, "also train on the first n books in txt-files.tar.zip")
	flag.Parse()
	if *gutenberg < 0 {
		log.Fatal("-gutenberg must be >= 0")
	}

	text, err := os.ReadFile(*file)
	if err != nil {
		log.Fatal(err)
	}
	body := prepare(string(text))
	if *gutenberg > 0 {
		extra, err := readGutenberg("txt-files.tar.zip", *gutenberg)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "gutenberg books %d\n", *gutenberg)
		if extra != "" {
			body += "\n" + extra
		}
	}
	runes := []rune(body)
	if len(runes) < order+1 {
		log.Fatalf("text is shorter than an order-%d transition", order)
	}

	g := buildGraph(runes)
	iters := g.rankPages(damp, tol)
	g.normalizeSuccessors()

	fmt.Fprintf(os.Stderr, "tokens %d edges %d pagerank iterations %d\n", len(g.token), g.edgeCount(), iters)
	g.printTop(os.Stderr, *top)

	out := g.generate([]rune(*prompt), *n, *seed)
	fmt.Println(string(out))
}

// prepare drops the Project Gutenberg header and license, joins a period
// to the next line, and removes blank lines. ".\\n" and "\\n\\n" are the
// breaks between speeches. Leading spaces and the right-margin verse
// numbers are removed.
func prepare(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	const startMark = "*** START OF THE PROJECT GUTENBERG EBOOK"
	const endMark = "*** END OF THE PROJECT GUTENBERG EBOOK"
	if i := strings.Index(s, startMark); i >= 0 {
		s = s[i+len(startMark):]
		if nl := strings.IndexByte(s, '\n'); nl >= 0 {
			s = s[nl+1:]
		} else {
			s = ""
		}
	}
	if i := strings.Index(s, endMark); i >= 0 {
		s = s[:i]
	}
	s = cleanLines(s)

	var b strings.Builder
	b.Grow(len(s))
	newlines := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '.' && i+1 < len(s) && s[i+1] == '\n' {
			b.WriteString(". ")
			i++
			newlines = 0
			continue
		}
		if s[i] == '\n' {
			newlines++
			if newlines == 1 {
				b.WriteByte('\n')
			}
			continue
		}
		newlines = 0
		b.WriteByte(s[i])
	}
	return strings.TrimSpace(b.String())
}

// readGutenberg returns the prepared text of the first n books in
// txt-files.tar.zip. Books are the .txt members of txt-files.tar, in the
// order they are stored. The books are separated by a newline.
func readGutenberg(path string, n int) (string, error) {
	z, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer z.Close()
	var member *zip.File
	for _, f := range z.File {
		if f.Name == "txt-files.tar" || strings.HasSuffix(f.Name, "/txt-files.tar") {
			member = f
			break
		}
	}
	if member == nil {
		return "", fmt.Errorf("%s has no txt-files.tar", path)
	}
	r, err := member.Open()
	if err != nil {
		return "", err
	}
	defer r.Close()
	return readBooks(tar.NewReader(r), n)
}

// readBooks prepares the first n .txt files from tr.
func readBooks(tr *tar.Reader, n int) (string, error) {
	var b strings.Builder
	got := 0
	for got < n {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		if !hdr.FileInfo().Mode().IsRegular() || !strings.HasSuffix(hdr.Name, ".txt") {
			continue
		}
		var buf strings.Builder
		if _, err := io.Copy(&buf, tr); err != nil {
			return "", err
		}
		got++
		text := prepare(buf.String())
		if text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(text)
	}
	if got < n {
		return "", fmt.Errorf("txt-files.tar.zip has %d books, want %d", got, n)
	}
	return b.String(), nil
}

// cleanLines trims leading spaces, drops a verse number that sits at the
// end of a line after at least two spaces, and drops lines that are only a
// number.
func cleanLines(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	start := 0
	for i := 0; i <= len(s); i++ {
		if i != len(s) && s[i] != '\n' {
			continue
		}
		line := strings.TrimLeft(s[start:i], " \t")
		start = i + 1
		if isDigits(line) {
			continue
		}
		line = stripMarginNumber(line)
		b.WriteString(line)
		if i != len(s) {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// stripMarginNumber removes a trailing run of digits when at least two
// spaces separate it from the rest of the line.
func stripMarginNumber(line string) string {
	i := len(line)
	for i > 0 && line[i-1] >= '0' && line[i-1] <= '9' {
		i--
	}
	if i == len(line) {
		return line
	}
	j := i
	for j > 0 && line[j-1] == ' ' {
		j--
	}
	if i-j < 2 {
		return line
	}
	return line[:j]
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// buildGraph counts order-6 transitions. A node links to the token shifted
// one character ahead once for every time that character follows it.
// Windows overlap by five characters.
func buildGraph(runes []rune) *Graph {
	seen := make(map[token]struct{})
	counts := make(map[token]map[rune]int)
	for i := 0; i+order <= len(runes); i++ {
		tok := tokenFrom(runes[i : i+order])
		seen[tok] = struct{}{}
		if i+order >= len(runes) {
			continue
		}
		ch := runes[i+order]
		m := counts[tok]
		if m == nil {
			m = make(map[rune]int)
			counts[tok] = m
		}
		m[ch]++
	}

	keys := make([]token, 0, len(seen))
	for tok := range seen {
		keys = append(keys, tok)
	}
	slices.SortFunc(keys, func(a, b token) int {
		for i := range a {
			if c := cmp.Compare(a[i], b[i]); c != 0 {
				return c
			}
		}
		return 0
	})

	g := &Graph{
		outs:  make([][]edge, len(keys)),
		token: keys,
		id:    make(map[token]int, len(keys)),
		words: wordCounts(runes),
		src:   runes,
	}
	for i, tok := range keys {
		g.id[tok] = i
	}
	for tok, m := range counts {
		from := g.id[tok]
		edges := make([]edge, 0, len(m))
		for ch, w := range m {
			to, ok := g.id[shifted(tok, ch)]
			if !ok {
				continue
			}
			edges = append(edges, edge{to: to, w: float64(w), ch: ch})
		}
		slices.SortFunc(edges, func(a, b edge) int {
			return cmp.Compare(a.ch, b.ch)
		})
		g.outs[from] = edges
	}
	return g
}

// wordCounts counts maximal runs of letters, folded to lower case.
func wordCounts(runes []rune) map[string]int {
	counts := make(map[string]int)
	for i := 0; i < len(runes); {
		if !isLetter(runes[i]) {
			i++
			continue
		}
		j := i + 1
		for j < len(runes) && isLetter(runes[j]) {
			j++
		}
		counts[strings.ToLower(string(runes[i:j]))]++
		i = j
	}
	return counts
}

func isLetter(r rune) bool {
	return unicode.IsLetter(r)
}

// trailingWord is the lowercased run of letters at the end of t.
func trailingWord(t token) string {
	i := order
	for i > 0 && isLetter(t[i-1]) {
		i--
	}
	if i == order {
		return ""
	}
	return strings.ToLower(string(t[i:]))
}

// closePenalty is below one when ch ends a word the corpus uses often.
func (g *Graph) closePenalty(tok token, ch rune) float64 {
	if g.words == nil || isLetter(ch) {
		return 1
	}
	word := trailingWord(tok)
	if word == "" {
		return 1
	}
	c := g.words[word]
	if c <= 1 {
		return 1
	}
	return 1 / (1 + freq*math.Log(float64(c)))
}

func tokenFrom(rs []rune) token {
	var t token
	copy(t[:], rs)
	return t
}

// shifted is the next token after appending ch.
func shifted(t token, ch rune) token {
	var n token
	copy(n[:order-1], t[1:])
	n[order-1] = ch
	return n
}

func (g *Graph) edgeCount() int {
	n := 0
	for _, outs := range g.outs {
		n += len(outs)
	}
	return n
}

// rankPages computes weighted PageRank by power iteration.
// A link is weighted by the log of its occurrence count. The result is the
// unique positive eigenvector of the Google matrix, normalized to sum to one.
func (g *Graph) rankPages(damp, tol float64) int {
	n := len(g.outs)
	if n == 0 {
		return 0
	}
	// lw is parallel to outs. log(1) is zero, so a once-seen link sends no rank.
	lw := make([][]float64, n)
	outw := make([]float64, n)
	for i, outs := range g.outs {
		lw[i] = make([]float64, len(outs))
		for j, e := range outs {
			if e.w > 1 {
				lw[i][j] = math.Log(e.w)
			}
			outw[i] += lw[i][j]
		}
	}
	rank := make([]float64, n)
	next := make([]float64, n)
	inv := 1 / float64(n)
	for i := range rank {
		rank[i] = inv
	}
	teleport := (1 - damp) * inv

	iters := 0
	for iters = 1; iters <= maxRankIter; iters++ {
		for i := range next {
			next[i] = teleport
		}
		dangling := 0.0
		for i, outs := range g.outs {
			if outw[i] == 0 {
				dangling += rank[i]
				continue
			}
			scale := damp * rank[i] / outw[i]
			for j, e := range outs {
				next[e.to] += scale * lw[i][j]
			}
		}
		if dangling > 0 {
			share := damp * dangling * inv
			for i := range next {
				next[i] += share
			}
		}
		delta := 0.0
		for i := range rank {
			delta += math.Abs(next[i] - rank[i])
		}
		rank, next = next, rank
		if delta < tol {
			break
		}
	}
	g.rank = rank
	return iters
}

// normalizeSuccessors stores, on each edge, the count-weighted successor
// PageRank divided by the sum of those weights. That is the local
// next-character distribution.
func (g *Graph) normalizeSuccessors() {
	for i, outs := range g.outs {
		sum := 0.0
		for _, e := range outs {
			sum += e.w * g.rank[e.to]
		}
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		for j := range outs {
			g.outs[i][j].p = outs[j].w * g.rank[outs[j].to] * inv
		}
	}
}

// generate pretrains the transformer on the text, then replays the prompt
// through it. Each character is chosen by Monte Carlo tree search. The
// tempered token distribution is tilted by the network. The walk then
// appends that successor to the context. A rollout only reads the network.
func (g *Graph) generate(prompt []rune, n int, seed uint64) []rune {
	if len(prompt) < order {
		log.Fatalf("prompt must contain at least %d characters", order)
	}
	net := newTransformer(seed, g.alphabet())
	g.prepareEmbed(net.id)
	g.pretrain(net)
	net.reset()
	cur := -1
	var ctx []int
	for i := 0; i+order <= len(prompt); i++ {
		node, ok := g.id[tokenFrom(prompt[i:i+order])]
		if !ok {
			continue
		}
		ctx = remember(ctx, node)
		cur = node
	}
	if cur < 0 {
		log.Fatalf("prompt shares no order-%d token with the text", order)
	}

	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	out := append([]rune(nil), prompt...)
	for len(out) < len(prompt)+n {
		if len(g.outs[cur]) == 0 {
			break
		}
		e := g.search(net, ctx, cur, rng)
		out = append(out, e.ch)
		cur = e.to
		ctx = remember(ctx, cur)
	}
	return out
}

// edgeWeight is the current token's edge probability, tilted by the
// network. The network's distribution is renormalized over the legal
// successors. A character the network does not predict keeps its share of
// the token distribution.
func edgeWeight(e edge, dist map[rune]float64, priorSum float64) float64 {
	w := (1 - tilt) * e.p
	if priorSum > 0 {
		w += tilt * dist[e.ch] / priorSum
	}
	return w
}

// edgeWeights is the tempered token distribution tilted by the network,
// one weight per outgoing edge.
func edgeWeights(dist map[rune]float64, outs []edge) []float64 {
	priorSum := 0.0
	for _, e := range outs {
		if p := dist[e.ch]; p > 0 {
			priorSum += p
		}
	}
	grams := temper(outs)
	weights := make([]float64, len(outs))
	for i, e := range outs {
		e.p = grams[i]
		weights[i] = edgeWeight(e, dist, priorSum)
	}
	return weights
}

// temper raises each edge probability to 1/temp and renormalizes, which
// flattens the token distribution when temp is above 1.
func temper(outs []edge) []float64 {
	grams := make([]float64, len(outs))
	if temp == 1 {
		for i, e := range outs {
			grams[i] = e.p
		}
		return grams
	}
	inv := 1 / temp
	sum := 0.0
	for i, e := range outs {
		if e.p > 0 {
			grams[i] = math.Pow(e.p, inv)
			sum += grams[i]
		}
	}
	if sum == 0 {
		return grams
	}
	for i := range grams {
		grams[i] /= sum
	}
	return grams
}

// priorWeights is the tempered token distribution, tilted by the network
// and discounted when a frequent word closes.
func (g *Graph) priorWeights(dist map[rune]float64, id int, outs []edge) []float64 {
	weights := edgeWeights(dist, outs)
	if id < 0 || id >= len(g.token) {
		return weights
	}
	tok := g.token[id]
	for i, e := range outs {
		weights[i] *= g.closePenalty(tok, e.ch)
	}
	return weights
}

// sampleProb draws a successor and returns its share of priorWeights.
func (g *Graph) sampleProb(dist map[rune]float64, id int, outs []edge, rng *rand.Rand) (edge, float64) {
	weights := g.priorWeights(dist, id, outs)
	total := 0.0
	for _, w := range weights {
		total += w
	}
	if total == 0 {
		i := rng.IntN(len(outs))
		return outs[i], 1 / float64(len(outs))
	}
	r := rng.Float64() * total
	acc := 0.0
	last := 0
	seen := false
	for i := range outs {
		if weights[i] <= 0 {
			continue
		}
		last = i
		seen = true
		acc += weights[i]
		if r <= acc {
			return outs[i], weights[i] / total
		}
	}
	if !seen {
		last = len(outs) - 1
	}
	return outs[last], weights[last] / total
}

// mctsChild is one explored continuation of an mctsNode.
type mctsChild struct {
	prior  float64
	visits float64
	value  float64
	edge   edge
	node   *mctsNode
}

// mctsNode is a position in the search tree. id is the graph node.
// ctx is the attention context after reading this token, oldest first.
type mctsNode struct {
	id       int
	ctx      []int
	visits   float64
	expanded bool
	children []mctsChild
}

// search chooses the next edge by Monte Carlo tree search and draws it
// from the root visit counts. A node with one continuation is returned as it is.
// ctx is the attention context at cur. The network is only read.
func (g *Graph) search(net *Transformer, ctx []int, cur int, rng *rand.Rand) edge {
	root := g.explore(net, ctx, cur, rng)
	return root.draw(rng)
}

// explore runs the simulations from cur and returns the search root.
// The network is only read.
func (g *Graph) explore(net *Transformer, ctx []int, cur int, rng *rand.Rand) *mctsNode {
	root := &mctsNode{id: cur, ctx: ctx}
	g.expand(root, net)
	if len(root.children) <= 1 {
		for i := range root.children {
			root.children[i].visits = 1
		}
		return root
	}
	for range mctsSims {
		g.playout(root, net, rng)
	}
	return root
}

// expand attaches one child per outgoing edge. The prior is the tilted
// distribution at this state.
func (g *Graph) expand(n *mctsNode, net *Transformer) {
	if n.expanded {
		return
	}
	n.expanded = true
	outs := g.outs[n.id]
	if len(outs) == 0 {
		return
	}
	weights := g.priorWeights(g.distAt(net, n.ctx, n.id), n.id, outs)
	sum := 0.0
	for _, w := range weights {
		sum += w
	}
	n.children = make([]mctsChild, len(outs))
	for i, e := range outs {
		p := 1 / float64(len(outs))
		if sum > 0 {
			p = weights[i] / sum
		}
		n.children[i] = mctsChild{prior: p, edge: e}
	}
}

// playout walks the tree by PUCT, rolls out the rest of the horizon, and
// backs up the geometric mean of the model probabilities along each suffix.
func (g *Graph) playout(root *mctsNode, net *Transformer, rng *rand.Rand) {
	node := root
	path := make([]*mctsChild, 0, mctsDepth)
	seq := make([]float64, 0, mctsDepth)
	for len(path) < mctsDepth && node.expanded && len(node.children) > 0 {
		c := node.pick()
		if c.node == nil {
			var ctx []int
			if net != nil {
				ctx = remember(node.ctx, c.edge.to)
			}
			c.node = &mctsNode{id: c.edge.to, ctx: ctx}
		}
		path = append(path, c)
		node = c.node
		seq = append(seq, logP(c.prior))
		if !node.expanded {
			break
		}
	}
	if !node.expanded {
		g.expand(node, net)
	}
	id := node.id
	ctx := node.ctx
	for len(seq) < mctsDepth {
		outs := g.outs[id]
		if len(outs) == 0 {
			break
		}
		e, p := g.sampleProb(g.distAt(net, ctx, id), id, outs, rng)
		id = e.to
		if net != nil {
			ctx = remember(ctx, id)
		}
		seq = append(seq, logP(p))
	}
	if len(seq) == 0 {
		return
	}
	root.visits++
	for i, c := range path {
		c.visits++
		c.value += math.Exp(meanFrom(seq, i))
		if i+1 < len(path) {
			c.node.visits++
		}
	}
}

// distAt is the network's prediction on this node's outgoing edges.
// A nil network leaves the token distribution untilted.
func (g *Graph) distAt(net *Transformer, ctx []int, id int) map[rune]float64 {
	if net == nil || id < 0 || id >= len(g.outs) {
		return nil
	}
	sm := g.predictNet(net, ctx)
	outs := g.outs[id]
	dist := make(map[rune]float64, len(outs))
	for _, e := range outs {
		if i, ok := net.id[e.ch]; ok && i < len(sm) {
			dist[e.ch] = sm[i]
		}
	}
	return dist
}

// pick selects the child with the highest PUCT score. Ties keep the
// earlier edge, and edges are ordered by character.
func (n *mctsNode) pick() *mctsChild {
	sqrtN := math.Sqrt(n.visits + 1)
	best := 0
	bestU := math.Inf(-1)
	for i := range n.children {
		c := &n.children[i]
		q := 0.0
		if c.visits > 0 {
			q = c.value / c.visits
		}
		u := q + mctsC*c.prior*sqrtN/(1+c.visits)
		if u > bestU {
			best = i
			bestU = u
		}
	}
	return &n.children[best]
}

// draw samples an edge in proportion to how often search visited it.
func (n *mctsNode) draw(rng *rand.Rand) edge {
	total := 0.0
	for i := range n.children {
		total += n.children[i].visits
	}
	if total == 0 {
		return n.children[rng.IntN(len(n.children))].edge
	}
	r := rng.Float64() * total
	acc := 0.0
	var last edge
	for i := range n.children {
		if n.children[i].visits <= 0 {
			continue
		}
		last = n.children[i].edge
		acc += n.children[i].visits
		if r <= acc {
			return n.children[i].edge
		}
	}
	return last
}

func meanFrom(v []float64, i int) float64 {
	sum := 0.0
	for _, x := range v[i:] {
		sum += x
	}
	return sum / float64(len(v)-i)
}

// alphabet is the sorted set of runes this graph can read or emit.
func (g *Graph) alphabet() []rune {
	seen := make(map[rune]struct{})
	add := func(r rune) {
		seen[r] = struct{}{}
	}
	for i := range g.token {
		for _, r := range g.token[i] {
			add(r)
		}
	}
	for _, outs := range g.outs {
		for _, e := range outs {
			add(e.ch)
		}
	}
	out := make([]rune, 0, len(seen))
	for r := range seen {
		out = append(out, r)
	}
	slices.Sort(out)
	return out
}

// Transformer is one decoder block. It reads PageRank embeddings and
// predicts the next rune. Attention sees the last ctxLen token vectors.
// The gradient on each step treats keys and values already in the context
// as constant. Products and rank-1 updates run on simd vectors.
type Transformer struct {
	id    map[rune]int
	vocab []rune
	in    int

	wx             []float64
	bx             []float64
	wq, wk, wv, wo []float64
	bq, bk, bv, bo []float64
	rel            []float64
	w1, b1         []float64
	w2, b2         []float64
	wy, by         []float64

	ringH, ringK, ringV       []float64
	hcur, kcur, vcur          []float64
	q, ctxv, a, pre, u, y     []float64
	dy, du, da, dctx          []float64
	dq, dk, dv, dh            []float64
	score, attn, logit, embed []float64
	genK, genV                []float64
	cache                     map[int]proj
}

// proj is the cached projection of one graph node. It depends only on the
// embedding, so a frozen network can reuse it for every visit of that node.
type proj struct {
	h, k, v []float64
}

// newTransformer draws a small deterministic initialization from seed.
// vocab is the runes the network can read and predict. Duplicates are dropped.
// Zero weights would leave the gradient at zero. The attention and
// feed-forward output maps start small so the residual path leads early updates.
func newTransformer(seed uint64, runes []rune) *Transformer {
	vocab := uniqueRunes(runes)
	n := &Transformer{
		id:    make(map[rune]int, len(vocab)),
		vocab: vocab,
		in:    len(vocab),
		wx:    make([]float64, dModel*len(vocab)),
		bx:    make([]float64, dModel),
		wq:    make([]float64, dModel*dModel),
		wk:    make([]float64, dModel*dModel),
		wv:    make([]float64, dModel*dModel),
		wo:    make([]float64, dModel*dModel),
		bq:    make([]float64, dModel),
		bk:    make([]float64, dModel),
		bv:    make([]float64, dModel),
		bo:    make([]float64, dModel),
		rel:   make([]float64, ctxLen),
		w1:    make([]float64, ffnDim*dModel),
		b1:    make([]float64, ffnDim),
		w2:    make([]float64, dModel*ffnDim),
		b2:    make([]float64, dModel),
		wy:    make([]float64, len(vocab)*dModel),
		by:    make([]float64, len(vocab)),
		ringH: make([]float64, 0, ctxLen*dModel),
		ringK: make([]float64, 0, ctxLen*dModel),
		ringV: make([]float64, 0, ctxLen*dModel),
		hcur:  make([]float64, dModel),
		kcur:  make([]float64, dModel),
		vcur:  make([]float64, dModel),
		q:     make([]float64, dModel),
		ctxv:  make([]float64, dModel),
		a:     make([]float64, dModel),
		pre:   make([]float64, ffnDim),
		u:     make([]float64, ffnDim),
		y:     make([]float64, dModel),
		dy:    make([]float64, dModel),
		du:    make([]float64, ffnDim),
		da:    make([]float64, dModel),
		dctx:  make([]float64, dModel),
		dq:    make([]float64, dModel),
		dk:    make([]float64, dModel),
		dv:    make([]float64, dModel),
		dh:    make([]float64, dModel),
		logit: make([]float64, len(vocab)),
		embed: make([]float64, len(vocab)),
		cache: make(map[int]proj),
	}
	for i, r := range vocab {
		n.id[r] = i
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	fanIn := n.in
	if fanIn < 1 {
		fanIn = 1
	}
	fillUnit(rng, n.wx, fanIn, 1)
	fillUnit(rng, n.wq, dModel, 1)
	fillUnit(rng, n.wk, dModel, 1)
	fillUnit(rng, n.wv, dModel, 1)
	fillUnit(rng, n.wo, dModel, 0.1)
	fillUnit(rng, n.w1, dModel, 1)
	fillUnit(rng, n.w2, ffnDim, 0.1)
	fillUnit(rng, n.wy, dModel, 1)
	return n
}

// fillUnit fills w with uniform values of width gain/sqrt(fanIn).
func fillUnit(rng *rand.Rand, w []float64, fanIn int, gain float64) {
	s := gain / math.Sqrt(float64(fanIn))
	for i := range w {
		w[i] = (rng.Float64()*2 - 1) * s
	}
}

func uniqueRunes(runes []rune) []rune {
	seen := make(map[rune]struct{}, len(runes))
	out := make([]rune, 0, len(runes))
	for _, r := range runes {
		if _, ok := seen[r]; ok {
			continue
		}
		seen[r] = struct{}{}
		out = append(out, r)
	}
	slices.Sort(out)
	return out
}

// prepareEmbed selects the rune coordinates of the PageRank embeddings.
func (g *Graph) prepareEmbed(index map[rune]int) {
	g.runeID = index
}

// writeEmbed writes the PageRank embedding of a node.
// A character coordinate receives the node's rank, weighted by where that
// character sits in the token, plus the PageRank probability of leaving
// through that character.
func (g *Graph) writeEmbed(id int, dst []float64) {
	for i := range dst {
		dst[i] = 0
	}
	if g.runeID == nil || id < 0 || id >= len(g.token) || id >= len(g.rank) {
		return
	}
	loud := math.Log1p(g.rank[id] * float64(len(g.rank)))
	posSum := float64(order * (order + 1) / 2)
	for k, r := range g.token[id] {
		j, ok := g.runeID[r]
		if !ok || j >= len(dst) {
			continue
		}
		dst[j] += loud * float64(k+1) / posSum
	}
	for _, e := range g.outs[id] {
		j, ok := g.runeID[e.ch]
		if !ok || j >= len(dst) {
			continue
		}
		dst[j] += e.p
	}
}

// remember appends id to the attention context and drops the oldest token
// once the context holds ctxLen vectors. The result does not alias ctx.
func remember(ctx []int, id int) []int {
	if len(ctx) >= ctxLen {
		next := make([]int, ctxLen)
		copy(next, ctx[len(ctx)-(ctxLen-1):])
		next[ctxLen-1] = id
		return next
	}
	next := make([]int, len(ctx)+1)
	copy(next, ctx)
	next[len(ctx)] = id
	return next
}

// pretrain walks the text once. Each step appends the token vector to the
// context and predicts the rune that follows.
func (g *Graph) pretrain(net *Transformer) {
	if net == nil || g.runeID == nil || len(g.src) < order+1 {
		return
	}
	fmt.Fprintf(os.Stderr, "pretraining transformer, alphabet %d, context %d\n", len(net.vocab), ctxLen)
	net.reset()
	x := make([]float64, net.in)
	for i := 0; i+order < len(g.src); i++ {
		id, ok := g.id[tokenFrom(g.src[i:i+order])]
		if !ok {
			continue
		}
		g.writeEmbed(id, x)
		net.train(x, g.src[i+order])
	}
}

// vecLen is the number of float64 lanes in one simd vector.
// The binary is built with GOEXPERIMENT=simd. The width stays fixed for
// the process. It is read from a zero vector because a package-level
// SIMD initializer is not reliable.
func vecLen() int {
	return (simd.Float64s{}).Len()
}

// hsum adds the lanes of v. A vector holds at most eight float64s.
func hsum(v simd.Float64s) float64 {
	var buf [8]float64
	v.Store(buf[:])
	s := 0.0
	for i := range v.Len() {
		s += buf[i]
	}
	return s
}

// addDots adds each row of w, dotted with x, into dst.
// w is row-major, with len(dst) rows of width. A short tail is loaded
// with zeros in the unused lanes. Lane sums are not a left fold.
func addDots(w, x, dst []float64, width int) {
	if width == 0 || len(dst) == 0 {
		return
	}
	x = x[:width:width]
	n := vecLen()
	full := width - width%n
	for r := range dst {
		base := r * width
		row := w[base : base+width : base+width]
		var acc simd.Float64s
		for i := n; i <= full; i += n {
			acc = simd.LoadFloat64s(row[i-n:i]).MulAdd(simd.LoadFloat64s(x[i-n:i]), acc)
		}
		if full < width {
			rv, _ := simd.LoadFloat64sPart(row[full:])
			xv, _ := simd.LoadFloat64sPart(x[full:])
			acc = rv.MulAdd(xv, acc)
		}
		dst[r] += hsum(acc)
	}
}

// mixRows adds coef[r] times row r of w into dst.
// w is row-major, with len(coef) rows of width. dst has length width.
func mixRows(dst, w, coef []float64, width int) {
	if width == 0 || len(coef) == 0 {
		return
	}
	dst = dst[:width:width]
	n := vecLen()
	full := width - width%n
	for r, step := range coef {
		scale := simd.BroadcastFloat64s(step)
		base := r * width
		row := w[base : base+width : base+width]
		for i := n; i <= full; i += n {
			scale.MulAdd(simd.LoadFloat64s(row[i-n:i]), simd.LoadFloat64s(dst[i-n:i])).Store(dst[i-n : i])
		}
		if full < width {
			rv, _ := simd.LoadFloat64sPart(row[full:])
			dv, _ := simd.LoadFloat64sPart(dst[full:])
			scale.MulAdd(rv, dv).StorePart(dst[full:])
		}
	}
}

// rank1 adds (alpha * left[r]) * right to row r of a.
// a is row-major, with len(left) rows of width.
func rank1(a, left, right []float64, width int, alpha float64) {
	if width == 0 || len(left) == 0 {
		return
	}
	right = right[:width:width]
	n := vecLen()
	full := width - width%n
	for r, v := range left {
		scale := simd.BroadcastFloat64s(alpha * v)
		base := r * width
		row := a[base : base+width : base+width]
		for i := n; i <= full; i += n {
			scale.MulAdd(simd.LoadFloat64s(right[i-n:i]), simd.LoadFloat64s(row[i-n:i])).Store(row[i-n : i])
		}
		if full < width {
			rv, _ := simd.LoadFloat64sPart(right[full:])
			av, _ := simd.LoadFloat64sPart(row[full:])
			scale.MulAdd(rv, av).StorePart(row[full:])
		}
	}
}

// dot is the inner product of two equal-length slices.
func dot(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	var dst [1]float64
	addDots(a, b, dst[:], len(a))
	return dst[0]
}

// addScaled adds step*b to a. a and b have the same length.
func addScaled(a, b []float64, step float64) {
	if len(a) == 0 {
		return
	}
	left := [1]float64{1}
	rank1(a, left[:], b, len(a), step)
}

// scale multiplies each element of a by step.
func scale(a []float64, step float64) {
	if len(a) == 0 {
		return
	}
	n := vecLen()
	factor := simd.BroadcastFloat64s(step)
	a = a[:len(a):len(a)]
	full := len(a) - len(a)%n
	for i := n; i <= full; i += n {
		simd.LoadFloat64s(a[i-n : i]).Mul(factor).Store(a[i-n : i])
	}
	if full < len(a) {
		av, _ := simd.LoadFloat64sPart(a[full:])
		av.Mul(factor).StorePart(a[full:])
	}
}

// reset clears the context and the projection cache. Weights stay.
func (n *Transformer) reset() {
	n.ringH = n.ringH[:0]
	n.ringK = n.ringK[:0]
	n.ringV = n.ringV[:0]
	n.cache = make(map[int]proj)
}

// project writes h, k, and v for one embedding.
func (n *Transformer) project(x, h, k, v []float64) {
	copy(h, n.bx)
	addDots(n.wx, x, h, n.in)
	copy(k, n.bk)
	addDots(n.wk, h, k, dModel)
	copy(v, n.bv)
	addDots(n.wv, h, v, dModel)
}

// appendRing stores one token vector, dropping the oldest at ctxLen.
func appendRing(ring, row []float64) []float64 {
	if len(ring)/dModel >= ctxLen {
		copy(ring, ring[dModel:])
		ring = ring[:len(ring)-dModel]
	}
	return append(ring, row...)
}

// logits writes the next-rune scores for h attending over keys and vals.
// keys and vals are row-major, one dModel row per context position, oldest first.
func (n *Transformer) logits(h, keys, vals []float64) {
	nctx := len(keys) / dModel
	copy(n.q, n.bq)
	addDots(n.wq, h, n.q, dModel)
	if cap(n.score) < nctx {
		n.score = make([]float64, nctx)
		n.attn = make([]float64, nctx)
	}
	n.score = n.score[:nctx]
	n.attn = n.attn[:nctx]
	scaleQ := 1 / math.Sqrt(float64(dModel))
	copy(n.dq, n.q)
	scale(n.dq, scaleQ)
	zero(n.score)
	addDots(keys, n.dq, n.score, dModel)
	for i := range nctx {
		n.score[i] += n.rel[nctx-1-i]
	}
	copy(n.attn, n.score)
	softmax(n.attn)
	for i := range n.ctxv {
		n.ctxv[i] = 0
	}
	mixRows(n.ctxv, vals, n.attn, dModel)
	copy(n.a, n.bo)
	for i := range n.a {
		n.a[i] += h[i]
	}
	addDots(n.wo, n.ctxv, n.a, dModel)
	copy(n.pre, n.b1)
	addDots(n.w1, n.a, n.pre, dModel)
	for i := range n.u {
		n.u[i] = math.Tanh(n.pre[i])
	}
	copy(n.y, n.b2)
	for i := range n.y {
		n.y[i] += n.a[i]
	}
	addDots(n.w2, n.u, n.y, ffnDim)
	copy(n.logit, n.by)
	addDots(n.wy, n.y, n.logit, dModel)
}

// train appends x to the context and takes one cross-entropy step toward next.
func (n *Transformer) train(x []float64, next rune) {
	if n.in == 0 || len(n.vocab) == 0 {
		return
	}
	n.project(x, n.hcur, n.kcur, n.vcur)
	n.ringH = appendRing(n.ringH, n.hcur)
	n.ringK = appendRing(n.ringK, n.kcur)
	n.ringV = appendRing(n.ringV, n.vcur)
	n.logits(n.hcur, n.ringK, n.ringV)
	if idx, ok := n.id[next]; ok {
		n.learn(x, idx)
	}
}

// learn applies the output gradient. Stored keys and values of earlier
// tokens stay as they were when those tokens were read.
func (n *Transformer) learn(x []float64, target int) {
	nctx := len(n.ringK) / dModel
	maxL := n.logit[0]
	for _, s := range n.logit[1:] {
		if s > maxL {
			maxL = s
		}
	}
	sum := 0.0
	for i, s := range n.logit {
		e := math.Exp(s - maxL)
		n.logit[i] = e
		sum += e
	}
	inv := 1 / sum
	for i := range n.logit {
		n.logit[i] *= inv
		if i == target {
			n.logit[i] -= 1
		}
	}
	zero(n.dy)
	mixRows(n.dy, n.wy, n.logit, dModel)
	clip(n.dy, clipNorm)

	zero(n.du)
	mixRows(n.du, n.w2, n.dy, ffnDim)
	for i := range n.du {
		n.du[i] *= 1 - n.u[i]*n.u[i]
	}
	copy(n.da, n.dy)
	mixRows(n.da, n.w1, n.du, dModel)

	zero(n.dctx)
	mixRows(n.dctx, n.wo, n.da, dModel)

	zero(n.score)
	addDots(n.ringV, n.dctx, n.score, dModel)
	var dotA float64
	for i := range nctx {
		dotA += n.attn[i] * n.score[i]
	}
	scaleQ := 1 / math.Sqrt(float64(dModel))
	for i := range nctx {
		n.score[i] = n.attn[i] * (n.score[i] - dotA)
		n.rel[nctx-1-i] -= learnRate * n.score[i]
	}
	zero(n.dq)
	mixRows(n.dq, n.ringK, n.score, dModel)
	scale(n.dq, scaleQ)

	last := nctx - 1
	ds := n.score[last] * scaleQ
	for i := range n.dk {
		n.dk[i] = ds * n.q[i]
		n.dv[i] = n.attn[last] * n.dctx[i]
	}
	copy(n.dh, n.da)
	mixRows(n.dh, n.wq, n.dq, dModel)
	mixRows(n.dh, n.wk, n.dk, dModel)
	mixRows(n.dh, n.wv, n.dv, dModel)

	for i := range n.by {
		n.by[i] -= learnRate * n.logit[i]
	}
	rank1(n.wy, n.logit, n.y, dModel, -learnRate)
	rank1(n.w2, n.dy, n.u, ffnDim, -learnRate)
	for i := range n.b2 {
		n.b2[i] -= learnRate * n.dy[i]
	}
	rank1(n.w1, n.du, n.a, dModel, -learnRate)
	for i := range n.b1 {
		n.b1[i] -= learnRate * n.du[i]
	}
	rank1(n.wo, n.da, n.ctxv, dModel, -learnRate)
	for i := range n.bo {
		n.bo[i] -= learnRate * n.da[i]
	}
	rank1(n.wq, n.dq, n.hcur, dModel, -learnRate)
	for i := range n.bq {
		n.bq[i] -= learnRate * n.dq[i]
	}
	rank1(n.wk, n.dk, n.hcur, dModel, -learnRate)
	for i := range n.bk {
		n.bk[i] -= learnRate * n.dk[i]
	}
	rank1(n.wv, n.dv, n.hcur, dModel, -learnRate)
	for i := range n.bv {
		n.bv[i] -= learnRate * n.dv[i]
	}
	rank1(n.wx, n.dh, x, n.in, -learnRate)
	for i := range n.bx {
		n.bx[i] -= learnRate * n.dh[i]
	}
}

// predictNet is the distribution over the next rune after reading ctx.
// An empty context is the softmax of the output bias.
func (g *Graph) predictNet(n *Transformer, ctx []int) []float64 {
	if n == nil || len(n.vocab) == 0 {
		return nil
	}
	if len(ctx) == 0 {
		p := make([]float64, len(n.by))
		copy(p, n.by)
		softmax(p)
		return p
	}
	if n.cache == nil {
		n.cache = make(map[int]proj)
	}
	n.genK = n.genK[:0]
	n.genV = n.genV[:0]
	var h []float64
	for _, id := range ctx {
		p := n.cached(g, id)
		n.genK = append(n.genK, p.k...)
		n.genV = append(n.genV, p.v...)
		h = p.h
	}
	n.logits(h, n.genK, n.genV)
	p := make([]float64, len(n.logit))
	copy(p, n.logit)
	softmax(p)
	return p
}

// cached returns the projection of id, computing it once for this network.
func (n *Transformer) cached(g *Graph, id int) proj {
	if p, ok := n.cache[id]; ok {
		return p
	}
	g.writeEmbed(id, n.embed)
	h := make([]float64, dModel)
	k := make([]float64, dModel)
	v := make([]float64, dModel)
	n.project(n.embed, h, k, v)
	p := proj{h: h, k: k, v: v}
	n.cache[id] = p
	return p
}

func softmax(v []float64) {
	if len(v) == 0 {
		return
	}
	maxL := v[0]
	for _, s := range v[1:] {
		if s > maxL {
			maxL = s
		}
	}
	sum := 0.0
	for i, s := range v {
		e := math.Exp(s - maxL)
		v[i] = e
		sum += e
	}
	inv := 1 / sum
	for i := range v {
		v[i] *= inv
	}
}

func zero(v []float64) {
	for i := range v {
		v[i] = 0
	}
}

func clip(v []float64, limit float64) {
	sum := dot(v, v)
	if sum == 0 || sum <= limit*limit {
		return
	}
	scale(v, limit/math.Sqrt(sum))
}

// logP is the log of a model probability, floored so a zero stays finite.
func logP(p float64) float64 {
	if p < 1e-12 {
		return math.Log(1e-12)
	}
	return math.Log(p)
}

// printTop writes the highest-ranking tokens.
func (g *Graph) printTop(w *os.File, k int) {
	if k <= 0 || len(g.rank) == 0 {
		return
	}
	order := make([]int, len(g.rank))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		return cmp.Compare(g.rank[b], g.rank[a])
	})
	if k > len(order) {
		k = len(order)
	}
	bw := bufio.NewWriter(w)
	for _, i := range order[:k] {
		fmt.Fprintf(bw, "  %.6f  %s\n", g.rank[i], strconv.Quote(string(g.token[i][:])))
	}
	bw.Flush()
}
