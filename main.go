// Copyright 2026 The eigenvec Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command eigenvec builds an order-6 Markov graph of the characters in a
// text, computes its PageRank (the principal eigenvector of the damped
// transition matrix), and generates text from that ranking.
//
// A prompt is walked as a path of overlapping 6-character nodes. At every
// node the count-weighted PageRank of the successors is normalized into a
// distribution over the next character. An order-2 network embeds the last
// two runes into 256 inputs and predicts the next rune, with one output for
// each rune that occurs. One gradient step per committed visit trains it on
// that node's successor distribution. Each new character is chosen by Monte
// Carlo tree search. The tempered token distribution is tilted by the network.
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
	// sumOrder is the order of the next-character network.
	sumOrder = 2
	// learnRate is the step size of one realtime update of the network.
	learnRate = 0.1
	// inputs is the width of the order-2 network. Each of the last two runes
	// is embedded into this many values, and each rune has one output.
	inputs = 256
	// tilt is the share of a draw taken from the order-2 network.
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

// context is the order-2 state the network conditions on.
type context [sumOrder]rune

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

// contextOf is the order-2 state a token predicts from: its last two characters.
func contextOf(t token) context {
	var c context
	copy(c[:], t[order-sumOrder:])
	return c
}

// generate walks the prompt and trains the order-2 network on each token.
// Each character is chosen by Monte Carlo tree search from the current
// token, tilted by the network. The walk then steps to that successor and
// trains on the new node. A rollout only reads the network.
func (g *Graph) generate(prompt []rune, n int, seed uint64) []rune {
	if len(prompt) < order {
		log.Fatalf("prompt must contain at least %d characters", order)
	}
	net := newNet(seed, g.alphabet())
	cur := -1
	for i := 0; i+order <= len(prompt); i++ {
		node, ok := g.id[tokenFrom(prompt[i:i+order])]
		if !ok {
			continue
		}
		g.learnNode(net, node)
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
		e := g.search(net, cur, rng)
		out = append(out, e.ch)
		cur = e.to
		g.learnNode(net, cur)
	}
	return out
}

// edgeWeight is the current token's edge probability, tilted by the
// order-2 conditional. The conditional is renormalized over the legal
// successors. A character the conditional has not seen keeps its share of
// the token distribution.
func edgeWeight(e edge, dist map[rune]float64, priorSum float64) float64 {
	w := (1 - tilt) * e.p
	if priorSum > 0 {
		w += tilt * dist[e.ch] / priorSum
	}
	return w
}

// edgeWeights is the tempered token distribution tilted by the order-2
// conditional, one weight per outgoing edge.
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
type mctsNode struct {
	id       int
	visits   float64
	expanded bool
	children []mctsChild
}

// search chooses the next edge by Monte Carlo tree search and draws it
// from the root visit counts. A node with one continuation is returned as it is.
// Predictions made during the search are cached. The network is only read.
func (g *Graph) search(net *Net, cur int, rng *rand.Rand) edge {
	if net != nil {
		net.cache = make(map[[2]rune][]float64)
	}
	root := g.explore(net, cur, rng)
	return root.draw(rng)
}

// explore runs the simulations from cur and returns the search root.
// The network is only read.
func (g *Graph) explore(net *Net, cur int, rng *rand.Rand) *mctsNode {
	root := &mctsNode{id: cur}
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
// distribution at this node.
func (g *Graph) expand(n *mctsNode, net *Net) {
	if n.expanded {
		return
	}
	n.expanded = true
	outs := g.outs[n.id]
	if len(outs) == 0 {
		return
	}
	weights := g.priorWeights(g.distAt(net, n.id), n.id, outs)
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
func (g *Graph) playout(root *mctsNode, net *Net, rng *rand.Rand) {
	node := root
	path := make([]*mctsChild, 0, mctsDepth)
	seq := make([]float64, 0, mctsDepth)
	for len(path) < mctsDepth && node.expanded && len(node.children) > 0 {
		c := node.pick()
		if c.node == nil {
			c.node = &mctsNode{id: c.edge.to}
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
	for len(seq) < mctsDepth {
		outs := g.outs[id]
		if len(outs) == 0 {
			break
		}
		e, p := g.sampleProb(g.distAt(net, id), id, outs, rng)
		id = e.to
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
func (g *Graph) distAt(net *Net, id int) map[rune]float64 {
	if net == nil || id < 0 || id >= len(g.token) || id >= len(g.outs) {
		return nil
	}
	ctx := contextOf(g.token[id])
	sm := net.predict(ctx[0], ctx[1])
	outs := g.outs[id]
	dist := make(map[rune]float64, len(outs))
	for _, e := range outs {
		if i, ok := net.id[e.ch]; ok {
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

// Net predicts the next rune from the last two runes.
// e0 and e1 embed the older and the newer character into inputs values.
// w maps those inputs to one logit per rune in vocab.
type Net struct {
	id     map[rune]int
	vocab  []rune
	e0, e1 [][]float64
	w      [][]float64
	b      []float64
	// cache holds predictions while the weights stay fixed.
	cache map[[2]rune][]float64
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

// newNet draws a small deterministic initialization from seed.
// vocab is the runes the network can read and predict. Duplicates are dropped.
// Zero weights would leave the gradient at zero.
func newNet(seed uint64, runes []rune) *Net {
	vocab := uniqueRunes(runes)
	n := &Net{
		id:    make(map[rune]int, len(vocab)),
		vocab: vocab,
		e0:    make([][]float64, len(vocab)),
		e1:    make([][]float64, len(vocab)),
		w:     make([][]float64, len(vocab)),
		b:     make([]float64, len(vocab)),
	}
	for i, r := range vocab {
		n.id[r] = i
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	scale := 1 / math.Sqrt(float64(inputs))
	for i := range vocab {
		n.e0[i] = make([]float64, inputs)
		n.e1[i] = make([]float64, inputs)
		n.w[i] = make([]float64, inputs)
		for j := range inputs {
			n.e0[i][j] = (rng.Float64()*2 - 1) * scale
			n.e1[i][j] = (rng.Float64()*2 - 1) * scale
			n.w[i][j] = (rng.Float64()*2 - 1) * scale
		}
	}
	return n
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

// addEmbed adds the embedding of r into h. A rune outside vocab adds nothing.
func addEmbed(table [][]float64, id map[rune]int, r rune, h []float64) {
	i, ok := id[r]
	if !ok {
		return
	}
	row := table[i]
	for j := range h {
		h[j] += row[j]
	}
}

// hidden is tanh of the summed embeddings, the inputs for this context.
func (n *Net) hidden(c0, c1 rune) []float64 {
	h := make([]float64, inputs)
	addEmbed(n.e0, n.id, c0, h)
	addEmbed(n.e1, n.id, c1, h)
	for i := range h {
		h[i] = math.Tanh(h[i])
	}
	return h
}

// probs is the softmax over the rune logits for this hidden state.
func (n *Net) probs(h []float64) []float64 {
	nv := len(n.vocab)
	p := make([]float64, nv)
	if nv == 0 {
		return p
	}
	maxL := math.Inf(-1)
	logit := make([]float64, nv)
	for o := range logit {
		s := n.b[o]
		row := n.w[o]
		for i := range row {
			s += row[i] * h[i]
		}
		logit[o] = s
		if s > maxL {
			maxL = s
		}
	}
	sum := 0.0
	for o := range p {
		p[o] = math.Exp(logit[o] - maxL)
		sum += p[o]
	}
	inv := 1 / sum
	for o := range p {
		p[o] *= inv
	}
	return p
}

// predict is the distribution over the next rune, indexed like vocab.
// A cache hit is returned while the weights are frozen.
func (n *Net) predict(c0, c1 rune) []float64 {
	key := [2]rune{c0, c1}
	if n.cache != nil {
		if p, ok := n.cache[key]; ok {
			return p
		}
	}
	p := n.probs(n.hidden(c0, c1))
	if n.cache != nil {
		n.cache[key] = p
	}
	return p
}

// learn takes one cross-entropy step so this context moves toward target.
// target is a distribution over vocab. The prediction cache is dropped.
func (n *Net) learn(c0, c1 rune, target []float64) {
	n.cache = nil
	if len(n.vocab) == 0 {
		return
	}
	h := n.hidden(c0, c1)
	p := n.probs(h)
	dlogit := make([]float64, len(n.vocab))
	for o := range dlogit {
		t := 0.0
		if o < len(target) {
			t = target[o]
		}
		dlogit[o] = p[o] - t
	}
	// dh uses the weights from before they change.
	dh := make([]float64, inputs)
	for o := range dlogit {
		d := dlogit[o]
		if d == 0 {
			continue
		}
		row := n.w[o]
		for i := range dh {
			dh[i] += d * row[i]
		}
	}
	for i := range dh {
		dh[i] *= 1 - h[i]*h[i]
	}
	for o := range dlogit {
		d := dlogit[o]
		if d == 0 {
			continue
		}
		n.b[o] -= learnRate * d
		row := n.w[o]
		step := learnRate * d
		for i := range row {
			row[i] -= step * h[i]
		}
	}
	stepEmbed := func(table [][]float64, r rune) {
		i, ok := n.id[r]
		if !ok {
			return
		}
		row := table[i]
		for j := range dh {
			row[j] -= learnRate * dh[j]
		}
	}
	stepEmbed(n.e0, c0)
	stepEmbed(n.e1, c1)
}

// runeTarget is this node's successor distribution over the network's runes.
func (g *Graph) runeTarget(net *Net, id int) []float64 {
	t := make([]float64, len(net.vocab))
	if id < 0 || id >= len(g.outs) {
		return t
	}
	sum := 0.0
	for _, e := range g.outs[id] {
		if e.p <= 0 {
			continue
		}
		i, ok := net.id[e.ch]
		if !ok {
			continue
		}
		t[i] += e.p
		sum += e.p
	}
	if sum == 0 || math.Abs(sum-1) <= 1e-12 {
		return t
	}
	inv := 1 / sum
	for i := range t {
		t[i] *= inv
	}
	return t
}

// learnNode trains the network on this node's successor distribution.
// A node with no successor mass is left alone.
func (g *Graph) learnNode(net *Net, id int) {
	if net == nil || id < 0 || id >= len(g.token) || id >= len(g.outs) {
		return
	}
	target := g.runeTarget(net, id)
	sum := 0.0
	for _, p := range target {
		sum += p
	}
	if sum == 0 {
		return
	}
	ctx := contextOf(g.token[id])
	net.learn(ctx[0], ctx[1], target)
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
