// Copyright 2026 The eigenvec Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command eigenvec builds an order-6 Markov graph of the characters in a
// text, computes its PageRank (the principal eigenvector of the damped
// transition matrix), and generates text from that ranking.
//
// A prompt is walked as a path of overlapping 6-character nodes. At every
// node the count-weighted PageRank of the successors is normalized into a
// distribution over the next character and mixed, with a small weight,
// into an order-2 model P(next | the last two characters). Each new
// character is chosen by Monte Carlo tree search. That distribution,
// softened by a temperature slightly above 1 and tilted by the order-2
// conditional, is the prior. A rollout scores a path by the geometric mean
// of those probabilities, and the character is drawn from the visit counts.
// Each committed step then mixes the new node in the same way.
package main

import (
	"bufio"
	"cmp"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"strings"
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
	// sumOrder is the order of the summed next-character model.
	sumOrder = 2
	// mix is the weight of a new visit in an order-2 conditional.
	// The previous conditional keeps the rest.
	mix = 0.15
	// tilt is the share of a draw taken from the order-2 conditional.
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
)

// context is the order-2 state the summed distribution conditions on.
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
}

func main() {
	file := flag.String("file", "pg100.txt", "text to model")
	prompt := flag.String("prompt", "To be, or not to be", "prompt that starts the walk")
	n := flag.Int("n", 400, "characters to generate after the prompt")
	seed := flag.Uint64("seed", 1, "sampling seed")
	top := flag.Int("top", 12, "how many PageRank tokens to print")
	flag.Parse()

	text, err := os.ReadFile(*file)
	if err != nil {
		log.Fatal(err)
	}
	runes := []rune(prepare(string(text)))
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
// The result is the unique positive eigenvector of the Google matrix,
// normalized to sum to one.
func (g *Graph) rankPages(damp, tol float64) int {
	n := len(g.outs)
	if n == 0 {
		return 0
	}
	outw := make([]float64, n)
	for i, outs := range g.outs {
		for _, e := range outs {
			outw[i] += e.w
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
			for _, e := range outs {
				next[e.to] += scale * e.w
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

// addNext mixes this node's next-character distribution into the order-2
// context given by the token's last two characters. A new visit has weight
// mix and the previous conditional keeps the rest. The result is normalized
// so its entries sum to one.
func (g *Graph) addNext(model map[context]map[rune]float64, node int) {
	ctx := contextOf(g.token[node])
	dist := model[ctx]
	if dist == nil {
		dist = make(map[rune]float64)
		model[ctx] = dist
	}
	keep := 1 - mix
	for ch := range dist {
		dist[ch] *= keep
	}
	for _, e := range g.outs[node] {
		dist[e.ch] += mix * e.p
	}
	normalizeDist(dist)
}

// contextOf is the order-2 state a token predicts from: its last two characters.
func contextOf(t token) context {
	var c context
	copy(c[:], t[order-sumOrder:])
	return c
}

// normalizeDist scales dist so its entries sum to one.
func normalizeDist(dist map[rune]float64) {
	sum := 0.0
	for _, p := range dist {
		sum += p
	}
	if sum == 0 {
		return
	}
	inv := 1 / sum
	for ch := range dist {
		dist[ch] *= inv
	}
}

// generate walks the prompt, mixing each token into the order-2 model.
// Each character is chosen by Monte Carlo tree search from the current
// token, then the walk steps to that successor and mixes the new node in.
func (g *Graph) generate(prompt []rune, n int, seed uint64) []rune {
	if len(prompt) < order {
		log.Fatalf("prompt must contain at least %d characters", order)
	}
	model := make(map[context]map[rune]float64)
	cur := -1
	for i := 0; i+order <= len(prompt); i++ {
		node, ok := g.id[tokenFrom(prompt[i:i+order])]
		if !ok {
			continue
		}
		g.addNext(model, node)
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
		e := g.search(model, cur, rng)
		out = append(out, e.ch)
		cur = e.to
		g.addNext(model, cur)
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

// sample draws a successor of the current token. The token distribution is
// softened by temp, then the order-2 conditional tilts it.
func sample(dist map[rune]float64, outs []edge, rng *rand.Rand) edge {
	e, _ := sampleProb(dist, outs, rng)
	return e
}

// sampleProb draws a successor and returns its share of the model distribution.
func sampleProb(dist map[rune]float64, outs []edge, rng *rand.Rand) (edge, float64) {
	weights := edgeWeights(dist, outs)
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
func (g *Graph) search(model map[context]map[rune]float64, cur int, rng *rand.Rand) edge {
	root := g.explore(model, cur, rng)
	return root.draw(rng)
}

// explore runs the simulations from cur and returns the search root.
func (g *Graph) explore(model map[context]map[rune]float64, cur int, rng *rand.Rand) *mctsNode {
	root := &mctsNode{id: cur}
	g.expand(root, model)
	if len(root.children) <= 1 {
		for i := range root.children {
			root.children[i].visits = 1
		}
		return root
	}
	for range mctsSims {
		g.playout(root, model, rng)
	}
	return root
}

// expand attaches one child per outgoing edge. The prior is the model's
// sampling distribution at this node.
func (g *Graph) expand(n *mctsNode, model map[context]map[rune]float64) {
	if n.expanded {
		return
	}
	n.expanded = true
	outs := g.outs[n.id]
	if len(outs) == 0 {
		return
	}
	weights := edgeWeights(model[contextOf(g.token[n.id])], outs)
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
func (g *Graph) playout(root *mctsNode, model map[context]map[rune]float64, rng *rand.Rand) {
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
		g.expand(node, model)
	}
	id := node.id
	for len(seq) < mctsDepth {
		outs := g.outs[id]
		if len(outs) == 0 {
			break
		}
		e, p := sampleProb(model[contextOf(g.token[id])], outs, rng)
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

// logP is the log of a model probability, floored so a zero stays finite.
func logP(p float64) float64 {
	if p < 1e-12 {
		return math.Log(1e-12)
	}
	return math.Log(p)
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
