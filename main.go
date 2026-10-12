// Copyright 2026 The eigenvec Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command eigenvec builds an order-4 Markov graph of the characters in a
// text, computes its PageRank (the principal eigenvector of the damped
// transition matrix), and generates text from that ranking.
//
// A prompt is walked as a path of overlapping 4-character nodes. At every
// node the PageRank scores of the successors are normalized into a
// distribution over the next character and mixed, with a small weight,
// into an order-2 model P(next | the last two characters). Generation
// follows the current 4-gram. The order-2 conditional only tilts that
// choice. Each step then mixes the new node in the same way.
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
	order = 4
	// sumOrder is the order of the summed next-character model.
	sumOrder = 2
	// mix is the weight of a new visit in an order-2 conditional.
	// The previous conditional keeps the rest.
	mix = 0.15
	// tilt is the share of a draw taken from the order-2 conditional.
	// The rest follows the current 4-gram.
	tilt = 0.15
)

// context is the order-2 state the summed distribution conditions on.
type context [sumOrder]rune

// token is an order-4 Markov state: four successive characters.
type token [order]rune

// edge is a weighted transition to the shifted token.
type edge struct {
	to int
	w  float64
	ch rune
	// p is the successor's PageRank normalized over this node's successors.
	p float64
}

// Graph is the order-4 character graph and its PageRank vector.
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
		log.Fatal("text is shorter than an order-4 transition")
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
// breaks between speeches.
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

// buildGraph counts order-4 transitions. Node abcd links to bcde once for
// every time e follows the characters abcd. Windows overlap by three
// characters.
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

// shifted is the next order-4 token after appending ch.
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

// normalizeSuccessors stores, on each edge, the successor PageRank divided
// by the sum of successor PageRanks. That is the local next-character distribution.
func (g *Graph) normalizeSuccessors() {
	for i, outs := range g.outs {
		sum := 0.0
		for _, e := range outs {
			sum += g.rank[e.to]
		}
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		for j := range outs {
			g.outs[i][j].p = g.rank[outs[j].to] * inv
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
// Each sample follows the current 4-gram, tilted by that model, then
// steps to the successor and mixes the new node in.
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
		log.Fatal("prompt shares no order-4 token with the text")
	}

	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	out := append([]rune(nil), prompt...)
	for len(out) < len(prompt)+n {
		outs := g.outs[cur]
		if len(outs) == 0 {
			break
		}
		e := sample(model[contextOf(g.token[cur])], outs, rng)
		out = append(out, e.ch)
		cur = e.to
		g.addNext(model, cur)
	}
	return out
}

// edgeWeight is the current 4-gram's edge probability, tilted by the
// order-2 conditional. The conditional is renormalized over the legal
// successors. A character the conditional has not seen keeps its 4-gram
// share.
func edgeWeight(e edge, dist map[rune]float64, priorSum float64) float64 {
	w := (1 - tilt) * e.p
	if priorSum > 0 {
		w += tilt * dist[e.ch] / priorSum
	}
	return w
}

// sample draws a successor of the current 4-gram. The order-2 conditional
// only tilts the 4-gram probabilities.
func sample(dist map[rune]float64, outs []edge, rng *rand.Rand) edge {
	priorSum := 0.0
	for _, e := range outs {
		if p := dist[e.ch]; p > 0 {
			priorSum += p
		}
	}
	total := 0.0
	for _, e := range outs {
		total += edgeWeight(e, dist, priorSum)
	}
	if total == 0 {
		return outs[rng.IntN(len(outs))]
	}
	r := rng.Float64() * total
	acc := 0.0
	var last edge
	seen := false
	for _, e := range outs {
		w := edgeWeight(e, dist, priorSum)
		if w <= 0 {
			continue
		}
		last = e
		seen = true
		acc += w
		if r <= acc {
			return e
		}
	}
	if seen {
		return last
	}
	return outs[len(outs)-1]
}

// printTop writes the highest-ranking order-4 tokens.
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
