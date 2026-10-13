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
// into an order-2 conditional kept by a program on a Turing machine's tape.
// The program decays that context's weights, adds the new visit, and
// normalizes. Value iteration stores, on each node, how likely a rollout
// from there stays under the tempered token distribution. Each new
// character is drawn from that distribution, tilted by the table the
// program wrote, and multiplied by the successor's continuation value.
// Each committed step runs the program again.
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
	// horizon is how many edges a continuation value looks ahead.
	horizon = 16
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
	// cont is the per-step probability of a horizon-step rollout from this node.
	cont []float64
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
	g.scoreContinuations()

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

// generate walks the prompt and runs the tape program on each token, which
// keeps the order-2 table. Each character is drawn from the current token,
// tilted by that table and multiplied by the successor's continuation value.
// The walk then steps to that successor and runs the program on the new node.
func (g *Graph) generate(prompt []rune, n int, seed uint64) []rune {
	if len(prompt) < order {
		log.Fatalf("prompt must contain at least %d characters", order)
	}
	m := tiltProgram()
	cur := -1
	for i := 0; i+order <= len(prompt); i++ {
		node, ok := g.id[tokenFrom(prompt[i:i+order])]
		if !ok {
			continue
		}
		m.observe(g, node)
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
		e := g.draw(m.conditional(contextOf(g.token[cur])), cur, rng)
		out = append(out, e.ch)
		cur = e.to
		m.observe(g, cur)
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

// scoreContinuations stores the per-step probability of a horizon-step
// rollout under the tempered token distribution. A missing step, including
// every step from a node with no outgoing edge, scores as a near-zero probability.
func (g *Graph) scoreContinuations() {
	n := len(g.outs)
	pi := make([][]float64, n)
	logpi := make([][]float64, n)
	for i, outs := range g.outs {
		if len(outs) == 0 {
			continue
		}
		grams := temper(outs)
		sum := 0.0
		for _, p := range grams {
			sum += p
		}
		if sum == 0 {
			continue
		}
		pi[i] = grams
		logpi[i] = make([]float64, len(grams))
		for j, p := range grams {
			logpi[i][j] = logP(p)
		}
	}
	floor := logP(0)
	V := make([]float64, n)
	next := make([]float64, n)
	for step := 1; step <= horizon; step++ {
		for i, outs := range g.outs {
			if len(outs) == 0 || pi[i] == nil {
				next[i] = float64(step) * floor
				continue
			}
			s := 0.0
			for j, e := range outs {
				s += pi[i][j] * (logpi[i][j] + V[e.to])
			}
			next[i] = s
		}
		V, next = next, V
	}
	g.cont = make([]float64, n)
	inv := 1 / float64(horizon)
	for i, v := range V {
		g.cont[i] = math.Exp(v * inv)
	}
}

// draw samples an outgoing edge. The tempered, tilted token distribution is
// multiplied by the continuation value of the node each edge enters.
func (g *Graph) draw(dist map[rune]float64, cur int, rng *rand.Rand) edge {
	outs := g.outs[cur]
	weights := edgeWeights(dist, outs)
	total := 0.0
	for i, e := range outs {
		weights[i] *= g.cont[e.to]
		total += weights[i]
	}
	if total == 0 {
		return outs[rng.IntN(len(outs))]
	}
	r := rng.Float64() * total
	acc := 0.0
	last := outs[0]
	for i, e := range outs {
		if weights[i] <= 0 {
			continue
		}
		last = e
		acc += weights[i]
		if r <= acc {
			return e
		}
	}
	return last
}

// tmKey is the finite control of one Turing transition: a state and the symbol read.
type tmKey struct {
	state int
	read  rune
}

// tmStep is the write, the head move, and the next state.
type tmStep struct {
	write rune
	move  int
	next  int
}

// tape is an unbounded work tape. Cells that have never been written are blank.
// The head may move either way without a fixed bound.
type tape struct {
	right []rune
	left  []rune
	head  int
}

const blank rune = 0

func (t *tape) read() rune {
	if t.head >= 0 {
		if t.head >= len(t.right) {
			return blank
		}
		return t.right[t.head]
	}
	i := -t.head - 1
	if i >= len(t.left) {
		return blank
	}
	return t.left[i]
}

func (t *tape) write(r rune) {
	if t.head >= 0 {
		for len(t.right) <= t.head {
			t.right = append(t.right, blank)
		}
		t.right[t.head] = r
		return
	}
	i := -t.head - 1
	for len(t.left) <= i {
		t.left = append(t.left, blank)
	}
	t.left[i] = r
}

func (t *tape) move(d int) { t.head += d }

func (t *tape) at(i int) rune {
	h := t.head
	t.head = i
	r := t.read()
	t.head = h
	return r
}

func (t *tape) writeAt(i int, r rune) {
	h := t.head
	t.head = i
	t.write(r)
	t.head = h
}

// Machine is a Turing machine. Any machine is a finite transition table on
// this tape, so the type is a universal interpreter. accept is the halting
// state of a decider, or -1 when the machine does not accept.
type Machine struct {
	state  int
	accept int
	tape   tape
	delta  map[tmKey]tmStep
}

func newMachine() *Machine {
	return &Machine{accept: -1, delta: make(map[tmKey]tmStep)}
}

func (m *Machine) set(state int, read, write rune, move, next int) {
	m.delta[tmKey{state, read}] = tmStep{write, move, next}
}

// step applies one transition. A missing transition halts the machine.
func (m *Machine) step() bool {
	s, ok := m.delta[tmKey{m.state, m.tape.read()}]
	if !ok {
		return false
	}
	m.tape.write(s.write)
	m.tape.move(s.move)
	m.state = s.next
	return true
}

// run steps until the machine halts, enters its accept state, or exhausts limit.
func (m *Machine) run(limit int) bool {
	for range limit {
		if m.accept >= 0 && m.state == m.accept {
			return true
		}
		if !m.step() {
			return m.accept >= 0 && m.state == m.accept
		}
	}
	return false
}

// bias rescales a candidate character from the symbol under the head.
// It is always positive, so a legal edge stays legal.
func (m *Machine) bias(ch rune) float64 {
	if m == nil {
		return 1
	}
	s := int(m.tape.read()) + 3*m.state + int(ch)
	if s < 0 {
		s = -s
	}
	return 1 + 0.05*float64(s%7)
}

// universal23 is Wolfram's 2-state 3-symbol machine. State 0 is A and state 1
// is B. Alex Smith proved this table universal for a repeating background on
// the infinite tape.
func universal23() *Machine {
	m := newMachine()
	// A
	m.set(0, 0, 1, +1, 1)
	m.set(0, 1, 2, -1, 0)
	m.set(0, 2, 1, -1, 0)
	// B
	m.set(1, 0, 2, -1, 0)
	m.set(1, 1, 2, +1, 1)
	m.set(1, 2, 0, +1, 0)
	return m
}

// The tilt program occupies cells [0, progLen). Arguments for one visit sit
// on the left of the origin. The order-2 table grows to the right of the program.
const (
	// opDecay scales the selected context's weights by 1-mix.
	opDecay rune = 1
	// opAdds folds the visit's successor weights into that context.
	opAdds rune = 2
	// opNorm makes the context's weights sum to one.
	opNorm rune = 3
	// opHalt stops the program.
	opHalt rune = 4

	progLen    = 4
	tableCount = progLen
	tableData  = progLen + 1

	argC0    = -1
	argC1    = -2
	argN     = -3
	argPairs = -4

	// weightScale is the fixed-point unit of a weight stored on the tape.
	weightScale int64 = 1_000_000
)

// tiltProgram writes the order-2 update on the tape and leaves the head on
// its first instruction. The table starts empty.
func tiltProgram() *Machine {
	m := newMachine()
	m.tape.writeAt(0, opDecay)
	m.tape.writeAt(1, opAdds)
	m.tape.writeAt(2, opNorm)
	m.tape.writeAt(3, opHalt)
	return m
}

// observe runs the tape program on this node's successor distribution.
// The context is the token's last two characters.
func (m *Machine) observe(g *Graph, node int) {
	ctx := contextOf(g.token[node])
	outs := g.outs[node]
	m.tape.writeAt(argC0, ctx[0])
	m.tape.writeAt(argC1, ctx[1])
	m.tape.writeAt(argN, rune(len(outs)))
	for i, e := range outs {
		base := argPairs - 2*i
		m.tape.writeAt(base, e.ch)
		m.tape.writeAt(base-1, fixWeight(e.p))
	}
	m.runTilt()
}

// conditional reads the order-2 distribution the program stored for c.
func (m *Machine) conditional(c context) map[rune]float64 {
	start, nw, ok := m.findContext(c[0], c[1])
	if !ok || nw == 0 {
		return nil
	}
	dist := make(map[rune]float64, nw)
	for i := 0; i < nw; i++ {
		ch := m.tape.at(start + 3 + 2*i)
		w := m.tape.at(start + 3 + 2*i + 1)
		if w != 0 {
			dist[ch] = float64(w) / float64(weightScale)
		}
	}
	normalizeDist(dist)
	return dist
}

// runTilt fetches each instruction of the program and stops at halt.
func (m *Machine) runTilt() {
	for pc := 0; ; pc++ {
		m.tape.head = pc
		switch m.tape.read() {
		case opDecay:
			m.decayContext()
		case opAdds:
			m.addVisit()
		case opNorm:
			m.normContext()
		default:
			return
		}
	}
}

func (m *Machine) decayContext() {
	start, nw, ok := m.findContext(m.tape.at(argC0), m.tape.at(argC1))
	if !ok {
		return
	}
	keep := int64(math.Round((1 - mix) * 100))
	for i := 0; i < nw; i++ {
		wpos := start + 3 + 2*i + 1
		w := mulDiv(int64(m.tape.at(wpos)), keep, 100)
		m.tape.writeAt(wpos, rune(w))
	}
}

func (m *Machine) addVisit() {
	c0, c1 := m.tape.at(argC0), m.tape.at(argC1)
	start, _, ok := m.findContext(c0, c1)
	if !ok {
		start = m.allocContext(c0, c1)
	}
	part := int64(math.Round(mix * 100))
	n := int(m.tape.at(argN))
	for i := 0; i < n; i++ {
		base := argPairs - 2*i
		ch := m.tape.at(base)
		add := mulDiv(int64(m.tape.at(base-1)), part, 100)
		m.addWeight(start, ch, add)
	}
}

func (m *Machine) normContext() {
	start, nw, ok := m.findContext(m.tape.at(argC0), m.tape.at(argC1))
	if !ok || nw == 0 {
		return
	}
	sum := int64(0)
	for i := 0; i < nw; i++ {
		sum += int64(m.tape.at(start + 3 + 2*i + 1))
	}
	if sum == 0 {
		return
	}
	acc := int64(0)
	last := -1
	for i := 0; i < nw; i++ {
		wpos := start + 3 + 2*i + 1
		w := int64(m.tape.at(wpos))
		if w > 0 {
			last = wpos
		}
		next := w * weightScale / sum
		acc += next
		m.tape.writeAt(wpos, rune(next))
	}
	if last >= 0 {
		m.tape.writeAt(last, rune(int64(m.tape.at(last))+(weightScale-acc)))
	}
}

// findContext returns the cell where (c0, c1) is stored. The third cell of
// a record is the number of (character, weight) pairs that follow.
func (m *Machine) findContext(c0, c1 rune) (start, nw int, ok bool) {
	n := int(m.tape.at(tableCount))
	pos := tableData
	for i := 0; i < n; i++ {
		pairs := int(m.tape.at(pos + 2))
		if m.tape.at(pos) == c0 && m.tape.at(pos+1) == c1 {
			return pos, pairs, true
		}
		pos += 3 + 2*pairs
	}
	return pos, 0, false
}

func (m *Machine) allocContext(c0, c1 rune) int {
	pos, _, ok := m.findContext(c0, c1)
	if ok {
		return pos
	}
	m.tape.writeAt(pos, c0)
	m.tape.writeAt(pos+1, c1)
	m.tape.writeAt(pos+2, 0)
	m.tape.writeAt(tableCount, m.tape.at(tableCount)+1)
	return pos
}

func (m *Machine) addWeight(start int, ch rune, add int64) {
	nw := int(m.tape.at(start + 2))
	for i := 0; i < nw; i++ {
		cpos := start + 3 + 2*i
		if m.tape.at(cpos) == ch {
			wpos := cpos + 1
			m.tape.writeAt(wpos, rune(int64(m.tape.at(wpos))+add))
			return
		}
	}
	at := start + 3 + 2*nw
	m.shiftTable(at, 2)
	m.tape.writeAt(at, ch)
	m.tape.writeAt(at+1, rune(add))
	m.tape.writeAt(start+2, rune(nw+1))
}

// shiftTable moves the table cells at and after from, n cells to the right.
func (m *Machine) shiftTable(from, n int) {
	end := tableData
	contexts := int(m.tape.at(tableCount))
	for i := 0; i < contexts; i++ {
		end += 3 + 2*int(m.tape.at(end+2))
	}
	for i := end - 1; i >= from; i-- {
		m.tape.writeAt(i+n, m.tape.at(i))
	}
}

func fixWeight(p float64) rune {
	if p <= 0 {
		return 0
	}
	return rune(math.Round(p * float64(weightScale)))
}

func mulDiv(w, num, den int64) int64 {
	if w <= 0 || num <= 0 {
		return 0
	}
	return (w*num + den/2) / den
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
