// Copyright 2026 The eigenvec Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadGutenbergTakesFirstBooks(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "txt-files.tar.zip")
	var tarBuf bytes.Buffer
	tw := tar.NewWriter(&tarBuf)
	add := func(name, body string) {
		t.Helper()
		b := []byte(body)
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(b))}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write(b); err != nil {
			t.Fatal(err)
		}
	}
	add("cache/epub/1/notes.dat", "skip me")
	add("cache/epub/1/pg1.txt", "HEADER\n*** START OF THE PROJECT GUTENBERG EBOOK X\nAlpha.\n*** END OF THE PROJECT GUTENBERG EBOOK X\nlicense")
	add("cache/epub/2/", "")
	add("cache/epub/2/pg2.txt", "Beta.")
	add("cache/epub/3/pg3.txt", "Gamma.")
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	w, err := zw.Create("txt-files.tar")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(tarBuf.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	got, err := readGutenberg(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got != "Alpha.\nBeta." {
		t.Fatalf("readGutenberg = %q", got)
	}
	if _, err := readGutenberg(path, 4); err == nil {
		t.Fatal("expected an error when fewer than 4 books are present")
	}
}

func TestPageRankKnownGraph(t *testing.T) {
	// Weighted graph from the alixaxel/pagerank README, solved to 1e-15.
	// Node 4 is dangling. Stored weights are the exponentials of the README
	// weights, because rankPages takes the log before it walks the links.
	g := &Graph{outs: make([][]edge, 4)}
	g.outs[0] = []edge{{to: 1, w: math.Exp(1)}, {to: 2, w: math.Exp(2)}}
	g.outs[1] = []edge{{to: 2, w: math.Exp(3)}, {to: 3, w: math.Exp(4)}}
	g.outs[2] = []edge{{to: 0, w: math.Exp(5)}}
	iters := g.rankPages(0.85, 1e-15)
	if iters <= 1 || iters >= maxRankIter {
		t.Fatalf("iterations = %d", iters)
	}
	want := []float64{
		0.34983786660804217,
		0.16887328140050842,
		0.3295121342115437,
		0.15177671777990553,
	}
	sum := 0.0
	for i, got := range g.rank {
		sum += got
		if math.Abs(got-want[i]) > 1e-12 {
			t.Errorf("rank[%d] = %.15f, want %.15f", i, got, want[i])
		}
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("rank sums to %.15f", sum)
	}
}

func TestGenerateStaysOnObservedEdges(t *testing.T) {
	src := []rune("To be or not to be, that is the question. To be a bee.")
	g := buildGraph(src)
	g.rankPages(damp, 1e-14)
	g.normalizeSuccessors()

	prompt := []rune("To be or")
	out := g.generate(prompt, 80, 1)
	if !strings.HasPrefix(string(out), string(prompt)) {
		t.Fatalf("output %q does not start with the prompt", string(out))
	}
	if len(out) < len(prompt)+2 {
		t.Fatalf("output too short: %q", string(out))
	}
	for i := 0; i+order < len(out); i++ {
		from, ok := g.id[tokenFrom(out[i:i+order])]
		if !ok {
			t.Fatalf("token %q missing at %d in %q", string(out[i:i+order]), i, string(out))
		}
		ch := out[i+order]
		found := false
		for _, e := range g.outs[from] {
			if e.ch == ch {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("no edge %q -%q at %d in %q", string(out[i:i+order]), string(ch), i, string(out))
		}
	}
}

func TestPrepareStripsHeaderAndSpeechBreaks(t *testing.T) {
	raw := "header\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\nHAMLET.\nTo be.\n\nKING.\nAye.\n\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense"
	got := prepare(raw)
	if strings.Contains(got, "PROJECT GUTENBERG") || strings.Contains(got, "license") || strings.Contains(got, "header") {
		t.Fatalf("boilerplate remains: %q", got)
	}
	if strings.Contains(got, ".\n") || strings.Contains(got, "\n\n") {
		t.Fatalf("speech breaks remain: %q", got)
	}
	want := "HAMLET. To be. \nKING. Aye."
	if got != want {
		t.Fatalf("prepare() = %q, want %q", got, want)
	}
}

func TestPrepareStripsIndentAndLineNumbers(t *testing.T) {
	raw := "" +
		"  Thy unused beauty must be tombed with thee,\n" +
		"And set dissension ’twixt the son and sire;         1160\n" +
		"                    5\n" +
		"HAMLET.\n" +
		"   ’One fair daughter, and no more,\n"
	got := prepare(raw)
	want := "" +
		"Thy unused beauty must be tombed with thee,\n" +
		"And set dissension ’twixt the son and sire;\n" +
		"HAMLET. ’One fair daughter, and no more,"
	if got != want {
		t.Fatalf("prepare() = %q, want %q", got, want)
	}
}

func TestSIMDKernelsMatchExactSums(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 7, 8, 9, 31, 32, 100}
	for _, n := range lengths {
		a := make([]float64, n)
		b := make([]float64, n)
		for i := range a {
			a[i] = float64(i + 1)
			b[i] = float64((i%5)+1) / 2
		}
		want := float64(n * (n + 1) / 2)
		if got := dot(a, ones(n)); got != want {
			t.Fatalf("dot len %d = %v, want %v", n, got, want)
		}
		sq := float64(n * (n + 1) * (2*n + 1) / 6)
		if got := dot(a, a); got != sq {
			t.Fatalf("sum of squares len %d = %v, want %v", n, got, sq)
		}
		got := append([]float64(nil), a...)
		addScaled(got, b, -0.5)
		for i := range a {
			want := a[i] - 0.5*b[i]
			if got[i] != want {
				t.Fatalf("addScaled len %d [%d] = %v, want %v", n, i, got[i], want)
			}
		}
		scaled := append([]float64(nil), a...)
		scale(scaled, 0.5)
		for i := range a {
			if scaled[i] != a[i]*0.5 {
				t.Fatalf("scale len %d [%d] = %v, want %v", n, i, scaled[i], a[i]*0.5)
			}
		}
	}
	v := []float64{3, 4, 12}
	clip(v, 5)
	s := 5 / 13.0
	for i, want := range []float64{3 * s, 4 * s, 12 * s} {
		if v[i] != want {
			t.Fatalf("clip[%d] = %v, want %v", i, v[i], want)
		}
	}
	same := []float64{0.25, -0.25}
	clip(same, 1)
	if same[0] != 0.25 || same[1] != -0.25 {
		t.Fatalf("short vector was clipped to %v", same)
	}
	clip(nil, 1)

	w := []float64{1, 2, 3, 4, 5, 6}
	dst := []float64{0, 0, 0}
	mixRows(dst, w, []float64{2, -1}, 3)
	for i, want := range []float64{-2, -1, 0} {
		if dst[i] != want {
			t.Fatalf("mixRows[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func ones(n int) []float64 {
	o := make([]float64, n)
	for i := range o {
		o[i] = 1
	}
	return o
}

func TestRNNLearnsTheText(t *testing.T) {
	unit := []rune("abcabc")
	var src []rune
	for range 80 {
		src = append(src, unit...)
	}
	g := buildGraph(src)
	g.rankPages(damp, 1e-15)
	g.normalizeSuccessors()
	net := newRNN(1, g.alphabet())
	g.prepareEmbed(net.id)
	before := rnnProb(g, net, unit, 'a')
	g.pretrain(net)
	after := rnnProb(g, net, unit, 'a')
	if after <= before {
		t.Fatalf("P(a)=%v, before %v", after, before)
	}
	sum := 0.0
	net.reset()
	p := net.predict(nil)
	for _, v := range p {
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("prediction sums to %v", sum)
	}
}

func TestRNNKeepsRunesDistinct(t *testing.T) {
	// U+2019 is the curly apostrophe in the Shakespeare text. Its low 8 bits
	// are a different character and must not share this rune's weights.
	quote := '’'
	low := rune(quote & 0xFF)
	var src []rune
	for range 40 {
		src = append(src, []rune("abcdX")...)
		src = append(src, quote, 'q')
		src = append(src, []rune("abcdX")...)
		src = append(src, low, 'z')
	}
	g := buildGraph(src)
	g.rankPages(damp, 1e-15)
	g.normalizeSuccessors()
	net := newRNN(1, g.alphabet())
	if net.id[quote] == net.id[low] {
		t.Fatal("quote shares an output with its low byte")
	}
	g.prepareEmbed(net.id)
	g.pretrain(net)
	quoteWin := append([]rune("abcdX"), quote)
	lowWin := append([]rune("abcdX"), low)
	if rnnProb(g, net, quoteWin, 'q') <= rnnProb(g, net, quoteWin, 'z') {
		t.Fatal("the quote context did not learn to predict q")
	}
	if rnnProb(g, net, lowWin, 'z') <= rnnProb(g, net, lowWin, 'q') {
		t.Fatal("the low-byte context did not learn to predict z")
	}
}

// rnnProb is the network's probability of next after reading window from a
// cleared state.
func rnnProb(g *Graph, net *RNN, window []rune, next rune) float64 {
	net.reset()
	var h []float64
	buf := make([]float64, net.in)
	for i := 0; i+order <= len(window); i++ {
		id, ok := g.id[tokenFrom(window[i:i+order])]
		if !ok {
			continue
		}
		h = g.stepRNN(net, h, id, buf)
	}
	return net.predict(h)[net.id[next]]
}

func TestCommonWordIsDiscounted(t *testing.T) {
	// "x the" closed by a space is the frequent word. 'm' continues it
	// into another word, so that edge is not a close. A rare word keeps
	// equal weights on its two exits.
	var commonTok, rareTok token
	copy(commonTok[:], []rune("xx the"))
	copy(rareTok[:], []rune("xxx ox"))
	g := &Graph{
		token: []token{commonTok, rareTok},
		words: map[string]int{"the": 1000, "ox": 1},
		outs: [][]edge{
			{
				{to: 1, w: 1, ch: ' ', p: 0.5},
				{to: 1, w: 1, ch: 'm', p: 0.5},
			},
			{
				{to: 0, w: 1, ch: ' ', p: 0.5},
				{to: 0, w: 1, ch: 'e', p: 0.5},
			},
		},
	}
	commonW := g.priorWeights(nil, 0, g.outs[0])
	if commonW[0] >= commonW[1] {
		t.Fatalf("closing \"the\" %v is not below continuing %v", commonW[0], commonW[1])
	}
	rareW := g.priorWeights(nil, 1, g.outs[1])
	if math.Abs(rareW[0]-rareW[1]) > 1e-12 {
		t.Fatalf("rare word weights %v %v", rareW[0], rareW[1])
	}
}

func TestEdgeWeightFollowsFourGram(t *testing.T) {
	outs := []edge{{ch: 'a', p: 0.8}, {ch: 'b', p: 0.2}}
	dist := map[rune]float64{'b': 1}
	wa := edgeWeight(outs[0], dist, 1)
	wb := edgeWeight(outs[1], dist, 1)
	if wa <= wb {
		t.Fatalf("token favorite a=%v should stay ahead of tilted b=%v", wa, wb)
	}
	if math.Abs(wa-(1-tilt)*0.8) > 1e-12 {
		t.Fatalf("a weight = %v", wa)
	}
	wantB := (1-tilt)*0.2 + tilt
	if math.Abs(wb-wantB) > 1e-12 {
		t.Fatalf("b weight = %v, want %v", wb, wantB)
	}
	// With no conditional mass the weights stay proportional to the token.
	wa = edgeWeight(outs[0], nil, 0)
	wb = edgeWeight(outs[1], nil, 0)
	if math.Abs(wa/wb-4) > 1e-9 {
		t.Fatalf("untilted ratio = %v, want 4", wa/wb)
	}
}

func TestEdgeProbabilityUsesCountAndRank(t *testing.T) {
	src := []rune("abcdXYqabcdXYzabcdXYq")
	g := buildGraph(src)
	g.rankPages(damp, 1e-15)
	g.normalizeSuccessors()
	node := g.id[tokenFrom([]rune("abcdXY"))]
	var pq, pz, wq, wz, rq, rz float64
	for _, e := range g.outs[node] {
		switch e.ch {
		case 'q':
			pq, wq, rq = e.p, e.w, g.rank[e.to]
		case 'z':
			pz, wz, rz = e.p, e.w, g.rank[e.to]
		}
	}
	if pq <= 0 || pz <= 0 {
		t.Fatalf("p(q)=%v p(z)=%v", pq, pz)
	}
	if math.Abs(pq+pz-1) > 1e-12 {
		t.Fatalf("edge probabilities sum to %v", pq+pz)
	}
	if math.Abs(pq/pz-(wq*rq)/(wz*rz)) > 1e-9 {
		t.Fatalf("p ratio %v, want count*rank ratio %v", pq/pz, (wq*rq)/(wz*rz))
	}
}

func TestMCTSLooksAhead(t *testing.T) {
	// 'a' is the likelier next character, then the walk enters a flat node.
	// 'b' is less likely now and certain afterwards. Visit counts should
	// follow the whole horizon, so 'b' is searched more.
	flat := make([]edge, 10)
	for i := range flat {
		flat[i] = edge{to: 1, w: 1, ch: rune('c' + i), p: 0.1}
	}
	g := &Graph{
		outs: [][]edge{
			{
				{to: 1, w: 1, ch: 'a', p: 0.65},
				{to: 2, w: 1, ch: 'b', p: 0.35},
			},
			flat,
			{{to: 2, w: 1, ch: 'b', p: 1}},
		},
		token: make([]token, 3),
	}
	rng := rand.New(rand.NewPCG(1, 1^0x9e3779b97f4a7c15))
	root := g.explore(nil, nil, 0, rng)
	var a, b float64
	for _, c := range root.children {
		switch c.edge.ch {
		case 'a':
			a = c.visits
		case 'b':
			b = c.visits
		}
	}
	if b <= a {
		t.Fatalf("lookahead visits b=%v, local favorite a=%v", b, a)
	}
	again := g.explore(nil, nil, 0, rand.New(rand.NewPCG(1, 1^0x9e3779b97f4a7c15)))
	for i := range root.children {
		if again.children[i].visits != root.children[i].visits {
			t.Fatalf("visits differ on the same seed at %q", root.children[i].edge.ch)
		}
	}
}

func TestMCTSReturnsTheOnlyEdge(t *testing.T) {
	g := &Graph{
		outs:  [][]edge{{{to: 1, w: 1, ch: 'z', p: 1}}, nil},
		token: make([]token, 2),
	}
	e := g.search(nil, nil, 0, rand.New(rand.NewPCG(3, 4)))
	if e.ch != 'z' || e.to != 1 {
		t.Fatalf("search = %+v, want the only edge", e)
	}
}
