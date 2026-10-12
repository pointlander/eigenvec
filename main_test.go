// Copyright 2026 The eigenvec Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"strings"
	"testing"
)

func TestPageRankKnownGraph(t *testing.T) {
	// Weighted graph from the alixaxel/pagerank README, solved to 1e-15.
	// Node 4 is dangling.
	g := &Graph{outs: make([][]edge, 4)}
	g.outs[0] = []edge{{to: 1, w: 1}, {to: 2, w: 2}}
	g.outs[1] = []edge{{to: 2, w: 3}, {to: 3, w: 4}}
	g.outs[2] = []edge{{to: 0, w: 5}}
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

func TestPromptSumsNextCharacterMass(t *testing.T) {
	// "abcdXY" is followed by 'q' twice and 'z' once. Both "abcdXY" and
	// "wxyzXY" end in the order-2 context "XY", so they share a conditional
	// distribution. "bcdXYq" ends in "Yq" and must not mix into it.
	src := []rune("abcdXYqabcdXYzabcdXYqwxyzXYr")
	g := buildGraph(src)
	g.rankPages(damp, 1e-15)
	g.normalizeSuccessors()

	model := map[context]map[rune]float64{}
	node := g.id[tokenFrom([]rune("abcdXY"))]
	g.addNext(model, node)
	ctx := contextOf(g.token[node])
	dist := model[ctx]
	if dist['q'] <= 0 || dist['z'] <= 0 {
		t.Fatalf("dist = %v, want mass on both successors", dist)
	}
	if sumDist(dist) != 1 {
		t.Fatalf("local distribution sums to %v", sumDist(dist))
	}

	firstQ, firstZ := dist['q'], dist['z']

	// Adding the same node again mixes the conditional with itself, then
	// normalizes, so the mass stays 1 and the ratios stay put.
	g.addNext(model, node)
	dist = model[ctx]
	if math.Abs(sumDist(dist)-1) > 1e-12 {
		t.Fatalf("normalized dist sums to %v", sumDist(dist))
	}
	if math.Abs(dist['q']-firstQ) > 1e-12 || math.Abs(dist['z']-firstZ) > 1e-12 {
		t.Fatalf("dist = %v, want q=%v z=%v", dist, firstQ, firstZ)
	}

	// A different node with the same two-character suffix is mixed in at
	// weight mix, then normalized.
	other := g.id[tokenFrom([]rune("wxyzXY"))]
	if contextOf(g.token[other]) != ctx {
		t.Fatal("wxyzXY should share the order-2 context of abcdXY")
	}
	before := make(map[rune]float64, len(dist))
	for ch, p := range dist {
		before[ch] = p
	}
	g.addNext(model, other)
	dist = model[ctx]
	if math.Abs(sumDist(dist)-1) > 1e-12 {
		t.Fatalf("mixed dist sums to %v", sumDist(dist))
	}
	added := map[rune]float64{}
	for _, e := range g.outs[other] {
		added[e.ch] += e.p
	}
	for ch, p := range before {
		want := (1-mix)*p + mix*added[ch]
		if math.Abs(dist[ch]-want) > 1e-12 {
			t.Fatalf("dist[%q] = %v, want %v", ch, dist[ch], want)
		}
	}
	for ch, p := range added {
		want := (1-mix)*before[ch] + mix*p
		if math.Abs(dist[ch]-want) > 1e-12 {
			t.Fatalf("dist[%q] = %v, want %v", ch, dist[ch], want)
		}
	}

	// A node with a different suffix gets its own conditional and leaves
	// the "XY" distribution unchanged.
	kept := make(map[rune]float64, len(dist))
	for ch, p := range dist {
		kept[ch] = p
	}
	separate := g.id[tokenFrom([]rune("bcdXYq"))]
	g.addNext(model, separate)
	dist = model[ctx]
	for ch, p := range kept {
		if math.Abs(dist[ch]-p) > 1e-12 {
			t.Fatalf("context XY changed at %q: %v, want %v", ch, dist[ch], p)
		}
	}
	otherCtx := contextOf(g.token[separate])
	if otherCtx == ctx {
		t.Fatal("bcdXYq should not share the order-2 context of abcdXY")
	}
	if math.Abs(sumDist(model[otherCtx])-1) > 1e-12 {
		t.Fatalf("separate conditional sums to %v", sumDist(model[otherCtx]))
	}
	if model[otherCtx]['a'] <= 0 {
		t.Fatalf("separate conditional = %v, want mass on 'a'", model[otherCtx])
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

func sumDist(dist map[rune]float64) float64 {
	sum := 0.0
	for _, p := range dist {
		sum += p
	}
	if math.Abs(sum-1) <= 1e-12 {
		return 1
	}
	return sum
}
