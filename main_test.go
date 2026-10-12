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

	prompt := []rune("To be")
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

func TestPromptSumsNextCharacterMass(t *testing.T) {
	// "abXY" is followed by 'q' twice and 'z' once. Both "abXY" and "cdXY"
	// end in the order-2 context "XY", so they share a conditional
	// distribution. "bXYq" ends in "Yq" and must not mix into it.
	src := []rune("abXYqabXYzabXYqcdXYr")
	g := buildGraph(src)
	g.rankPages(damp, 1e-15)
	g.normalizeSuccessors()

	model := map[context]map[rune]float64{}
	node := g.id[tokenFrom([]rune("abXY"))]
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

	// A different order-4 node with the same two-character suffix is
	// averaged into this conditional, then normalized.
	other := g.id[tokenFrom([]rune("cdXY"))]
	if contextOf(g.token[other]) != ctx {
		t.Fatal("cdXY should share the order-2 context of abXY")
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
		want := (p + added[ch]) / 2
		if math.Abs(dist[ch]-want) > 1e-12 {
			t.Fatalf("dist[%q] = %v, want %v", ch, dist[ch], want)
		}
	}
	for ch, p := range added {
		want := (before[ch] + p) / 2
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
	separate := g.id[tokenFrom([]rune("bXYq"))]
	g.addNext(model, separate)
	dist = model[ctx]
	for ch, p := range kept {
		if math.Abs(dist[ch]-p) > 1e-12 {
			t.Fatalf("context XY changed at %q: %v, want %v", ch, dist[ch], p)
		}
	}
	otherCtx := contextOf(g.token[separate])
	if otherCtx == ctx {
		t.Fatal("bXYq should not share the order-2 context of abXY")
	}
	if math.Abs(sumDist(model[otherCtx])-1) > 1e-12 {
		t.Fatalf("separate conditional sums to %v", sumDist(model[otherCtx]))
	}
	if model[otherCtx]['a'] <= 0 {
		t.Fatalf("separate conditional = %v, want mass on 'a'", model[otherCtx])
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
