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
	// "abcd" is followed by 'e' twice and 'f' once. Walking a prompt that
	// visits "abcd" must put mass on both continuations, in PageRank order
	// rather than as a single sample.
	src := []rune("abcdeabcdfabcde")
	g := buildGraph(src)
	g.rankPages(damp, 1e-15)
	g.normalizeSuccessors()

	node := g.id[tokenFrom([]rune("abcd"))]
	dist := map[rune]float64{}
	g.addNext(dist, node)
	if dist['e'] <= 0 || dist['f'] <= 0 {
		t.Fatalf("dist = %v, want mass on both successors", dist)
	}
	sum := 0.0
	for _, p := range dist {
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("local distribution sums to %v", sum)
	}

	firstC, firstD := dist['e'], dist['f']

	// Adding the same node again mixes the distribution with itself, then
	// normalizes, so the mass stays 1 and the ratios stay put.
	g.addNext(dist, node)
	sum = 0
	for _, p := range dist {
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("normalized dist sums to %v", sum)
	}
	if math.Abs(dist['e']-firstC) > 1e-12 || math.Abs(dist['f']-firstD) > 1e-12 {
		t.Fatalf("dist = %v, want e=%v f=%v", dist, firstC, firstD)
	}

	// A different node is averaged in at equal weight, then normalized.
	other := g.id[tokenFrom([]rune("bcde"))]
	before := make(map[rune]float64, len(dist))
	for ch, p := range dist {
		before[ch] = p
	}
	g.addNext(dist, other)
	sum = 0
	for _, p := range dist {
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("mixed dist sums to %v", sum)
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
}
