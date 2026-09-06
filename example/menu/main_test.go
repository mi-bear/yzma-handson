package main

import (
	"strings"
	"testing"
)

func TestBuildUserPrompt(t *testing.T) {
	got := buildUserPrompt("卵、キャベツ、豚こま", 2, 3)

	wantSubstrings := []string{
		"卵、キャベツ、豚こま",
		"2人分",
		"3個",
		"3行",
	}
	for _, want := range wantSubstrings {
		if !strings.Contains(got, want) {
			t.Errorf("buildUserPrompt output = %q, want it to contain %q", got, want)
		}
	}
}
