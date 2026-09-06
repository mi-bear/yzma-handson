package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/hybridgroup/yzma/pkg/download"
)

func TestSystemPromptFor(t *testing.T) {
	for mode := range systemPrompts {
		t.Run(mode, func(t *testing.T) {
			got, err := systemPromptFor(mode)
			if err != nil {
				t.Fatalf("systemPromptFor(%q) returned error: %v", mode, err)
			}
			if got != systemPrompts[mode] {
				t.Errorf("systemPromptFor(%q) = %q, want %q", mode, got, systemPrompts[mode])
			}
		})
	}

	if _, err := systemPromptFor("unknown"); err == nil {
		t.Error("systemPromptFor(\"unknown\") should return an error")
	}
}

func TestResolveModelPath(t *testing.T) {
	const explicit = "/path/to/model.gguf"
	if got := resolveModelPath(explicit); got != explicit {
		t.Errorf("resolveModelPath(%q) = %q, want %q", explicit, got, explicit)
	}

	got := resolveModelPath("")
	want := filepath.Join(download.DefaultModelsDir(), defaultModelFile)
	if got != want {
		t.Errorf("resolveModelPath(\"\") = %q, want %q", got, want)
	}
	if !strings.HasSuffix(got, defaultModelFile) {
		t.Errorf("resolveModelPath(\"\") = %q, want suffix %q", got, defaultModelFile)
	}
}

func TestLibPathFromEnv(t *testing.T) {
	const libPath = "/opt/yzma/lib"
	getenvSet := func(key string) string {
		if key == "YZMA_LIB" {
			return libPath
		}
		return ""
	}
	got, err := libPathFromEnv(getenvSet)
	if err != nil {
		t.Fatalf("libPathFromEnv with set value returned error: %v", err)
	}
	if got != libPath {
		t.Errorf("libPathFromEnv = %q, want %q", got, libPath)
	}

	getenvEmpty := func(string) string { return "" }
	if _, err := libPathFromEnv(getenvEmpty); err == nil {
		t.Error("libPathFromEnv with empty YZMA_LIB should return an error")
	}
}
