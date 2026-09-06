package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hybridgroup/yzma/pkg/download"
)

func TestReadInputFromStdin(t *testing.T) {
	stdin := strings.NewReader("標準入力からのテキスト")
	got, err := readInput("", stdin)
	if err != nil {
		t.Fatalf("readInput returned error: %v", err)
	}
	if string(got) != "標準入力からのテキスト" {
		t.Errorf("readInput = %q, want %q", got, "標準入力からのテキスト")
	}
}

func TestReadInputFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "input.txt")
	want := "ファイルからのテキスト"
	if err := os.WriteFile(path, []byte(want), 0o644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	got, err := readInput(path, strings.NewReader(""))
	if err != nil {
		t.Fatalf("readInput returned error: %v", err)
	}
	if string(got) != want {
		t.Errorf("readInput = %q, want %q", got, want)
	}
}

func TestReadInputMissingFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "does-not-exist.txt")

	if _, err := readInput(path, strings.NewReader("")); err == nil {
		t.Error("readInput with a missing file should return an error")
	}
}

func TestReadInputEmpty(t *testing.T) {
	if _, err := readInput("", strings.NewReader("")); err == nil {
		t.Error("readInput with empty stdin should return an error")
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(path, []byte(""), 0o644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}
	if _, err := readInput(path, strings.NewReader("")); err == nil {
		t.Error("readInput with an empty file should return an error")
	}
}

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
