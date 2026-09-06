package main

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/hybridgroup/yzma/pkg/download"
)

func TestResolveModelList(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{
			name: "通常のカンマ区切りリスト",
			in:   "a.gguf,b.gguf,c.gguf",
			want: []string{"a.gguf", "b.gguf", "c.gguf"},
		},
		{
			name: "前後に空白がある要素",
			in:   " a.gguf , b.gguf ",
			want: []string{"a.gguf", "b.gguf"},
		},
		{
			name: "カンマの間に空の要素がある",
			in:   "a.gguf,,b.gguf",
			want: []string{"a.gguf", "b.gguf"},
		},
		{
			name: "空白だけの入力は要素なしになる",
			in:   " , ",
			want: []string{},
		},
		{
			name: "空文字は既定モデル1つを使う",
			in:   "",
			want: []string{resolveModelPath("")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveModelList(tt.in)
			if !slices.Equal(got, tt.want) {
				t.Errorf("resolveModelList(%q) = %#v, want %#v", tt.in, got, tt.want)
			}
		})
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
}
