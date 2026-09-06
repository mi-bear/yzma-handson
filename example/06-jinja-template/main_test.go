package main

import (
	"strings"
	"testing"
)

// TestApplyTemplateFallback は、モデルにチャットテンプレートが埋め込まれて
// いない場合に組み込みテンプレートを代わりに使うことを検証する。
// model にゼロ値を渡すと llama.ModelChatTemplate は共有ライブラリを
// 呼び出さずに空文字を返す ( pkg/llama/model.go の ModelChatTemplate 参照 ) ので、
// ライブラリやモデルファイルを一切ロードせずにこの経路だけをテストできる。
func TestApplyTemplateFallback(t *testing.T) {
	prompt, err := applyTemplate(0, "sys", "user")
	if err != nil {
		t.Fatalf("applyTemplate returned error: %v", err)
	}
	if prompt == "" {
		t.Fatal("applyTemplate returned an empty prompt")
	}
	if !strings.Contains(prompt, "sys") {
		t.Errorf("prompt = %q, want it to contain the system text %q", prompt, "sys")
	}
	if !strings.Contains(prompt, "user") {
		t.Errorf("prompt = %q, want it to contain the user text %q", prompt, "user")
	}
}
