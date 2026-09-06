//go:build integration

package main

import (
	"os"
	"testing"
)

// TestRunIntegration は実際に libllama とモデルを使って run() を最後まで実行する。
// ローカル LLM の実行環境 ( YZMA_LIB とモデルファイル ) が無いと成立しないテストなので、
// 前提条件が揃っていない場合は失敗ではなくスキップする。
//
// 実行方法:
//
//	YZMA_LIB=<yzma install --lib で指定したディレクトリ> go test -tags integration ./...
//
// モデルファイルは既定で resolveModelPath("") の解決先を使う。
// 別のモデルで試したい場合は環境変数 YZMA_TEST_MODEL でパスを上書きできる。
func TestRunIntegration(t *testing.T) {
	if _, err := libPathFromEnv(os.Getenv); err != nil {
		t.Skipf("YZMA_LIB が未設定のためスキップ: %v", err)
	}

	modelFile := os.Getenv("YZMA_TEST_MODEL")
	if modelFile == "" {
		modelFile = resolveModelPath("")
	}
	if _, err := os.Stat(modelFile); err != nil {
		t.Skipf("モデルファイルが見つからないためスキップ ( %s ): %v", modelFile, err)
	}

	system, err := systemPromptFor("summary")
	if err != nil {
		t.Fatalf("systemPromptFor failed: %v", err)
	}

	const input = "今日は天気が良く、公園には多くの人が散歩に来ていた。子どもたちは芝生で遊び、大人たちはベンチで日光浴を楽しんでいた。"

	result, err := run(modelFile, system, input)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if result == "" {
		t.Error("run returned an empty result")
	}
}
