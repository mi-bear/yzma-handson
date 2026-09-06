// この例は練習問題 6 ( Jinja テンプレートエンジンの利用 ) の解答例。
// ../summarize/ は llama.ChatApplyTemplate ( llama.cpp 組み込みの C 実装 ) で
// プロンプトを整形していたが、ここでは pkg/template の Jinja テンプレートエンジンと
// pkg/message のメッセージ型を使う。モデルのテンプレートが取得できない場合は
// 組み込みの "gemma3" テンプレートを代わりに使う。
//
// 使い方:
//
//	echo "要約したい文章" | go run . -mode summary
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
	"github.com/hybridgroup/yzma/pkg/template"
)

// 学習用に定数として持っておく。実運用ではフラグや設定ファイルで指定できるようにするのが良い。
const (
	contextTokens   = 4096 // モデルが一度に扱えるトークン数の上限
	maxOutputTokens = 512  // 生成する応答の最大トークン数

	defaultModelFile = "gemma-3-1b-it-Q4_K_M.gguf" // -model 未指定時に使う既定モデル

	tokenPieceBufSize = 256 // 1トークン分の文字列を受け取るバッファサイズ

	builtinTemplateName = "gemma3" // モデルにテンプレートが埋め込まれていない場合に使う組み込みテンプレート
)

// classifyPrompt は分類モードのシステムプロンプト。gemma-3-1b-it のような
// 1B 級の小さいモデルは、ラベルの定義や具体例 ( few-shot ) を与えないと
// 分類がブレたり、常に同じラベルに偏ったりする。実務でも、まずラベルの定義と例を
// 書くところからプロンプトを調整する。
const classifyPrompt = `あなたはカスタマーサポートの受付係です。お客様からの文章を次の 4 つのどれか 1 つに分類してください。
- 質問: 使い方や仕様を知りたい。「〜ですか」「教えてください」など
- 要望: 新しい対応やサービスの追加をお願いしている。「〜してほしい」「〜できませんか」など
- 苦情: 不満や被害を訴え、謝罪や返金などを求めている
- その他: 上のどれにも当てはまらない
例:
文章: この掃除機は畳でも使えますか。
答え: 質問
文章: 領収書をメールでも送ってもらえると助かります。
答え: 要望
文章: 届いた皿が割れていました。交換してください。
答え: 苦情
ラベルを 1 語だけ出力してください。説明は書かないでください。`

// mode ごとにシステムプロンプト ( モデルへの役割指示 ) を切り替える。
var systemPrompts = map[string]string{
	"summary":  "あなたは優秀な編集者です。与えられた文章を、日本語で3行以内に要約してください。要約以外は出力しないでください。",
	"classify": classifyPrompt,
}

func main() {
	mode := flag.String("mode", "summary", "処理モード: summary または classify")
	modelFile := flag.String("model", "", "GGUF モデルファイルのパス ( 省略時は "+download.DefaultModelsDir()+" 配下の "+defaultModelFile+" を使う )")
	flag.Parse()

	system, err := systemPromptFor(*mode)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// 標準入力から処理対象テキストを読み込む。
	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintln(os.Stderr, "標準入力の読み込みに失敗:", err)
		os.Exit(1)
	}
	if len(input) == 0 {
		fmt.Fprintln(os.Stderr, "標準入力が空です。処理するテキストを渡してください")
		os.Exit(1)
	}

	result, err := run(resolveModelPath(*modelFile), system, string(input))
	if err != nil {
		fmt.Fprintln(os.Stderr, "推論に失敗:", err)
		os.Exit(1)
	}

	fmt.Println(result)
}

// systemPromptFor は mode に対応するシステムプロンプトを返す。
func systemPromptFor(mode string) (string, error) {
	system, ok := systemPrompts[mode]
	if !ok {
		return "", fmt.Errorf("不明な mode: %q ( summary か classify を指定 )", mode)
	}
	return system, nil
}

// resolveModelPath は -model フラグの値を解決する。
func resolveModelPath(flagValue string) string {
	if flagValue != "" {
		return flagValue
	}
	return filepath.Join(download.DefaultModelsDir(), defaultModelFile)
}

// libPathFromEnv は共有ライブラリ ( libllama ) のパスを環境変数から取得する。
func libPathFromEnv(getenv func(string) string) (string, error) {
	path := getenv("YZMA_LIB")
	if path == "" {
		return "", fmt.Errorf("環境変数 YZMA_LIB が未設定です。`yzma install --lib` に渡したディレクトリを設定してください")
	}
	return path, nil
}

// run はモデルをロードして1回の推論を実行し、生成テキストを返す。
func run(modelFile, system, userText string) (string, error) {
	// 1. 共有ライブラリ ( libllama ) をロードする。
	libPath, err := libPathFromEnv(os.Getenv)
	if err != nil {
		return "", err
	}
	if err := llama.Load(libPath); err != nil {
		return "", fmt.Errorf("ライブラリのロード失敗 ( YZMA_LIB を確認 ): %w", err)
	}
	llama.LogSet(llama.LogSilent())

	// 2. ランタイムを初期化。
	llama.Init()
	defer llama.Close()

	// 3. モデルをファイルから読み込む。
	model, err := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	if err != nil {
		return "", fmt.Errorf("モデル読み込み失敗: %w", err)
	}
	defer llama.ModelFree(model)

	vocab := llama.ModelGetVocab(model)

	// 4. 推論コンテキストを作る。
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = contextTokens
	lctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		return "", fmt.Errorf("コンテキスト初期化失敗: %w", err)
	}
	defer llama.Free(lctx)

	// 5. サンプラーを作る。
	sp := llama.DefaultSamplerParams()
	sp.Temp = 0.2
	samplers := []llama.SamplerType{llama.SamplerTypeTopK, llama.SamplerTypeTopP, llama.SamplerTypeTemperature}
	sampler := llama.NewSampler(model, samplers, sp)
	defer llama.SamplerFree(sampler)

	// 6. Jinja テンプレートを使ってプロンプトを組み立てる。
	prompt, err := applyTemplate(model, system, userText)
	if err != nil {
		return "", err
	}

	// 7. プロンプトをトークン化して推論ループを回す。
	tokens := llama.Tokenize(vocab, prompt, true, true)
	batch := llama.BatchGetOne(tokens)

	// エンコーダー・デコーダー型のモデル ( 例: T5 ) はデコードの前にエンコードが必要。
	// gemma はデコーダーだけのモデルなので、この分岐は通らない。
	if llama.ModelHasEncoder(model) {
		code, err := llama.Encode(lctx, batch)
		if err != nil {
			return "", fmt.Errorf("エンコード失敗: %w", err)
		}
		if code != 0 {
			return "", fmt.Errorf("エンコード失敗: llama_encode がコード %d を返した", code)
		}

		start := llama.ModelDecoderStartToken(model)
		if start == llama.TokenNull {
			start = llama.VocabBOS(vocab)
		}
		batch = llama.BatchGetOne([]llama.Token{start})
	}

	var response strings.Builder
	for pos := int32(0); pos < maxOutputTokens; pos += batch.NTokens {
		code, err := llama.Decode(lctx, batch)
		if err != nil {
			return "", fmt.Errorf("デコード失敗: %w", err)
		}
		if code != 0 {
			return "", fmt.Errorf("デコード失敗: llama_decode がコード %d を返した", code)
		}
		token := llama.SamplerSample(sampler, lctx, -1)

		if llama.VocabIsEOG(vocab, token) {
			break
		}

		buf := make([]byte, tokenPieceBufSize)
		n := llama.TokenToPiece(vocab, token, buf, 0, false)
		if n < 0 {
			return "", fmt.Errorf("トークンの文字列化に失敗した ( token=%d )", token)
		}
		response.Write(buf[:n])

		batch = llama.BatchGetOne([]llama.Token{token})
	}

	return response.String(), nil
}

// applyTemplate は Jinja チャットテンプレートを適用して、モデルが期待する
// 形式のプロンプト文字列を作る。モデルにテンプレートが埋め込まれていない
// 場合は、組み込みの builtinTemplateName を代わりに使う。
func applyTemplate(model llama.Model, system, userText string) (string, error) {
	tmpl := llama.ModelChatTemplate(model, "")
	if tmpl == "" {
		var ok bool
		tmpl, ok = template.BuiltinTemplate(builtinTemplateName)
		if !ok {
			return "", fmt.Errorf("組み込みテンプレート %q が見つからない", builtinTemplateName)
		}
	}

	messages := []message.Message{
		message.Chat{Role: "system", Content: system},
		message.Chat{Role: "user", Content: userText},
	}

	prompt, err := template.Apply(tmpl, messages, true)
	if err != nil {
		return "", fmt.Errorf("チャットテンプレートの適用に失敗した: %w", err)
	}
	return prompt, nil
}
