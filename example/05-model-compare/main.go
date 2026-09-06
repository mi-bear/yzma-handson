// この例は練習問題 5 ( 複数モデルの比較 ) の解答例。
// ../summarize/ は1つのモデルだけを読み込んでいたが、ここでは -models に
// カンマ区切りで複数の GGUF ファイルを渡すと、同じプロンプトを全モデルに渡して
// 応答内容と所要時間を並べて表示する。共有ライブラリは1回だけロードし、
// モデルとコンテキストはモデルごとに読み込んでは解放する。
//
// 使い方:
//
//	echo "要約したい文章" | go run . -mode summary -models a.gguf,b.gguf
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
)

// 学習用に定数として持っておく。実運用ではフラグや設定ファイルで指定できるようにするのが良い。
const (
	contextTokens   = 4096 // モデルが一度に扱えるトークン数の上限
	maxOutputTokens = 512  // 生成する応答の最大トークン数

	defaultModelFile = "gemma-3-1b-it-Q4_K_M.gguf" // -models 未指定時に使う既定モデル

	tokenPieceBufSize   = 256  // 1トークン分の文字列を受け取るバッファサイズ
	templateBufOverhead = 1024 // チャットテンプレート適用後に増える分の余裕
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

// modelResult は1つのモデルで生成した結果と所要時間をまとめたもの。
type modelResult struct {
	ModelFile string
	Elapsed   time.Duration
	Text      string
	Err       error
}

func main() {
	mode := flag.String("mode", "summary", "処理モード: summary または classify")
	modelsFlag := flag.String("models", "", "比較する GGUF モデルファイルのパス ( カンマ区切り。省略時は既定モデル1つだけを使う )")
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

	modelFiles := resolveModelList(*modelsFlag)
	if len(modelFiles) == 0 {
		fmt.Fprintln(os.Stderr, "比較するモデルが 1 つも指定されていません")
		os.Exit(1)
	}

	results, err := run(modelFiles, system, string(input))
	if err != nil {
		fmt.Fprintln(os.Stderr, "推論に失敗:", err)
		os.Exit(1)
	}

	for _, r := range results {
		fmt.Printf("=== %s ( %s ) ===\n", filepath.Base(r.ModelFile), r.Elapsed)
		if r.Err != nil {
			fmt.Printf("エラー: %v\n\n", r.Err)
			continue
		}
		fmt.Printf("%s\n\n", r.Text)
	}
}

// resolveModelList は -models フラグの値をモデルファイルパスの一覧に変換する。
// 空文字の場合は既定モデル1つだけの一覧を返す。
func resolveModelList(flagValue string) []string {
	if flagValue == "" {
		return []string{resolveModelPath("")}
	}
	parts := strings.Split(flagValue, ",")
	files := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			files = append(files, p)
		}
	}
	return files
}

// systemPromptFor は mode に対応するシステムプロンプトを返す。
func systemPromptFor(mode string) (string, error) {
	system, ok := systemPrompts[mode]
	if !ok {
		return "", fmt.Errorf("不明な mode: %q ( summary か classify を指定 )", mode)
	}
	return system, nil
}

// resolveModelPath は -model 相当のフラグ値を解決する。
// 未指定 ( 空文字 ) の場合は download.DefaultModelsDir() 配下の既定モデルを使う。
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

// run は共有ライブラリを1回だけロードし、modelFiles の各モデルについて
// 同じプロンプトで1回ずつ推論を行って結果をまとめて返す。
func run(modelFiles []string, system, userText string) ([]modelResult, error) {
	// 1. 共有ライブラリ ( libllama ) をロードする。全モデルで共通。
	libPath, err := libPathFromEnv(os.Getenv)
	if err != nil {
		return nil, err
	}
	if err := llama.Load(libPath); err != nil {
		return nil, fmt.Errorf("ライブラリのロード失敗 ( YZMA_LIB を確認 ): %w", err)
	}
	llama.LogSet(llama.LogSilent())

	// 2. ランタイムを初期化。全モデルで共通。
	llama.Init()
	defer llama.Close()

	// 3〜7. モデルごとにロード・推論・解放を行う。
	results := make([]modelResult, 0, len(modelFiles))
	for _, modelFile := range modelFiles {
		start := time.Now()
		text, err := runOneModel(modelFile, system, userText)
		results = append(results, modelResult{
			ModelFile: modelFile,
			Elapsed:   time.Since(start),
			Text:      text,
			Err:       err,
		})
	}

	return results, nil
}

// runOneModel は1つのモデルファイルを読み込み、1回の推論を実行して結果を返す。
// defer によって、このモデルに関するリソースはこの関数を抜けるときに解放される。
func runOneModel(modelFile, system, userText string) (string, error) {
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

	// 6. チャットメッセージを組み立て、モデル付属のチャットテンプレートを適用する。
	messages := []llama.ChatMessage{
		llama.NewChatMessage("system", system),
		llama.NewChatMessage("user", userText),
	}
	prompt, err := applyTemplate(model, messages, len(system)+len(userText))
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

// applyTemplate はモデルに埋め込まれたチャットテンプレートを適用して、
// モデルが期待する形式のプロンプト文字列を作る。
func applyTemplate(model llama.Model, messages []llama.ChatMessage, contentLen int) (string, error) {
	tmpl := llama.ModelChatTemplate(model, "")
	if tmpl == "" {
		// テンプレートを持たないモデルでは chatml を使う
		tmpl = "chatml"
	}

	bufSize := contentLen + templateBufOverhead
	buf := make([]byte, bufSize)
	n := llama.ChatApplyTemplate(tmpl, messages, true, buf)
	if n <= 0 {
		return "", fmt.Errorf("チャットテンプレートの適用に失敗した ( template=%q )", tmpl)
	}
	if int(n) > len(buf) {
		buf = make([]byte, n)
		n = llama.ChatApplyTemplate(tmpl, messages, true, buf)
		if n <= 0 {
			return "", fmt.Errorf("チャットテンプレートの適用に失敗した ( template=%q )", tmpl)
		}
	}
	return string(buf[:n]), nil
}
