// この例は練習問題 2 ( 温度によるブレの観察 ) の解答例。
// ../summarize/ に -temp ( サンプリング温度 ) と -runs ( 同じプロンプトで生成する回数 ) の
// 2つのフラグを追加した。モデルとコンテキストは1回だけロードして使い回し、
// 実行のたびにサンプラーを作り直し、KV キャッシュ ( 直前までの計算結果 ) を
// llama.MemoryClear でクリアしてから次の生成に入る。
//
// 使い方:
//
//	echo "要約したい文章" | go run . -mode summary -temp 0.8 -runs 5
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
)

// 学習用に定数として持っておく。実運用ではフラグや設定ファイルで指定できるようにするのが良い。
const (
	contextTokens   = 4096 // モデルが一度に扱えるトークン数の上限
	maxOutputTokens = 512  // 生成する応答の最大トークン数

	defaultModelFile = "gemma-3-1b-it-Q4_K_M.gguf" // -model 未指定時に使う既定モデル
	defaultTemp      = 0.2                         // -temp 未指定時のサンプリング温度
	defaultRuns      = 3                           // -runs 未指定時の実行回数

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

func main() {
	mode := flag.String("mode", "summary", "処理モード: summary または classify")
	modelFile := flag.String("model", "", "GGUF モデルファイルのパス ( 省略時は "+download.DefaultModelsDir()+" 配下の "+defaultModelFile+" を使う )")
	temp := flag.Float64("temp", defaultTemp, "サンプリング温度 ( 0 に近いほど決定的、大きいほど多様な出力になる )")
	runs := flag.Int("runs", defaultRuns, "同じプロンプトで生成を繰り返す回数 ( 温度によるブレを見比べるため )")
	flag.Parse()

	system, err := systemPromptFor(*mode)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if *runs < 1 {
		fmt.Fprintln(os.Stderr, "-runs は 1 以上を指定してください")
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

	results, err := run(resolveModelPath(*modelFile), system, string(input), float32(*temp), *runs)
	if err != nil {
		fmt.Fprintln(os.Stderr, "推論に失敗:", err)
		os.Exit(1)
	}

	for i, result := range results {
		fmt.Printf("--- 実行 %d/%d ( temp=%.2f ) ---\n%s\n\n", i+1, len(results), *temp, result)
	}
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

// run はモデルとコンテキストを1回だけロードし、同じプロンプトに対して
// temp で指定した温度の生成を runs 回繰り返して、その結果をすべて返す。
func run(modelFile, system, userText string, temp float32, runs int) ([]string, error) {
	// 1. 共有ライブラリ ( libllama ) をロードする。
	libPath, err := libPathFromEnv(os.Getenv)
	if err != nil {
		return nil, err
	}
	if err := llama.Load(libPath); err != nil {
		return nil, fmt.Errorf("ライブラリのロード失敗 ( YZMA_LIB を確認 ): %w", err)
	}
	llama.LogSet(llama.LogSilent())

	// 2. ランタイムを初期化。
	llama.Init()
	defer llama.Close()

	// 3. モデルをファイルから読み込む。
	model, err := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	if err != nil {
		return nil, fmt.Errorf("モデル読み込み失敗: %w", err)
	}
	defer llama.ModelFree(model)

	vocab := llama.ModelGetVocab(model)

	// 4. 推論コンテキストを作る。すべての実行で使い回す。
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = contextTokens
	lctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		return nil, fmt.Errorf("コンテキスト初期化失敗: %w", err)
	}
	defer llama.Free(lctx)

	// 5. チャットメッセージを組み立て、プロンプトをトークン化する。
	//    プロンプト自体は runs 回とも共通なので、ここで1回だけ作る。
	messages := []llama.ChatMessage{
		llama.NewChatMessage("system", system),
		llama.NewChatMessage("user", userText),
	}
	prompt, err := applyTemplate(model, messages, len(system)+len(userText))
	if err != nil {
		return nil, err
	}
	tokens := llama.Tokenize(vocab, prompt, true, true)

	// 6. runs 回の生成ループ。回ごとにサンプラーを作り直し、
	//    前回の生成で溜まった KV キャッシュをクリアしてから始める。
	results := make([]string, 0, runs)
	for i := 0; i < runs; i++ {
		text, err := generateOnce(lctx, vocab, model, tokens, temp)
		if err != nil {
			return nil, fmt.Errorf("%d 回目の生成に失敗: %w", i+1, err)
		}
		results = append(results, text)

		// 次の実行に前回の生成内容を持ち越さないよう、KV キャッシュをクリアする。
		if i < runs-1 {
			mem, err := llama.GetMemory(lctx)
			if err != nil {
				return nil, fmt.Errorf("メモリ取得失敗: %w", err)
			}
			if err := llama.MemoryClear(mem, true); err != nil {
				return nil, fmt.Errorf("KV キャッシュのクリアに失敗: %w", err)
			}
		}
	}

	return results, nil
}

// generateOnce は指定した温度のサンプラーを作り、tokens から1回分の生成を行う。
// サンプラーは呼び出しごとに作り直し、使い終わったら必ず解放する。
func generateOnce(lctx llama.Context, vocab llama.Vocab, model llama.Model, tokens []llama.Token, temp float32) (string, error) {
	// 7. サンプラーを作る。temp が高いほど毎回の生成結果がばらつく。
	sp := llama.DefaultSamplerParams()
	sp.Temp = temp
	samplers := []llama.SamplerType{llama.SamplerTypeTopK, llama.SamplerTypeTopP, llama.SamplerTypeTemperature}
	sampler := llama.NewSampler(model, samplers, sp)
	defer llama.SamplerFree(sampler)

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
