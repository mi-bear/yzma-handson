// menu は yzma を使ってローカル LLM に献立を提案してもらう最小例。
// 冷蔵庫の中身を標準入力から渡すと、家にある一般的な調味料だけを使う前提で
// 人数と件数を指定して夕食のメニュー案を提案してもらう。
// ../summarize/ と同じ7ステップ構成で、mode による切り替えの代わりに
// buildUserPrompt でユーザープロンプトを組み立てる点が異なる。
//
// 使い方:
//
//	echo "卵、キャベツ、豚こま、豆腐" | go run . -people 2 -meals 3
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
	defaultPeople    = 2                           // -people 未指定時の人数
	defaultMeals     = 3                           // -meals 未指定時の提案件数

	tokenPieceBufSize   = 256  // 1トークン分の文字列を受け取るバッファサイズ
	templateBufOverhead = 1024 // チャットテンプレート適用後に増える分の余裕
)

// systemPrompt は献立提案アシスタントとしての役割をモデルに与える。
// mode による切り替えが無いので、summarize/ の systemPrompts マップの代わりに
// 単一の定数として持つ。
const systemPrompt = "あなたは家庭料理の献立を考えるアドバイザーです。" +
	"与えられた食材と、家に常備している一般的な調味料 ( 塩、こしょう、しょうゆ、みそ、砂糖、酒、油など ) だけを使い、" +
	"それ以外の食材は使わずに作れる夕食のメニューを考えてください。"

func main() {
	modelFile := flag.String("model", "", "GGUF モデルファイルのパス ( 省略時は "+download.DefaultModelsDir()+" 配下の "+defaultModelFile+" を使う )")
	people := flag.Int("people", defaultPeople, "何人分の献立を考えるか")
	meals := flag.Int("meals", defaultMeals, "いくつメニュー案を出すか")
	flag.Parse()

	if *people < 1 {
		fmt.Fprintln(os.Stderr, "-people は 1 以上を指定してください")
		os.Exit(1)
	}
	if *meals < 1 {
		fmt.Fprintln(os.Stderr, "-meals は 1 以上を指定してください")
		os.Exit(1)
	}

	// 標準入力から冷蔵庫の食材リストを読み込む。
	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintln(os.Stderr, "標準入力の読み込みに失敗:", err)
		os.Exit(1)
	}
	ingredients := strings.TrimSpace(string(input))
	if ingredients == "" {
		fmt.Fprintln(os.Stderr, "標準入力が空です。冷蔵庫にある食材をカンマや読点区切りで渡してください")
		os.Exit(1)
	}

	userPrompt := buildUserPrompt(ingredients, *people, *meals)

	result, err := run(resolveModelPath(*modelFile), systemPrompt, userPrompt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "推論に失敗:", err)
		os.Exit(1)
	}

	fmt.Println(result)
}

// buildUserPrompt はユーザーが渡した食材リストと人数・件数から、
// モデルに渡すユーザープロンプトの文字列を組み立てる。
func buildUserPrompt(ingredients string, people, meals int) string {
	return fmt.Sprintf(
		"冷蔵庫にある食材: %s\n"+
			"%d人分の夕食メニューを%d個提案してください。\n"+
			"出力は「番号. 料理名 - 一言理由」の形式で、%d行だけ出力してください。",
		ingredients, people, meals, meals,
	)
}

// resolveModelPath は -model フラグの値を解決する。
// 未指定 ( 空文字 ) の場合は download.DefaultModelsDir() 配下の既定モデルを使う。
func resolveModelPath(flagValue string) string {
	if flagValue != "" {
		return flagValue
	}
	return filepath.Join(download.DefaultModelsDir(), defaultModelFile)
}

// libPathFromEnv は共有ライブラリ ( libllama ) のパスを環境変数から取得する。
// 未設定の場合は、設定方法を示すエラーを返す。
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
	//    パスは環境変数 YZMA_LIB から取得する。
	libPath, err := libPathFromEnv(os.Getenv)
	if err != nil {
		return "", err
	}
	if err := llama.Load(libPath); err != nil {
		return "", fmt.Errorf("ライブラリのロード失敗 ( YZMA_LIB を確認 ): %w", err)
	}
	llama.LogSet(llama.LogSilent()) // llama.cpp の冗長ログを抑制

	// 2. ランタイムを初期化。defer で確実に後始末する。
	llama.Init()
	defer llama.Close()

	// 3. モデルをファイルから読み込む。
	model, err := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	if err != nil {
		return "", fmt.Errorf("モデル読み込み失敗: %w", err)
	}
	defer llama.ModelFree(model)

	vocab := llama.ModelGetVocab(model)

	// 4. 推論コンテキストを作る。コンテキスト長を明示的に指定する。
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = contextTokens
	lctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		return "", fmt.Errorf("コンテキスト初期化失敗: %w", err)
	}
	defer llama.Free(lctx)

	// 5. サンプラーを作る。突飛な食材の組み合わせを提案されても困るので温度は低め。
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
		// llama_decode の戻り値は int32 で、0 以外は失敗を意味する。
		code, err := llama.Decode(lctx, batch)
		if err != nil {
			return "", fmt.Errorf("デコード失敗: %w", err)
		}
		if code != 0 {
			return "", fmt.Errorf("デコード失敗: llama_decode がコード %d を返した", code)
		}
		token := llama.SamplerSample(sampler, lctx, -1)

		// EOG ( End Of Generation ) トークンが来たら生成終了。
		if llama.VocabIsEOG(vocab, token) {
			break
		}

		buf := make([]byte, tokenPieceBufSize)
		n := llama.TokenToPiece(vocab, token, buf, 0, false)
		if n < 0 {
			return "", fmt.Errorf("トークンの文字列化に失敗した ( token=%d )", token)
		}
		response.Write(buf[:n])

		// 直前に生成したトークンを次の入力にして続きを生成する。
		batch = llama.BatchGetOne([]llama.Token{token})
	}

	return response.String(), nil
}

// applyTemplate はモデルに埋め込まれたチャットテンプレートを適用して、
// モデルが期待する形式のプロンプト文字列を作る。
// ChatMessage.Content は C 文字列 ( *byte ) なので、バッファサイズは
// 元のメッセージ文字列の合計長 ( contentLen ) から計算する。
func applyTemplate(model llama.Model, messages []llama.ChatMessage, contentLen int) (string, error) {
	tmpl := llama.ModelChatTemplate(model, "")
	if tmpl == "" {
		// テンプレートを持たないモデルでは chatml を使う
		tmpl = "chatml"
	}

	// バッファは入力長に応じて十分大きく確保する。
	// llama.ChatApplyTemplate の戻り値 n は「実際に必要なバイト数」で、
	// n <= 0 なら適用失敗、n > len(buf) ならバッファ不足 ( 中身は切り詰められている ) を意味する。
	// pkg/llama/vocab.go の Tokenize と同じように、足りなければ必要なサイズで取り直してもう一度呼ぶ。
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
