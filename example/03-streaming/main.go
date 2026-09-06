// この例は練習問題 3 ( ストリーミング出力 ) の解答例。
// ../summarize/ では生成したトークンを strings.Builder に貯めてから最後にまとめて
// 出力していたが、ここではトークンが出るたびに bufio.Writer へ即座に書き出す。
// 文字が順に出てくるので、待ち時間が短く感じられる。
//
// 使い方:
//
//	echo "要約したい文章" | go run . -mode summary
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
)

// 学習用に定数として持っておく。実運用ではフラグや設定ファイルで指定できるようにするのが良い。
const (
	contextTokens   = 4096 // モデルが一度に扱えるトークン数の上限
	maxOutputTokens = 512  // 生成する応答の最大トークン数

	defaultModelFile = "gemma-3-1b-it-Q4_K_M.gguf" // -model 未指定時に使う既定モデル

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

	if err := run(resolveModelPath(*modelFile), system, string(input), os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "推論に失敗:", err)
		os.Exit(1)
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

// run はモデルをロードして1回の推論を実行し、生成されたトークンを
// out へ逐次書き出す。まとまった文字列を返さない点が summarize/ との違い。
// 戻り値を名前付きにしているのは、末尾の Flush の失敗を見逃さずに
// 呼び出し元へ伝えるため ( 生成そのものは成功していても Flush は失敗しうる )。
func run(modelFile, system, userText string, out io.Writer) (err error) {
	// 1. 共有ライブラリ ( libllama ) をロードする。
	libPath, err := libPathFromEnv(os.Getenv)
	if err != nil {
		return err
	}
	if err := llama.Load(libPath); err != nil {
		return fmt.Errorf("ライブラリのロード失敗 ( YZMA_LIB を確認 ): %w", err)
	}
	llama.LogSet(llama.LogSilent())

	// 2. ランタイムを初期化。
	llama.Init()
	defer llama.Close()

	// 3. モデルをファイルから読み込む。
	model, err := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	if err != nil {
		return fmt.Errorf("モデル読み込み失敗: %w", err)
	}
	defer llama.ModelFree(model)

	vocab := llama.ModelGetVocab(model)

	// 4. 推論コンテキストを作る。
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = contextTokens
	lctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		return fmt.Errorf("コンテキスト初期化失敗: %w", err)
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
		return err
	}

	// 7. プロンプトをトークン化して推論ループを回す。
	//    トークンが確定するたびに bufio.Writer へ書き出し、最後に Flush する。
	tokens := llama.Tokenize(vocab, prompt, true, true)
	batch := llama.BatchGetOne(tokens)

	// エンコーダー・デコーダー型のモデル ( 例: T5 ) はデコードの前にエンコードが必要。
	// gemma はデコーダーだけのモデルなので、この分岐は通らない。
	if llama.ModelHasEncoder(model) {
		code, err := llama.Encode(lctx, batch)
		if err != nil {
			return fmt.Errorf("エンコード失敗: %w", err)
		}
		if code != 0 {
			return fmt.Errorf("エンコード失敗: llama_encode がコード %d を返した", code)
		}

		start := llama.ModelDecoderStartToken(model)
		if start == llama.TokenNull {
			start = llama.VocabBOS(vocab)
		}
		batch = llama.BatchGetOne([]llama.Token{start})
	}

	w := bufio.NewWriter(out)
	defer func() {
		// 生成ループ自体が既にエラーで抜けている場合は、そのエラーを優先する。
		if ferr := w.Flush(); ferr != nil && err == nil {
			err = fmt.Errorf("出力のフラッシュに失敗: %w", ferr)
		}
	}()

	for pos := int32(0); pos < maxOutputTokens; pos += batch.NTokens {
		code, err := llama.Decode(lctx, batch)
		if err != nil {
			return fmt.Errorf("デコード失敗: %w", err)
		}
		if code != 0 {
			return fmt.Errorf("デコード失敗: llama_decode がコード %d を返した", code)
		}
		token := llama.SamplerSample(sampler, lctx, -1)

		// EOG ( End Of Generation ) トークンが来たら生成終了。
		if llama.VocabIsEOG(vocab, token) {
			break
		}

		buf := make([]byte, tokenPieceBufSize)
		n := llama.TokenToPiece(vocab, token, buf, 0, false)
		if n < 0 {
			return fmt.Errorf("トークンの文字列化に失敗した ( token=%d )", token)
		}
		if _, err := w.Write(buf[:n]); err != nil {
			return fmt.Errorf("出力の書き込みに失敗: %w", err)
		}

		// 直前に生成したトークンを次の入力にして続きを生成する。
		batch = llama.BatchGetOne([]llama.Token{token})
	}

	if _, err := w.WriteString("\n"); err != nil {
		return fmt.Errorf("出力の書き込みに失敗: %w", err)
	}

	return nil
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
