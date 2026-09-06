# ハンズオン: yzma でローカル LLM テキスト要約・分類 CLI を作る

Go の標準的な知識があれば読める、ローカル AI 入門ハンズオンです。
外部の AI サーバーや API キーは一切使いません。手元のマシンだけで完結します。

- 対象: Go の基本 (`go run` / `go mod` / 関数・構造体) が分かる人。AI/LLM は初めてで OK。
- ゴール: 標準入力のテキストを「要約」または「分類」する CLI を自分で動かす。
- 所要時間の目安: 30 分前後 (モデルのダウンロード時間を除く)。

> 注意: 本教材は yzma v1.25.0 (2026-08-25 リリース) と llama.cpp b10645 の組み合わせで動作確認しています。
> 最新の正確な手順は必ず [公式リポジトリ](https://github.com/hybridgroup/yzma) と
> [INSTALL.md](https://github.com/hybridgroup/yzma/blob/main/INSTALL.md) を確認してください。

---

## 0. yzma とは (30 秒で)

[yzma](https://github.com/hybridgroup/yzma) は、Go から `llama.cpp` を呼んでローカル LLM 推論をするライブラリ + CLI です。
ポイントは次の 3 つです。

- **CGo 不要**: `purego` と `ffi` で共有ライブラリを呼ぶので、C コンパイラなしで `go build` できる。
- **インプロセス**: Ollama のような別サーバーを立てず、アプリのプロセス内で直接推論する。
- **GGUF モデル対応**: Hugging Face にある大量の軽量モデルをそのまま使える。

```
[あなたの Go アプリ] --(purego/ffi)--> [libllama.dylib] --> [GGUF モデル]
```

このほか、画像を扱う VLM (Vision Language Model) や、TinyGo と WebAssembly でブラウザの中で動かす仕組みもありますが、本教材ではテキストの推論だけを扱います。

---

## 1. 準備 — インストールするもの

### 1-0. Go のバージョン

yzma v1.25.0 は `go 1.26.0` 以上を要求します。手元の Go を確認してください。

```sh
go version   # go1.26 以上であること
```

### 1-1. yzma CLI

```sh
go install github.com/hybridgroup/yzma@latest
```

`yzma` コマンドが使えるか確認します (`$GOPATH/bin` か `~/go/bin` に PATH を通しておくこと)。

```sh
yzma version
```

### 1-2. llama.cpp 共有ライブラリ

yzma が `llama.cpp` のプリビルドライブラリを自動ダウンロードしてくれます。
好きな置き場所を決めて実行します (例として `~/yzma/lib`)。

```sh
yzma install --lib ~/yzma/lib --version b10645
```

> **`--version b10645` を必ず付けてください。** バージョンを省略すると `llama.cpp` の最新ナイトリービルドが入りますが、
> 2026-08-27 以降のビルドは `llama_model_params` 構造体に `lazy_mode` が追加されていて、yzma v1.25.0 とは構造体のレイアウトが合いません。
> その状態でモデルを読み込むと、次のアサートで落ちます。
>
> ```
> src/llama-model.cpp:1735: GGML_ASSERT(!ml.no_alloc) failed
> ```
>
> b10645 は追加直前のビルドで、本教材はこの組み合わせで動作確認しています。
> yzma の次のリリース (main ブランチでは対応済み) が出たら、`--version` なしでも動くようになる見込みです。

完了後、環境変数 `YZMA_LIB` にそのパスを設定します。サンプルコードはここを見てライブラリを探します。

```sh
# fish の場合
set -Ux YZMA_LIB ~/yzma/lib

# bash/zsh の場合
export YZMA_LIB=~/yzma/lib
```

> macOS では `libllama.dylib` などの `.dylib` が置かれます。`yzma install` 実行後に表示される
> OS 別の追加指示があれば、それにも従ってください。
> Linux で NVIDIA GPU (CUDA) や AMD GPU (ROCm) を使う場合は `--processor cuda` / `--processor rocm` を付けます。
> ドライバが入っていれば自動検出もされます。詳細は公式の INSTALL.md を参照してください。

### 1-3. モデル (GGUF) をダウンロード

軽くて、指示に従って答えられる `gemma-3-1b-it` を使います (約 800MB)。

```sh
yzma model get -u https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
```

ダウンロード先は yzma の既定モデルディレクトリ `~/models` です (`-o` で変更できます)。
ファイルが置かれたことを確認しておきます。

```sh
ls ~/models
yzma model info -m ~/models/gemma-3-1b-it-Q4_K_M.gguf   # モデルのメタ情報を表示
```

> マシンが非力なら、超軽量の `SmolLM2-135M-Instruct` (約 100MB) でも動きます。ただし要約・分類の精度は落ちます。
> `yzma model get -u https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf`

---

## 2. プロジェクトを取得して動かす

このリポジトリの `example/` は 1 つの Go モジュールで、プログラムごとにフォルダが分かれています。

| フォルダ | 内容 |
| --- | --- |
| `example/summarize/` | 本編。標準入力のテキストを要約・分類する CLI |
| `example/menu/` | 応用例。冷蔵庫の食材から晩ごはんの献立を提案する CLI |
| `example/01-translate/` 〜 `example/06-jinja-template/` | 練習問題の解答例 (4 章) |
| `example/testdata/` | 生活の場面を題材にした入力サンプル |

`go.mod` は yzma v1.25.0 を指定済みで、`go.sum` も含めています。

```sh
cd example
go build ./...   # 初回は依存モジュールのダウンロードが走る
```

### 2-1. 町内会のお知らせを要約する

`testdata/chonaikai.txt` は、回覧板でよく見る町内会の清掃活動のお知らせです。
`-model` を省略すると `~/models/gemma-3-1b-it-Q4_K_M.gguf` を使います。

```sh
go run ./summarize -mode summary < testdata/chonaikai.txt
```

実際の出力 (macOS, gemma-3-1b-it-Q4_K_M。モデルの出力は毎回少しずつ変わります):

```
桜台町内会は、10月19日午前9時から清掃活動を行います。桜台公園の時計台前集合場所で、軍手とゴミ袋を用意します。雨天の場合は翌週26日に延期し、清掃後お茶菓子を用意します。粗大ゴミの臨時回収も実施します。
```

「3 行以内」という指示は守れていませんが、日付・場所・持ち物・延期の条件は拾えています。1B (パラメータ数 10 億) のモデルの実力はだいたいこのくらいです。

学校からの連絡 (`testdata/school-notice.txt`) でも試してみてください。持ち物や集合時間が拾えているかを見ると、要約の良し悪しが分かります。

### 2-2. お店への問い合わせを分類する

`testdata/inquiry-*.txt` は、家電や宅配サービスへの問い合わせ文です。人が読めば「質問」「要望」「苦情」と分かる 3 通を用意しました。

```sh
go run ./summarize -mode classify < testdata/inquiry-question.txt    # 出力: 質問
go run ./summarize -mode classify < testdata/inquiry-request.txt     # 出力: 要望
go run ./summarize -mode classify < testdata/inquiry-complaint.txt   # 出力: 苦情
```

分類のシステムプロンプトは、ラベルの定義と例文を含む少し長いものになっています (例文を見せる書き方を few-shot と呼びます)。
最初は「[質問/要望/苦情/その他] のいずれか 1 つに分類し、ラベルだけを出力してください」という 1 行だけで試しましたが、
gemma-3-1b-it はこの 3 通をすべて「苦情」と答えました。定義と例を足したところ、3 回ずつ繰り返しても全問正解で安定しました。
コードは 1 文字も変えていません。これが実務で言う「プロンプトを調整する」作業です。

自分で書いた文を渡すこともできます。

```sh
echo "先週買った商品がまだ届きません。いつ発送されますか？" | go run ./summarize -mode classify
```

### 2-3. 冷蔵庫の中身から献立を考える

`example/menu/` は、要約・分類と同じ 7 ステップの構成で、プロンプトだけを「献立の提案」に変えたプログラムです。
食材のリストを渡すと、人数と提案する献立の数に合わせて献立を提案します。

```sh
go run ./menu -people 2 -meals 3 < testdata/fridge.txt
```

実際の出力:

```
1. 豚こま切れ肉とキャベツの炒め物 - 豚肉とキャベツを炒め合わせることで、シンプルながらも美味しい一品に。豆腐としめじを加えて、栄養バランスもアップ。
2. 卵と豆腐の和風パスタ - 卵と豆腐をたっぷり使ったパスタは、手軽に作れて栄養も満点。牛乳で煮込むことで、より美味しくなる。
3. しめじと豚肉の味噌汁 - 豆腐と豚肉を煮込んで味噌汁にすることで、温まるだけでなく栄養も満点。長ねぎを添えて、彩りも豊かに。
```

「パスタ」は冷蔵庫に無い食材なので、指示を完全には守れていません。`menu/main.go` のシステムプロンプトを直して、指示を守らせる方法を考えてみてください。

別の場所に置いたモデルを使うときは `-model` でフルパスを渡します。

```sh
go run ./summarize -mode summary -model ~/models/SmolLM2-135M-Instruct-Q4_K_M.gguf < testdata/chonaikai.txt
```

### 2-4. テストを実行する

モデルもライブラリも使わないユニットテストが付いています。環境が整っていなくても実行できます。

```sh
go test ./...
```

同じ内容 (gofmt、`go vet`、`go test`、`modernize`) を GitHub Actions でも PR ごとに実行しています (`.github/workflows/go.yml`)。

`YZMA_LIB` とモデルが揃っていれば、実際に推論を 1 回走らせる結合テストも実行できます。
`integration` ビルドタグを付けたときだけ動き、環境が無ければスキップします。

```sh
go test -tags integration ./...
```

動作確認した環境 (Apple Silicon の Mac) では、結合テストは初回 12 秒ほどで PASS しました。モデルファイルが OS にキャッシュされた 2 回目以降は 1 秒ほどで終わります。

---

## 3. コードを読む — 推論 7 ステップ

`example/summarize/main.go` の `run()` 関数が本体です。LLM 推論はだいたいどのライブラリでも同じ流れで、yzma では次の 7 ステップになります。

1. **ライブラリのロード** — `llama.Load(libPath)` で共有ライブラリを読み込む。パスは `YZMA_LIB` から取る。
2. **初期化** — `llama.Init()`。`defer llama.Close()` で後始末。
3. **モデル読み込み** — `llama.ModelLoadFromFile(path, params)`。
4. **コンテキスト作成** — `llama.InitFromModel(model, params)`。会話の作業領域。`NCtx` で長さを指定。
5. **サンプラー作成** — どう次の単語を選ぶかの戦略。要約・分類は安定させたいので `Temp` を低く (0.2)。
6. **プロンプト整形** — `system` (役割指示) と `user` (入力) を `ChatApplyTemplate` でモデル指定の形式に変換。
7. **生成ループ** — トークンを 1 つずつ生成し、終了トークン (EOG) が出るまで繰り返す。

モデル・コンテキスト・サンプラーは C 側のリソースなので、それぞれ `ModelFree` / `Free` / `SamplerFree` を `defer` で解放しています。

生成ループの手前には `ModelHasEncoder` の分岐があります。T5 のようにエンコーダーとデコーダーの両方を持つモデルでは、
先に `Encode` を呼んでからデコーダーの開始トークンで生成を始める必要があるためです。
Gemma はデコーダーだけのモデルなので、このハンズオンではこの分岐は通りません。

### ポイントは「プロンプトの差し替えだけで動作が変わる」こと

`main.go` の `systemPrompts` を見てください。要約と分類で**コードは全く同じ**で、
モデルに渡すシステムプロンプト (役割指示の文章) だけが違います。

```go
var systemPrompts = map[string]string{
	"summary":  "...3行以内に要約してください...",
	"classify": classifyPrompt, // ラベルの定義と例文を含む長いプロンプト (定数)
}
```

これが LLM アプリの面白さです。新しいタスクを足したいときは、関数を書くのではなく
**プロンプトを 1 つ追加するだけ**で済むことが多いのです。

### 生成ループの中身

```go
for pos := int32(0); pos < maxOutputTokens; pos += batch.NTokens {
	code, err := llama.Decode(lctx, batch)          // モデルに通す
	if err != nil || code != 0 {                    // 0 以外は llama_decode の失敗
		return "", fmt.Errorf("デコード失敗: ...")
	}
	token := llama.SamplerSample(sampler, lctx, -1) // 次トークンを1つ選ぶ
	if llama.VocabIsEOG(vocab, token) {             // 終了トークンなら打ち切り
		break
	}
	// トークンを文字列に直して貯める
	buf := make([]byte, tokenPieceBufSize)
	n := llama.TokenToPiece(vocab, token, buf, 0, false)
	response.Write(buf[:n])
	// いま出したトークンを次の入力にして続きを生成
	batch = llama.BatchGetOne([]llama.Token{token})
}
```

(実際のコードはエラー処理をもう少し細かく分けています。)

LLM は「次に来る 1 トークンを予測する」のをひたすら繰り返しているだけだ、と実感できます。
`llama.Decode` は Go 側のエラーと `llama_decode` の戻りコードの両方を返すので、
コンテキスト長を超えた場合などはここで検出できます。

### テストしやすい形に切り出した関数

`main.go` には、モデルやライブラリに触れない小さな関数が 3 つあります。

| 関数 | 役割 |
| --- | --- |
| `systemPromptFor(mode)` | `-mode` の値からシステムプロンプトを引く。未知の値はエラー |
| `resolveModelPath(flagValue)` | `-model` が空なら `~/models` 配下の既定モデルにする |
| `libPathFromEnv(getenv)` | `YZMA_LIB` を読む。未設定なら設定方法を含むエラー |

`main_test.go` はこの 3 つだけをテストします。推論そのものは環境依存が大きいので、
入力の解釈のような純粋なロジックを分けておくと、CI でも回せるテストになります。

---

## 4. 練習問題 (ステップアップ)

まず `example/summarize/main.go` を自分で書き換えて挑戦してください。
解答例は `example/` 配下のフォルダにあります。どれも `summarize/` と同じ構成の完成したプログラムで、
先頭のコメントに「何を変えたか」を書いてあります。`diff` で本編と見比べると、変えた箇所が分かります。

```sh
cd example
diff summarize/main.go 01-translate/main.go
```

| # | 問題 | 解答例 | 試し方 |
| --- | --- | --- | --- |
| 1 | **新しいモード追加**: `systemPrompts` に `"translate"` (英訳) を足す。コードは 1 行追加で済むはず。テストは何もしなくても通ることを確認する | `01-translate/` | `go run ./01-translate -mode translate < testdata/school-notice.txt` |
| 2 | **温度を上げてみる**: `sp.Temp` を `0.2` → `0.9` にして、出力のブレ方を比べる | `02-temperature/` | `go run ./02-temperature -temp 0.9 -runs 3 < testdata/chonaikai.txt` |
| 3 | **出力をストリーミング表示**: いまは全部貯めてから出力している。ループ内で `fmt.Print` して逐次表示にする | `03-streaming/` | `go run ./03-streaming < testdata/chonaikai.txt` |
| 4 | **ファイル入力対応**: 標準入力だけでなく `-file` フラグでファイルを読めるようにする | `04-file-input/` | `go run ./04-file-input -file testdata/school-notice.txt` |
| 5 | **モデル比較**: `gemma-3-1b-it` と `SmolLM2-135M-Instruct` で同じ文章を要約し、品質と速度の差を観察する。動作確認では SmolLM2 は要約せずにシステムプロンプトをそのまま繰り返した。小型モデルの限界が見える | `05-model-compare/` | `go run ./05-model-compare -models $HOME/models/gemma-3-1b-it-Q4_K_M.gguf,$HOME/models/SmolLM2-135M-Instruct-Q4_K_M.gguf < testdata/chonaikai.txt` |
| 6 | **Jinja テンプレートを使う**: yzma の `pkg/template` には `gemma3` や `chatml` の組み込みテンプレートがある。`ChatApplyTemplate` の代わりに使う | `06-jinja-template/` | `go run ./06-jinja-template < testdata/chonaikai.txt` |

さらに進みたい人向けの題材です (解答例はありません)。

- `menu/` に `-avoid` フラグを足して、苦手な食材を除外する。
- `summarize/` の分類ラベルを、自分の家に届く郵便物の仕分け (請求書 / 広告 / 学校 / 自治体) に変える。
- 献立の出力を JSON にして、買い物リストを別のプログラムで作る。

---

## 5. よくあるエラー

| 症状 | 原因と対処 |
| --- | --- |
| `環境変数 YZMA_LIB が未設定です` | `YZMA_LIB` を `yzma install --lib` の場所に設定する |
| `ライブラリのロード失敗` | `YZMA_LIB` のパスが誤り。`ls $YZMA_LIB` に `libllama.dylib` (Linux は `.so`) があるか確認する |
| `GGML_ASSERT(!ml.no_alloc) failed` で落ちる | `llama.cpp` が新しすぎる。`yzma install --lib ~/yzma/lib --version b10645 --upgrade` で入れ直す |
| `モデル読み込み失敗` | モデルファイルが無い。`ls ~/models` で確認し、必要なら `-model` でパスを渡す |
| `go: ... requires go >= 1.26.0` | Go が古い。1.26 以上に更新する |
| `デコード失敗` | 入力がコンテキスト長を超えている可能性。`contextTokens` を増やすか入力を短くする |
| 出力が途中で切れる | `maxOutputTokens` を増やす |
| 要約がおかしい/関係ない | モデルが小さすぎる可能性。より大きいモデルを試す |

---

## まとめ

- yzma を使うと、**API キーもサーバーもなし**で Go アプリにローカル AI を組み込める。
- LLM 推論は「ロード → コンテキスト → サンプラー → プロンプト → 生成ループ」の定型。
- 要約も分類も**プロンプト次第**。コードはほぼ共通。

次の一歩としては、この CLI を HTTP サーバーに組み込んで「社内文書を要約する小さな API」にしてみると実践的です。
