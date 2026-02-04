# JacRED KG Extraction: Relation-Split Multi-Pass Extraction

日本語文書レベル関係抽出データセット **JacRED** を用いた、低コストLLM（Gemini Flash系）による知識グラフトリプル抽出の実験リポジトリ。関係タイプを意味グループに分割し、グループ別に複数パスで抽出する **Relation-Split Multi-Pass Extraction** 手法を提案・検証する。

---

## 目次

1. [概要](#1-概要)
2. [背景・動機](#2-背景動機)
3. [データセット](#3-データセット)
4. [実験設定](#4-実験設定)
5. [手法](#5-手法)
6. [結果](#6-結果)
7. [分析](#7-分析)
8. [再現方法](#8-再現方法)
9. [ファイル構成](#9-ファイル構成)
10. [参考文献](#10-参考文献)

---

## 1. 概要

本プロジェクトは、日本語文書群からエンティティ（固有表現）とエンティティ間の関係（relation）を抽出し、知識グラフ（Knowledge Graph）のトリプル `(head_entity, relation, tail_entity)` を自動構築する手法の実験である。

Google Gemini Flash系モデルの **Structured Outputs**（JSON Schema強制出力）を活用し、以下の2条件を比較する:

- **Baseline（One-shot抽出）**: 1回のLLM呼び出しでエンティティと関係を同時に抽出
- **Relation-Split Multi-Pass Extraction**: 35種類の関係タイプを5つの意味グループに分割し、グループごとに専用プロンプトでLLMを呼び出す。各パスはそのグループの関係タイプのみに集中して抽出を行い、全パスの結果を統合した後にdomain/range型制約でフィルタリングする

評価はJacRED devセットから選択した10文書に対して、文書レベル関係抽出の標準指標（Precision / Recall / F1）で行う。

**仮説**: 35種類の関係タイプを一度に処理するとLLMの注意が分散し、特に出現頻度の低い関係タイプが見落とされる。関係タイプを意味的に近いグループに分割し、各パスで少数の関係タイプのみを提示することで、LLMの注意を集中させRecallを改善できると期待した。

**結果概要**: Relation-Split手法はBaselineに対してRecallを改善できず、むしろ悪化した（F1: 0.25 → 0.18）。詳細は[結果](#6-結果)および[分析](#7-分析)を参照。

---

## 2. 背景・動機

### 2.1 文書レベル関係抽出（Document-level Relation Extraction, DocRE）

文書レベル関係抽出（DocRE）は、1つの文書全体を入力として、文書中に出現するエンティティペア間の関係を全て抽出するタスクである。文単位の関係抽出（Sentence-level RE）とは異なり、複数文にまたがる推論や共参照解析が必要となる。

具体的には以下の処理を行う:
1. 文書中のエンティティ（人名、組織名、地名など）を認識する
2. 全てのエンティティペア `(head, tail)` について、既定の関係タイプ集合から該当する関係を判定する
3. 関係が存在しないペアには "NA"（関係なし）を割り当てる

### 2.2 JacREDデータセット

**JacRED**（Japanese Document-level Relation Extraction Dataset）は、英語DocREデータセット **DocRED**（Yao et al., ACL 2019）の構造を日本語Wikipedia記事に適用して構築されたデータセットである。Ma et al.（LREC-COLING 2024）が、cross-lingual transferを活用したアノテーション支援手法により作成した。

### 2.3 本実験の目的

1. **低コストLLMの有効性検証**: 高価なGPT-4系モデルではなく、Gemini Flash系（低コスト・高速）モデルでDocREがどの程度可能かを検証する
2. **Structured Outputsの活用**: 自由形式テキスト出力ではなくJSON Schema強制出力を用い、パースエラーや不正出力を排除する
3. **関係タイプ分割による注意集中**: 35種類の関係タイプを一度に処理するのではなく、意味的に近いグループに分割して各パスで少数の関係タイプのみを提示することで、LLMの注意を集中させRecallを改善できるかを検証する
4. **最終応用**: 日本語文書コレクションからの大規模知識グラフ自動構築への基盤技術確立

---

## 3. データセット

### 3.1 JacRED概要

| 項目 | 内容 |
|---|---|
| 名称 | JacRED (Japanese Document-level Relation Extraction Dataset) |
| ソース | https://github.com/YoumiMa/JacRED |
| 論文 | Ma et al., "Building a Japanese Document-Level Relation Extraction Dataset Assisted by Cross-Lingual Transfer", LREC-COLING 2024 |
| 言語 | 日本語（Wikipedia記事由来） |
| ベース | DocRED（Yao et al., ACL 2019）の構造を日本語に適用 |

### 3.2 データ分割

| 分割 | 文書数 | 用途 |
|---|---|---|
| train | 1,400 | 訓練（本実験ではfew-shot例選択とdomain/range制約テーブル構築に使用） |
| dev | 300 | 開発・評価（本実験では10文書を選択して評価に使用） |
| test | 300 | テスト（本実験では未使用） |

各分割間に文書の重複はない。

### 3.3 データフォーマット

各文書は以下のフィールドを持つJSONオブジェクトである:

```json
{
  "title": "文書タイトル（Wikipedia記事名）",
  "sents": [
    ["トークン1", "トークン2", "..."],
    ["トークン1", "トークン2", "..."]
  ],
  "vertexSet": [
    [
      {"name": "エンティティ名", "type": "PER", "sent_id": 0, "pos": [3, 5]}
    ]
  ],
  "labels": [
    {"h": 0, "t": 1, "r": "P27", "evidence": [0, 2]}
  ]
}
```

**各フィールドの説明:**

- **`title`**: Wikipedia記事のタイトル文字列
- **`sents`**: トークン化済みの文のリスト。各文はトークン（文字列）のリスト。元のテキストは各文のトークンを結合（join）して再構成する
- **`vertexSet`**: エンティティのリスト。各エンティティは1つ以上の **mention**（言及）を持つ。各mentionは:
  - `name`: 言及テキスト（例: "東京都"）
  - `type`: エンティティタイプ（後述の9種類のいずれか）
  - `sent_id`: この言及が出現する文のインデックス（0始まり）
  - `pos`: 文中のトークン位置 `[start, end)`（半開区間）
- **`labels`**: 正解関係ラベルのリスト。各ラベルは:
  - `h`: headエンティティのvertexSetインデックス（0始まり）
  - `t`: tailエンティティのvertexSetインデックス（0始まり）
  - `r`: 関係タイプのPコード（例: "P27"）
  - `evidence`: 根拠となる文のインデックスリスト

### 3.4 エンティティタイプ（9種類）

| タイプコード | 日本語説明 | 例 |
|---|---|---|
| `PER` | 人物 | 織田信長、アインシュタイン |
| `ORG` | 組織 | トヨタ自動車、国連 |
| `LOC` | 場所・地名 | 東京都、ナイル川 |
| `ART` | 作品・人工物・賞 | あずきちゃん、ノーベル賞 |
| `DAT` | 日付 | 1964年5月12日、2011年9月 |
| `TIM` | 時間 | 午前10時 |
| `MON` | 金額 | 100万円 |
| `%` | パーセンテージ・数値 | 50%、3.14 |
| `NA` | 該当なし（未分類） | -- |

注: 本実験のLLMプロンプトでは `NA` を除く8種類をエンティティタイプとして指定する。Structured Outputsのスキーマ（`schemas.py`の`EXTRACTION_SCHEMA`）では `enum: ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]` として8タイプに制限している。

### 3.5 関係タイプ（35種類）

以下はJacREDで定義される全35種類の関係タイプである。各行はWikidataのプロパティコード（Pコード）、英語名、日本語説明を示す。日本語説明は本実験のLLMプロンプト（`prompts.py`の`RELATION_JAPANESE`辞書）で使用されるものと同一である。

| Pコード | English Name | 日本語説明 |
|---|---|---|
| `P1376` | capital of | 首都（〜の首都である） |
| `P131` | located in the administrative territorial entity | 行政区画（〜に位置する行政区画） |
| `P276` | location | 所在地（〜に所在する） |
| `P937` | work location | 活動場所（〜で活動した） |
| `P27` | country of citizenship | 国籍（〜の国籍を持つ） |
| `P569` | date of birth | 生年月日 |
| `P570` | date of death | 没年月日 |
| `P19` | place of birth | 出生地 |
| `P20` | place of death | 死没地 |
| `P155` | follows | 前任・前作（〜の後に続く） |
| `P40` | child | 子（〜の子である） |
| `P3373` | sibling | 兄弟姉妹 |
| `P26` | spouse | 配偶者 |
| `P1344` | participant in | 参加イベント（〜に参加した） |
| `P463` | member of | 所属（〜に所属する） |
| `P361` | part of | 上位概念（〜の一部である） |
| `P6` | head of government | 首長（〜の首長である） |
| `P127` | owned by | 所有者（〜に所有される） |
| `P112` | founded by | 設立者（〜が設立した） |
| `P108` | employer | 雇用主（〜に雇用される） |
| `P137` | operator | 運営者（〜が運営する） |
| `P69` | educated at | 出身校（〜で教育を受けた） |
| `P166` | award received | 受賞（〜を受賞した） |
| `P170` | creator | 制作者（〜が制作した） |
| `P175` | performer | 出演者・パフォーマー |
| `P123` | publisher | 出版社（〜が出版した） |
| `P1441` | present in work | 登場作品（〜に登場する） |
| `P400` | platform | プラットフォーム |
| `P36` | capital | 首都（〜が首都である） |
| `P156` | followed by | 後任・次作（〜の前にある） |
| `P710` | participant | 参加者（〜が参加した） |
| `P527` | has part | 構成要素（〜を含む） |
| `P1830` | owner of | 所有物（〜を所有する） |
| `P121` | item operated | 運営対象（〜を運営する） |
| `P674` | characters | 登場人物（作品の登場人物） |

注: `P1376`（capital of）と`P36`（capital）、`P155`（follows）と`P156`（followed by）、`P361`（part of）と`P527`（has part）、`P127`（owned by）と`P1830`（owner of）、`P137`（operator）と`P121`（item operated）はそれぞれ逆方向の関係ペアである。

### 3.6 データセット統計（参考値）

| 指標 | 値（概算） |
|---|---|
| 平均エンティティ数/文書 | 約17 |
| 平均関係数/文書 | 約20 |
| 平均トークン数/文書 | 約253 |
| 関係密度（関係数 / 可能なペア数） | 約6.5% |

---

## 4. 実験設定

### 4.1 文書選択

#### 評価文書（10文書）

JacRED devセット（300文書）から、文書の文字数（`char_count`）でソートし、等間隔で10文書を選択する **層化サンプリング** を行った。具体的には:

```python
sorted_docs = sorted(dev_data, key=char_count)  # 文字数昇順ソート
total = len(sorted_docs)  # 300
indices = [int(total * (i + 0.5) / 10) for i in range(10)]
# indices = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285]
```

これにより、短い文書から長い文書まで均等にカバーする10文書が選ばれる。

選択された10文書:

| # | タイトル | Gold エンティティ数 | Gold 関係数 |
|---|---|---|---|
| 1 | ダニエル・ウールフォール | 9 | 12 |
| 2 | アンソニー世界を駆ける | 9 | 6 |
| 3 | 青ナイル州 | 11 | 20 |
| 4 | 小谷建仁 | 17 | 11 |
| 5 | 窪田僚 | 18 | 11 |
| 6 | イーオー | 17 | 10 |
| 7 | 堂山鉄橋 | 15 | 9 |
| 8 | 木村千歌 | 21 | 25 |
| 9 | バハン地区 | 18 | 28 |
| 10 | ジョー・ギブス | 31 | 16 |

合計: Gold関係数 = 148

#### Few-shot例（1文書）

**Few-shot prompting（少数例プロンプティング）** とは、LLMに対してタスクの入出力例を少数（1〜数個）プロンプト中に含めることで、タスクの期待フォーマットや振る舞いをモデルに示す手法である。例を0個与える場合を **zero-shot**、1個の場合を **one-shot**、複数個の場合を **few-shot** と呼ぶ。

本実験では **1例（one-shot）** を採用した。これは以下のトレードオフに基づく判断である:
- **コスト・コンテキスト長の制約**: 例を増やすと入力トークン数が増加し、APIコストが上昇する。また、モデルのコンテキストウィンドウ（入力可能なトークン数の上限）を圧迫し、対象文書の処理に使えるトークン数が減少する。
- **品質向上の限界**: 1例でタスク形式を十分に示せる場合、2例以上に増やしても品質改善は限定的であることが多い。
- **例の品質**: 使用する例は**訓練データのGold label（正解ラベル）** から構成される。つまり、人手でアノテーションされた正しい入出力ペアをモデルに見せることで、期待される出力形式と粒度を正確に伝えている。

訓練データから以下の条件を満たす文書を1つ選択する:

- 文字数: 150 - 250文字
- エンティティ数: 5 - 12
- 関係ラベル数: 3 - 15

条件を満たす候補のうち最も短いものを選択する。

選択されたfew-shot文書: **「スタッド (バンド)」**

### 4.2 モデル構成

本実験では以下の1つのモデル構成を使用した:

| 構成名 | モデルID | thinking_budget | 説明 |
|---|---|---|---|
| gemini-3-flash-preview (ON) | `gemini-3-flash-preview` | 2048 | `ThinkingConfig(thinking_budget=2048)` を指定。内部推論有効 |

#### "thinking"（思考モード）とは何か

Gemini 2.5 Flash以降のモデルは **thinking**（内部推論 / 思考モード）機能を持つ。これは **chain-of-thought reasoning（連鎖的思考推論）** をモデル内部に組み込んだ機能であり、従来のプロンプトエンジニアリングで「ステップバイステップで考えてください」と指示する手法（chain-of-thought prompting）とは異なり、モデルのアーキテクチャレベルで推論プロセスが組み込まれている。

**動作原理:**

thinking が有効な場合、モデルはユーザのリクエストに対して以下の2段階で応答を生成する:

1. **内部推論フェーズ（thinking tokens）**: モデルはまず「推論トークン」（reasoning tokens / thinking tokens）を内部的に生成する。これはモデルが問題を分解し、ステップバイステップで考えるためのトークン列である。重要な点として、**これらの推論トークンはAPIレスポンスのテキストには含まれない**（呼び出し側からは見えない）。つまり、ユーザが受け取る最終出力には推論過程は表示されず、結論のみが返される。
2. **最終回答生成フェーズ**: 内部推論が完了した後、モデルはその推論結果を踏まえて最終的な回答を生成する。この回答のみがAPIレスポンスとして返される。

**`thinking_budget` パラメータ:**

`thinking_budget` は `ThinkingConfig` の設定項目で、**モデルが内部推論に使用できるトークンの最大数**を制御する。

- **`thinking_budget=0`**: thinking機能を**完全に無効化**する。モデルは内部推論フェーズをスキップし、通常の（non-reasoning）モデルと同等に動作する。即座に最終回答の生成を開始するため、レイテンシが低い。
- **`thinking_budget=2048`**: モデルが**最大2048トークンの内部推論**を行うことを許可する。ただし、タスクが単純な場合、モデルは2048トークン全てを使い切らず、より少ないトークン数で推論を完了する場合がある（上限であり、必ず使い切るわけではない）。
- **一般的な傾向**: thinking_budgetを大きくすると、(a) レイテンシが増加する（推論トークン生成の時間がかかる）、(b) APIコストが増加する（推論トークンも課金対象として計上される）、(c) 複雑なタスクでは回答品質が向上する可能性がある。

**コードでの指定方法（`llm_client.py`）:**

```python
from google.genai.types import GenerateContentConfig, ThinkingConfig

config = GenerateContentConfig(
    system_instruction=system_prompt,
    response_mime_type="application/json",
    response_schema=response_schema,
    temperature=0.2,
    thinking_config=ThinkingConfig(thinking_budget=2048),
)
```

### 4.3 Structured Outputs（構造化出力）

全てのLLM呼び出しでGoogle GenAI SDKの **Structured Outputs** 機能を使用する。これにより、モデルの出力が指定したJSON Schemaに厳密に従うことが保証される。自由形式テキストの出力やJSONパースエラーは原理的に発生しない。

**Structured Outputsの仕組み:**

通常のLLM呼び出しでは、モデルは自由形式のテキストを生成する。プロンプトで「JSON形式で出力してください」と指示しても、モデルが不正なJSON（閉じ括弧の欠落、余分なテキストの混入など）を出力するリスクがある。Structured Outputsはこの問題を根本的に解決する。

APIリクエストで以下の2つのパラメータを指定する:
- **`response_mime_type="application/json"`**: モデルの出力をJSON形式に強制する。モデルのデコーディングプロセス（トークンを1つずつ選択する過程）において、JSON構文に違反するトークンは選択候補から除外される。これは単なるプロンプト指示ではなく、**デコーディング時のハード制約**（constrained decoding）である。
- **`response_schema=...`**: 出力JSONが準拠すべきJSON Schemaを指定する。スキーマで定義されたフィールド名、型、必須フィールドに違反するトークンはデコーディング時に除外される。

**`enum` 制約の効果:**

本実験では `entities[].type` フィールドに `enum: ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]` を指定している。これにより、モデルはエンティティタイプとしてこの8種類以外の文字列を**物理的に出力できない**。デコーディング時にenum外のトークン列は確率0に設定されるため、「モデルが勝手に新しいタイプを発明する」という問題は原理的に排除される。これはプロンプトで「以下のタイプのみ使用してください」と指示するよりも遥かに信頼性が高い。

#### 抽出用スキーマ（EXTRACTION_SCHEMA）

Baseline・Relation-Splitの両方で使用する。モデルの出力を以下の構造に強制する:

```json
{
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "type": {
            "type": "string",
            "enum": ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]
          }
        },
        "required": ["id", "name", "type"]
      }
    },
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "head": {"type": "string"},
          "relation": {"type": "string"},
          "tail": {"type": "string"},
          "evidence": {"type": "string"}
        },
        "required": ["head", "relation", "tail", "evidence"]
      }
    }
  },
  "required": ["entities", "relations"]
}
```

- `entities[].type` は `enum` 制約により8種類のエンティティタイプのいずれかに強制される
- `relations[].relation` は文字列型だが `enum` 制約はない（後処理で不正なPコードをフィルタする）
- `relations[].head` と `relations[].tail` は `entities[].id` を参照する文字列

### 4.4 API呼び出し設定

**Temperature（温度パラメータ）について:**

LLMがトークン（単語の断片）を1つずつ生成する際、各ステップで次のトークンの確率分布が計算される。**temperature** はこの確率分布の「鋭さ」を制御するパラメータである:
- **temperature=0（またはそれに近い値）**: 確率分布が極端に鋭くなり、最も確率の高いトークンがほぼ確定的に選択される。出力の再現性が高いが、temperature=0では**退化的な繰り返し**（同じフレーズを無限に繰り返す現象）が発生するリスクがある。
- **temperature=1.0**: モデルの学習時の確率分布がそのまま使用される。出力に適度な多様性がある。
- **temperature > 1.0**: 確率分布が平坦化され、低確率のトークンも選択されやすくなる。出力が創造的になるが、不正確な内容が増える。

本実験では **temperature=0.2** を採用した。これは退化的な繰り返しを回避しつつ（temperature=0の問題を避ける）、出力の一貫性と再現性を高める設定である。情報抽出タスクでは創造性より正確性が重要なため、低いtemperatureが適切である。

| パラメータ | 値 | 説明 |
|---|---|---|
| `temperature` | 0.2 | 低めの温度で出力の再現性を高める（上記の解説参照） |
| `max_retries` | 3 | API呼び出し失敗時の最大リトライ回数 |
| リトライ間隔 | 指数バックオフ（2秒、4秒、8秒） | `wait = 2 ** (attempt + 1)` |
| SDK | `google-genai` Python パッケージ | `from google import genai` |
| 認証 | APIキー方式 | 環境変数 `GEMINI_API_KEY` またはファイルから読み込み |

---

## 5. 手法

### 5.1 Baseline: One-shot抽出

1回のLLM呼び出しでエンティティと関係を同時に抽出する。

#### 処理フロー

**Step 1: システムプロンプト構築**

以下の情報を含むシステムプロンプトを構築する:
- タスク説明:「日本語文書から知識グラフ（エンティティと関係）を抽出する」
- エンティティタイプ一覧: 8種類（PER, ORG, LOC, ART, DAT, TIM, MON, %）とその日本語説明
- 関係タイプ一覧: 35種類のPコードと英語名と日本語説明
- ルール: 指定タイプのみ使用、Pコードのみ使用、evidence付与、headとtailにはentities IDを使用

システムプロンプトの実際のテンプレート:
```
あなたは日本語文書から知識グラフ（エンティティと関係）を抽出する専門家です。

## タスク
与えられた日本語文書から、エンティティ（固有表現）とエンティティ間の関係を抽出してください。

## エンティティタイプ（8種類）
  - PER: 人物
  - ORG: 組織
  - LOC: 場所・地名
  - ART: 作品・人工物・賞
  ...（以下略）

## 関係タイプ（35種類、Pコードで指定）
  - P1376 (capital of): 首都（〜の首都である）
  - P131 (located in the administrative territorial entity): 行政区画（〜に位置する行政区画）
  ...（以下略）

## ルール
- エンティティには上記のタイプのみ使用してください。
- 関係には上記のPコード（P131, P27等）のみ使用してください。自由記述は禁止です。
- 各関係には、根拠となる文書中のテキストをevidenceとして付与してください。
- headとtailにはentitiesのidを指定してください。
```

**Step 2: ユーザプロンプト構築**

以下を含むユーザプロンプトを構築する:
1. **Few-shot例**: 訓練データから選択した1文書のテキストと、その正解をJSON形式に変換した期待出力
2. **対象文書**: 抽出対象のテキスト

```
## 例
入力文書:
{few_shot文書のテキスト}

出力:
{few_shot文書の正解をJSON化したもの}

## 対象文書
{対象文書のテキスト}

上記の文書からエンティティと関係を抽出してください。
```

**Step 3: LLM呼び出し**

`EXTRACTION_SCHEMA` を `response_schema` として指定し、Gemini APIを1回呼び出す。レスポンスはJSON形式で `entities` と `relations` を含む。

**Step 4: 後処理（フィルタリング）**

1. **不正関係フィルタ**: `relations` 中の `relation` フィールドが35種類のPコードに含まれないものを除去
2. **不正エンティティタイプフィルタ**: `entities` 中の `type` が8種類のタイプに含まれないものを除去し、そのエンティティを参照する関係も除去

### 5.2 Relation-Split Multi-Pass Extraction

#### 設計の動機

Baselineでは35種類の関係タイプを全てシステムプロンプトに含めて1回のLLM呼び出しで抽出を行う。しかし、LLMのコンテキストウィンドウ内に多数の関係タイプ定義が存在すると、モデルの注意が分散し、特に出現頻度の低い関係タイプが見落とされる可能性がある。

Relation-Split手法では、35種類の関係タイプを**意味的に近い5つのグループ**に分割し、各グループに対して専用のLLM呼び出しを行う。各パスでは対象グループの関係タイプのみがシステムプロンプトに含まれるため、モデルはより少数の関係タイプに集中して抽出を行うことができる。

#### 関係タイプの5グループ分割

| グループ名 | 関係タイプ（Pコード） | 意味的カテゴリ |
|---|---|---|
| **biographical** | P569, P570, P19, P20, P27, P69, P108, P937 | 人物の経歴情報（生年月日、没年月日、出生地、死没地、国籍、出身校、雇用主、活動場所） |
| **family_social** | P40, P3373, P26 | 家族・社会的関係（子、兄弟姉妹、配偶者） |
| **geographic** | P131, P276, P1376, P36, P361, P527 | 地理的・行政的関係（行政区画、所在地、首都、上位概念、構成要素） |
| **organizational** | P463, P6, P112, P127, P137, P1830, P121, P166 | 組織に関する関係（所属、首長、設立者、所有者、運営者、所有物、運営対象、受賞） |
| **creative_media** | P170, P175, P123, P1441, P400, P674, P155, P156, P1344, P710 | 創作・メディアに関する関係（制作者、出演者、出版社、登場作品、プラットフォーム、登場人物、前任・後任、参加イベント、参加者） |

グループ分割の設計基準:
- **意味的一貫性**: 同じグループ内の関係タイプは、同じドメイン（人物、地理、組織など）に属し、抽出時に関連するエンティティが重なりやすい
- **逆方向ペアの統合**: P361（part of）とP527（has part）のように逆方向の関係ペアは同じグループに配置し、両方向の関係を同時に捉えやすくする
- **グループサイズのバランス**: 最小3種類（family_social）から最大10種類（creative_media）まで。1パスあたりの関係タイプ数を大幅に削減する

#### 処理フロー

**Step 1: グループ別システムプロンプト構築**

各グループに対して、そのグループの関係タイプのみを含むシステムプロンプトを構築する。Baselineのシステムプロンプトと同じ構造だが、関係タイプセクションに含まれるのは対象グループのPコードのみである。

さらに、各グループに特化した焦点指示（focus instruction）を追加する:
- biographical: 「この文書から、人物の経歴に関する情報（生年月日、没年月日、出生地、死没地、国籍、出身校、雇用主、活動場所）に注目して抽出してください。」
- family_social: 「この文書から、家族・社会的関係（子、兄弟姉妹、配偶者）に注目して抽出してください。」
- geographic: 「この文書から、地理的関係（行政区画、所在地、首都、上位概念、構成要素）に注目して抽出してください。」
- organizational: 「この文書から、組織に関する関係（所属、首長、設立者、所有者、運営者、所有物、運営対象、受賞）に注目して抽出してください。」
- creative_media: 「この文書から、創作・メディアに関する関係（制作者、出演者、出版社、登場作品、プラットフォーム、登場人物、前任・後任、参加イベント、参加者）に注目して抽出してください。」

**Step 2: グループ別ユーザプロンプト構築**

Few-shot例の出力を対象グループの関係タイプでフィルタする。つまり、few-shot文書の正解ラベルのうち、対象グループのPコードに該当するもののみを含める。対象グループに該当するラベルがない場合は、エンティティのみを含め関係リストは空とする。

```python
# build_group_extraction_prompt() の要点
filtered_relations = [
    r for r in few_shot_output["relations"]
    if r["relation"] in group_pcode_set
]
```

**Step 3: 5回のLLM呼び出し**

各グループに対して1回ずつ、合計5回のLLM呼び出しを行う。各呼び出しでは `EXTRACTION_SCHEMA` を使用し、エンティティと関係の両方を返す。

**Step 4: エンティティ統合（マージ）**

5つのパスの結果を統合する際、異なるパスで同じエンティティが異なるIDで出現する問題に対処する必要がある。エンティティ名をUnicode NFKC正規化 + 小文字化 + 前後空白除去した上で、同一の正規化名を持つエンティティを同一エンティティとして統合する。

```python
# _merge_entities_across_passes() の要点
def _normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).strip().lower()

# 正規化名が同じエンティティは同一IDにマッピング
# トリプルのhead/tail IDも新しいIDに更新
# (head, relation, tail) が同一のトリプルは重複除去
```

**Step 5: フィルタリングとdomain/range制約**

1. **不正関係フィルタ**: 35種類のPコードに含まれない関係を除去
2. **不正エンティティタイプフィルタ**: 8種類に含まれないタイプを除去
3. **Domain/Range型制約**: 訓練データで未観測の `(head_type, tail_type)` ペアを持つトリプルを除去

#### Domain/Range型制約

知識グラフやオントロジーの文脈において、**domain（定義域）** と **range（値域）** は関係（relation / property）に対する型制約を表す用語である:
- **Domain（定義域）**: その関係の **head（主語）** に許容されるエンティティタイプの集合。例えば、「生年月日」（P569）のdomainは `{PER}`（人物のみが生年月日を持つ）。
- **Range（値域）**: その関係の **tail（目的語）** に許容されるエンティティタイプの集合。例えば、「生年月日」（P569）のrangeは `{DAT}`（生年月日の値は日付である）。

したがって、`ORG --[P569]--> LOC`（組織の生年月日が場所である）というトリプルは、domainにもrangeにも違反するため明らかに不正であり、フィルタで除去すべきである。

本実験では、オントロジーで明示的に定義されたdomain/rangeではなく、**訓練データから経験的に観測されたtype pairの集合**を制約として使用する。これにより、厳密なオントロジー定義がなくても、データ駆動で型制約を適用できる。

**制約テーブルの構築方法:**

訓練データ全体（1,400文書）をスキャンし、各関係Pコードについて、実際に出現した `(head_entity_type, tail_entity_type)` ペアの集合を収集する。

```python
# data_loader.py の build_constraint_table() 関数
constraint_table = defaultdict(set)
for doc in train_data:
    for label in doc["labels"]:
        h_type = doc["vertexSet"][label["h"]][0]["type"]  # headの最初のmentionの型
        t_type = doc["vertexSet"][label["t"]][0]["type"]  # tailの最初のmentionの型
        constraint_table[label["r"]].add((h_type, t_type))
```

例: `P27`（国籍）の制約テーブルが `{("PER", "LOC")}` のみであれば、`(ORG, LOC)` のペアで `P27` を持つトリプルは訓練データで一度も観測されていないため除去する。

**フィルタの適用:**

```python
# extraction.py の apply_domain_range_constraints() 関数
for triple in candidates:
    allowed_pairs = constraint_table.get(triple.relation)
    if allowed_pairs is not None:
        if (triple.head_type, triple.tail_type) not in allowed_pairs:
            # 除去（訓練データで観測されていない型ペア）
            continue
    # 保持
```

### 5.3 評価方法

#### エンティティアライメント（予測エンティティ → Gold vertexSet）

予測されたエンティティをGoldデータのvertexSetインデックスに対応付ける。3パスマッチングを以下の優先順で行う:

**Pass 1: 完全一致**
- 予測エンティティの `name` が、GoldのvertexSet中のいずれかのmentionの `name` と完全一致する場合にマッチ

**Pass 2: 正規化一致**
- 予測エンティティの `name` をUnicode NFKC正規化 + 小文字化 + 前後空白除去したものが、Goldのいずれかのmention nameの同様の正規化結果と一致する場合にマッチ

**Pass 3: 部分文字列一致**
- 正規化後の予測名がGold名の部分文字列であるか、またはGold名が予測名の部分文字列である場合にマッチ
- 複数候補がある場合は、重複する文字数（`min(len(pred), len(gold))`）が最大のものを優先
- 最小重複文字数は2文字（1文字のみの一致は無視）

**制約**: 各予測エンティティは最大1つのGoldエンティティにマッチする（1:1マッピング、先着順）。一度マッチしたGoldエンティティは以降のマッチ候補から除外される。

#### 関係評価

- **True Positive (TP)**: 予測トリプル `(head_id, relation, tail_id)` のhead, tailがともにGoldエンティティにアライメント済みで、かつGoldラベルに `(aligned_h_idx, aligned_t_idx, relation)` が存在する
- **False Positive (FP)**: 予測トリプルが以下のいずれかに該当:
  - headまたはtailがGoldエンティティにアライメントされていない（`entity_not_aligned`）
  - アライメントは成功したが、対応するGoldラベルが存在しない（`wrong_relation`）
- **False Negative (FN)**: Goldラベルのうち、いずれの予測トリプルにもマッチしなかったもの

#### 集計

10文書全体で **マイクロ平均（micro-average）** を計算する。

**マイクロ平均 vs マクロ平均:**

評価指標の集計方法には主に2種類ある:
- **マイクロ平均（micro-average）**: 全文書のTP, FP, FNを**合算してから**P/R/F1を計算する。文書ごとの重みは関係数に比例する（関係数の多い文書がスコアに与える影響が大きい）。本実験ではこちらを採用。
- **マクロ平均（macro-average）**: 各文書のP/R/F1を**個別に計算してから平均**する。文書ごとの重みは均等（関係数によらず各文書のスコアが等しく寄与する）。関係数が少ない文書のスコア変動が大きいため、サンプル数が少ない場合は不安定になりやすい。

本実験でマイクロ平均を採用した理由は、(a) DocRE分野の標準的な評価方式であること、(b) 10文書というサンプル数では文書あたりのスコア変動が大きく、マクロ平均は不安定になりやすいことである。

```
Precision = TP_total / (TP_total + FP_total)
Recall    = TP_total / (TP_total + FN_total)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

---

## 6. 結果

### 6.1 集計結果の比較表

| 条件 | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Baseline | 0.30 | 0.21 | 0.25 | 31 | 73 | 117 |
| Relation-Split | 0.21 | 0.16 | 0.18 | 24 | 89 | 124 |

モデル: `gemini-3-flash-preview`, `thinking_budget=2048`

### 6.2 Baseline 文書別結果

| 文書 | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| ダニエル・ウールフォール | 0.64 | 0.58 | 0.61 | 7 | 4 | 5 |
| アンソニー世界を駆ける | 0.43 | 0.50 | 0.46 | 3 | 4 | 3 |
| 青ナイル州 | 0.44 | 0.20 | 0.28 | 4 | 5 | 16 |
| 小谷建仁 | 0.31 | 0.36 | 0.33 | 4 | 9 | 7 |
| 窪田僚 | 0.07 | 0.09 | 0.08 | 1 | 14 | 10 |
| イーオー | 0.20 | 0.10 | 0.13 | 1 | 4 | 9 |
| 堂山鉄橋 | 0.29 | 0.22 | 0.25 | 2 | 5 | 7 |
| 木村千歌 | 0.11 | 0.08 | 0.09 | 2 | 16 | 23 |
| バハン地区 | 0.25 | 0.07 | 0.11 | 2 | 6 | 26 |
| ジョー・ギブス | 0.45 | 0.31 | 0.37 | 5 | 6 | 11 |

### 6.3 Relation-Split 文書別結果

| 文書 | P | R | F1 | TP | FP | FN | 合計候補 | 制約後 |
|---|---|---|---|---|---|---|---|---|
| ダニエル・ウールフォール | 0.14 | 0.08 | 0.11 | 1 | 6 | 11 | 11 | 7 |
| アンソニー世界を駆ける | 0.38 | 0.50 | 0.43 | 3 | 5 | 3 | 9 | 8 |
| 青ナイル州 | 0.50 | 0.20 | 0.29 | 4 | 4 | 16 | 8 | 8 |
| 小谷建仁 | 0.67 | 0.73 | 0.70 | 8 | 4 | 3 | 17 | 12 |
| 窪田僚 | 0.07 | 0.09 | 0.08 | 1 | 14 | 10 | 20 | 15 |
| イーオー | 0.15 | 0.20 | 0.17 | 2 | 11 | 8 | 13 | 13 |
| 堂山鉄橋 | 0.20 | 0.11 | 0.14 | 1 | 4 | 8 | 7 | 5 |
| 木村千歌 | 0.08 | 0.08 | 0.08 | 2 | 23 | 23 | 25 | 25 |
| バハン地区 | 0.10 | 0.04 | 0.05 | 1 | 9 | 27 | 10 | 10 |
| ジョー・ギブス | 0.10 | 0.06 | 0.08 | 1 | 9 | 15 | 22 | 10 |

合計候補 = 5パスの統合・重複除去後の候補数、制約後 = domain/range制約適用後の最終数

### 6.4 Relation-Split グループ別抽出統計

各文書・各グループで抽出されたトリプル数:

| 文書 | biographical | family_social | geographic | organizational | creative_media |
|---|---|---|---|---|---|
| ダニエル・ウールフォール | 5 | 0 | 2 | 4 | 0 |
| アンソニー世界を駆ける | 2 | 0 | 1 | 3 | 3 |
| 青ナイル州 | 0 | 0 | 5 | 1 | 2 |
| 小谷建仁 | 6 | 0 | 2 | 1 | 8 |
| 窪田僚 | 8 | 0 | 3 | 3 | 6 |
| イーオー | 3 | 4 | 2 | 1 | 3 |
| 堂山鉄橋 | 0 | 0 | 5 | 2 | 0 |
| 木村千歌 | 6 | 0 | 3 | 5 | 11 |
| バハン地区 | 0 | 0 | 7 | 2 | 1 |
| ジョー・ギブス | 4 | 0 | 4 | 6 | 8 |
| **合計** | **34** | **4** | **34** | **28** | **42** |

family_socialグループは全文書でほぼ0件であり、家族関係の抽出が特に困難であることが分かる。

---

## 7. 分析

### 7.1 主な知見

1. **Relation-Split手法はBaselineより悪化した**: F1スコアはBaseline 0.25 → RelSplit 0.18に低下。Precision（0.30 → 0.21）、Recall（0.21 → 0.16）ともに悪化した。当初の仮説（関係タイプ分割によるRecall改善）は支持されなかった。

2. **FP数が増加**: Baselineの73件に対してRelSplitは89件（22%増加）。5回のLLM呼び出しの結果を統合するため、各パスで生成される不正な関係が累積してFPが増加した。

3. **エンティティアライメント失敗の増加**: RelSplitでは各パスが独立にエンティティを認識するため、エンティティ名の表記揺れが発生しやすい。例えば、「ジョー・ギブス」文書では「ジョー_ギブス」「アマルガメイテッド_レーベル」「プロフェッショナルズ」などGoldのmention名と一致しないエンティティ名が多数生成され、9件のFPのうち全てが`entity_not_aligned`であった。

4. **文書によって効果が異なる**: 「小谷建仁」文書ではRelSplitがBaselineを大幅に上回った（F1: 0.33 → 0.70、TP: 4 → 8）。一方、「ジョー・ギブス」文書ではRelSplitが大幅に悪化した（F1: 0.37 → 0.08、TP: 5 → 1）。

5. **family_socialグループの空振り**: 家族・配偶者関係（P40, P3373, P26）のグループは10文書中ほぼ全てで0件の抽出結果であった（合計4件のみ）。このグループに含まれる関係がGoldラベルに存在する文書（イーオー: P40が3件）でも正しく抽出できていない。

### 7.2 FP/FNパターン分析

#### Relation-SplitのFPパターン

| FP理由 | Baseline件数 | RelSplit件数 |
|---|---|---|
| entity_not_aligned | 23 | 44 |
| wrong_relation | 50 | 45 |
| **合計** | **73** | **89** |

- `entity_not_aligned`がBaselineの23件からRelSplitの44件に倍増。各パスが独立にエンティティを生成するため、Goldのmention名と異なる表記（ハイフンvs中点、略称、複合表現の分割など）が多く生じた。
- `wrong_relation`はBaselineの50件からRelSplitの45件にわずかに減少。グループ限定によって一部の無関係な関係タイプの誤抽出が抑制されたと考えられる。

#### FNパターン（両条件共通の傾向）

- **暗黙的関係**: 文書中に直接記述されていないが推論で導出できる関係（例: 行政区画の包含関係）
- **多ホップ推論**: 複数文にまたがる推論が必要な関係
- **逆方向関係ペア**: P361 (part of) と P527 (has part) の両方が正解に含まれるケースで、片方しか抽出できない
- **地理的関係の網羅性不足**: 「青ナイル州」（FN=16）「バハン地区」（FN=26/27）のように行政区画関係が多数含まれる文書で、Recall が特に低い

### 7.3 Relation-Split失敗の原因分析

Relation-Split手法がBaselineより悪化した主な原因は以下の3点と考えられる:

1. **エンティティの文脈分断**: 各パスは文書中のエンティティを独立に認識する。Baselineでは全ての関係タイプを考慮しながらエンティティ間の相互関係を総合的に把握できるが、RelSplitでは各パスがグループに関連するエンティティのみに注目する傾向があり、エンティティの認識精度自体が低下する可能性がある。

2. **エンティティ名の表記不一致**: 5つのパスが独立にエンティティ名を生成するため、同一エンティティに対して異なる表記が生じる。正規化名によるマージでは捉えきれない表記差（例: 中点 vs ハイフン vs アンダースコア）が残り、Goldのmentionとの照合時にアライメント失敗を引き起こす。

3. **コスト対効果の悪化**: 5倍のAPI呼び出しコストに対して品質改善がなく、むしろ悪化した。各パスが少数の関係タイプに集中することでRecallが改善されるという仮説は、少なくとも今回の実験設定では成立しなかった。

### 7.4 文書別の詳細分析

**成功事例: 小谷建仁（F1: 0.33 → 0.70）**

- creative_mediaグループが8件のトリプルを抽出し、そのうち大半がTPとなった。この文書は映画監督に関する記事であり、制作者（P170）関係が多く含まれる。RelSplitではcreative_mediaグループが集中的にP170を抽出でき、Baselineで見落としていた関係を捕捉できた。

**失敗事例: ジョー・ギブス（F1: 0.37 → 0.08）**

- 全9件のFPが`entity_not_aligned`。レゲエ音楽プロデューサーに関する記事で、人名にアンダースコア区切り（「ジョー_ギブス」「ロビー_シェイクスピア」「アール_チナ_スミス」）を使用したためGoldの中点区切りと不一致。Baselineでは同じ問題が部分的に発生したが、RelSplitでは5パス全体でこのパターンが増幅された。

**失敗事例: バハン地区（F1: 0.11 → 0.05）**

- Gold関係28件に対してRelSplitは10件しか抽出できず（制約後）、TP=1。geographicグループが7件抽出したが大半がwrong_relation。行政区画の階層構造（地区 → 州 → 国）を正しく把握できていない。

---

## 8. 再現方法

### 8.1 前提条件

- Python 3.10以上
- Google Gemini APIキー（[Google AI Studio](https://aistudio.google.com/)で取得）
- インターネット接続（API呼び出し用）

### 8.2 手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/auto-res2/kg-extraction-relation-split
cd kg-extraction-relation-split

# 2. JacREDデータセットを取得
git clone https://github.com/YoumiMa/JacRED /tmp/JacRED

# 3. 依存パッケージをインストール
pip install google-genai

# 4. APIキーを設定（2つの方法）
# 方法A: 環境変数（推奨）
export GEMINI_API_KEY="your-gemini-api-key"

# 方法B: ファイルから読み込み（デフォルトのパスを変更する場合はrun_experiment.pyのENV_PATHを編集）

# 5. 実験を実行
python3 run_experiment.py
```

**注意**: デフォルトでは `run_experiment.py` の `ENV_PATH` がDropbox上のファイルを参照するよう設定されている。環境変数 `GEMINI_API_KEY` を使用する場合は、`llm_client.py` の `load_api_key()` 呼び出し部分を `os.environ["GEMINI_API_KEY"]` に置き換えるか、`run_experiment.py` の `ENV_PATH` を適切なパスに変更する必要がある。

### 8.3 モデル・thinking設定の変更方法

`llm_client.py` を直接編集する:

```python
# llm_client.py の9行目
MODEL = "gemini-3-flash-preview"  # 変更先: "gemini-2.0-flash", "gemini-2.5-flash", etc.

# llm_client.py の41行目（call_gemini関数内）
thinking_config=ThinkingConfig(thinking_budget=2048),  # 0でOFF、2048でON
# gemini-2.0-flashを使う場合はこの行を削除する（thinking非対応のため）
```

### 8.4 JacREDデータのパス変更

デフォルトでは `/tmp/JacRED/` を参照する。変更する場合は `data_loader.py` の `load_jacred()` 関数の `base_path` 引数を変更する。

### 8.5 実行時間の目安

| 条件 | API呼び出し数/文書 | 概算時間（10文書） |
|---|---|---|
| Baseline | 1回 | 約2分 |
| Relation-Split | 5回 | 約8-10分 |

注: API呼び出しの待機時間に大きく依存するため、上記は概算である。Relation-Splitはグループ数（5）分のAPI呼び出しを行うため、Baselineの約5倍のコストがかかる。

---

## 9. ファイル構成

```
kg-extraction-relation-split/
  run_experiment.py   # メインスクリプト
  data_loader.py      # データ読み込み・選択
  llm_client.py       # Gemini API呼び出し
  prompts.py          # プロンプトテンプレート
  extraction.py       # 抽出ロジック
  evaluation.py       # 評価ロジック
  schemas.py          # JSON Schema定義
  results.json        # 最新の実験結果
  README.md           # 本ファイル
```

### 9.1 `run_experiment.py` -- メインスクリプト

**目的**: 実験全体のオーケストレーション（データ読み込み → 条件実行 → 結果比較・保存）。

**主要関数:**
- `run_condition(name, docs, few_shot, client, schema_info, extraction_fn, constraint_table=None)`:
  - 1つの実験条件（BaselineまたはRelation-Split）を全文書に対して実行し、文書別・集計のP/R/F1を算出する
  - `extraction_fn` が `"baseline"` の場合は `run_baseline()` を呼び出し、`"relation_split"` の場合は `run_relation_split()` を呼び出す
  - 入力: 文書リスト、few-shot例、Geminiクライアント、スキーマ情報、抽出関数名、（任意）制約テーブル
  - 出力: `{"per_doc": [...], "aggregate": {...}}` の辞書
- `main()`:
  - データ読み込み（`load_jacred()`）、文書選択（`select_dev_docs()`）、few-shot選択（`select_few_shot()`）、制約テーブル構築（`build_constraint_table()`）を実行
  - Baseline, Relation-Split の2条件を順に実行し、結果を比較表示
  - `results.json` に全結果を保存

### 9.2 `data_loader.py` -- データ読み込み・選択

**目的**: JacREDデータセットの読み込み、実験用文書の選択、domain/range制約テーブルの構築。

**主要関数:**
- `load_jacred(base_path="/tmp/JacRED/") -> dict`:
  - train/dev/test の3分割JSONと、メタデータ（rel2id, ent2id, rel_info）を読み込む
  - 出力: `{"train": [...], "dev": [...], "test": [...], "rel2id": {...}, "ent2id": {...}, "rel_info": {...}}`
- `doc_to_text(doc) -> str`:
  - トークン化された文（`doc["sents"]`）を平文テキストに変換する。各文のトークンを結合し、さらに全文を結合する
- `char_count(doc) -> int`:
  - 文書の総文字数を計算する（全トークンの文字数合計）
- `select_dev_docs(dev_data, n=10) -> list[dict]`:
  - devセットから文字数順にソートし、量子位置で `n` 文書を選択する層化サンプリング
- `select_few_shot(train_data) -> dict`:
  - 訓練データからfew-shot例に適した文書を選択する（150-250文字、5-12エンティティ、3-15ラベル）
- `format_few_shot_output(doc) -> dict`:
  - JacRED文書のvertexSetとlabelsから、EXTRACTION_SCHEMAに準拠したJSON形式の出力例を生成する
- `build_constraint_table(train_data) -> dict[str, set[tuple[str, str]]]`:
  - 訓練データ全体から、各関係Pコードに対する観測済み `(head_type, tail_type)` ペアの集合を構築する

### 9.3 `llm_client.py` -- Gemini API呼び出し

**目的**: Google Gemini APIの呼び出し、Structured Outputs対応、リトライロジック。

**主要関数・定数:**
- `MODEL = "gemini-3-flash-preview"`: 使用するモデルID（変更時はここを編集）
- `load_api_key(env_path) -> str`: `.env` ファイルから `GEMINI_API_KEY` を読み込む
- `create_client(api_key) -> genai.Client`: Geminiクライアントを生成する
- `call_gemini(client, system_prompt, user_prompt, response_schema, temperature=0.2, max_retries=3) -> dict`:
  - Gemini APIを呼び出し、Structured OutputsでJSON応答を取得してパース済み辞書として返す
  - `GenerateContentConfig` に `response_mime_type="application/json"` と `response_schema` を設定
  - `ThinkingConfig(thinking_budget=2048)` がハードコードされている（変更時はここを編集）
  - 失敗時は指数バックオフ（2^(attempt+1) 秒）でリトライ

### 9.4 `prompts.py` -- プロンプトテンプレート

**目的**: 全LLM呼び出し用のプロンプト構築ロジック。

**主要定数:**
- `RELATION_JAPANESE`: 35種類の関係PコードからJapanese descriptionへのマッピング辞書
- `ENTITY_TYPES_JAPANESE`: 8種類のエンティティタイプからJapanese descriptionへのマッピング辞書
- `RELATION_GROUPS`: 5つの意味グループへの関係タイプ分割定義
- `GROUP_FOCUS_INSTRUCTIONS`: 各グループに対する焦点指示テキスト

**主要関数:**
- `build_system_prompt(rel_info) -> str`: エンティティタイプ・関係タイプ（全35種類）を含むシステムプロンプトを構築する。Baselineで使用。`rel_info` は `{Pコード: 英語名}` の辞書（JacREDメタデータ由来）
- `build_extraction_prompt(doc_text, few_shot_text, few_shot_output, mode="baseline") -> str`: 抽出用ユーザプロンプトを構築する。`mode="recall"` の場合はRecall重視の指示を追加する
- `build_verification_prompt(doc_text, candidates, entity_map, rel_info) -> str`: Stage 2検証用プロンプトを構築する。各候補トリプルのhead名・tail名・Pコード・英語名・日本語定義・evidence を含む
- `build_group_system_prompt(group_name, group_pcodes, rel_info) -> str`: グループ別システムプロンプトを構築する。対象グループの関係タイプのみを含み、焦点指示を追加する。Relation-Splitで使用
- `build_group_extraction_prompt(doc_text, few_shot_text, few_shot_output, group_pcodes) -> str`: グループ別抽出プロンプトを構築する。few-shot出力を対象グループの関係タイプでフィルタする

### 9.5 `extraction.py` -- 抽出ロジック

**目的**: Baseline・Relation-Split条件の抽出パイプライン全体を実装する。

**主要クラス:**
- `Triple`: データクラス。抽出されたトリプルを表現する
  - フィールド: `head`（エンティティID）, `head_name`, `head_type`, `relation`（Pコード）, `tail`, `tail_name`, `tail_type`, `evidence`

**主要関数:**
- `run_baseline(doc, few_shot, client, schema_info) -> (entities, triples)`:
  - Baseline条件を1文書に対して実行する。システムプロンプト構築 → ユーザプロンプト構築（mode="baseline"） → LLM呼び出し → パース → フィルタ
- `run_relation_split(doc, few_shot, client, schema_info, constraint_table) -> (entities, triples, stats)`:
  - Relation-Split条件を1文書に対して実行する。5グループそれぞれに対してグループ別プロンプトでLLMを呼び出し、エンティティを統合し、domain/range制約を適用する
  - `stats` にはグループ別抽出数とパイプライン各段階の候補数を記録: `{"per_group": {...}, "total_union": N, "after_constraints": K}`
- `_merge_entities_across_passes(all_pass_entities, all_pass_triples) -> (entities, triples)`:
  - 複数パスの結果を統合する。エンティティ名の正規化名（NFKC + 小文字 + strip）に基づいて同一エンティティを識別し、IDを統一する。`(head, relation, tail)` が同一のトリプルを重複除去する
- `_parse_extraction_result(result) -> (entities, triples)`:
  - LLM出力のJSON辞書をパースし、entitiesリストとTriplesリストに変換する
- `filter_invalid_labels(triples, valid_relations) -> list[Triple]`:
  - 不正なPコードを持つトリプルを除去する
- `filter_invalid_entity_types(triples, valid_types) -> list[Triple]`:
  - 不正なエンティティタイプを持つトリプルを除去する
- `apply_domain_range_constraints(triples, constraint_table) -> list[Triple]`:
  - 訓練データで未観測の `(head_type, tail_type)` ペアを持つトリプルを除去する
- `_verify_candidates(doc, candidates, entity_id_to_name, client, schema_info, batch_size=10) -> list[Triple]`:
  - Stage 2のバッチ検証を実行する。候補をbatch_size件ずつに分割し、各バッチに対して検証プロンプトを送信する。本リポではRelation-Splitの主手法に含まれないが、Proposed（Two-Stage）条件として `run_proposed()` から呼び出される

### 9.6 `evaluation.py` -- 評価ロジック

**目的**: エンティティアライメントとP/R/F1の算出。

**主要関数:**
- `align_entities(predicted_entities, gold_vertex_set) -> dict[str, int]`:
  - 予測エンティティをGold vertexSetにアライメントする（3パスマッチング: 完全一致 → 正規化一致 → 部分文字列一致）
  - 出力: `{予測エンティティID: Gold vertexSetインデックス}`
- `evaluate_relations(predicted_triples, gold_labels, entity_alignment) -> dict`:
  - アライメント結果を用いて予測トリプルをGoldラベルと照合し、TP/FP/FN/P/R/F1 を算出する
  - FP詳細（理由: `entity_not_aligned` or `wrong_relation`）とFN詳細を含む
- `aggregate_results(per_doc) -> dict`:
  - 文書別結果リストからマイクロ平均のP/R/F1を算出する

### 9.7 `schemas.py` -- JSON Schema定義

**目的**: Gemini Structured Outputs用のJSON Schema定義。

**定数:**
- `EXTRACTION_SCHEMA`: 抽出用スキーマ。`entities`（id, name, typeの配列）と `relations`（head, relation, tail, evidenceの配列）を要求する
- `VERIFICATION_SCHEMA`: 検証用スキーマ。`decisions`（candidate_index, keepの配列）を要求する

### 9.8 `results.json` -- 最新の実験結果

**目的**: 最後に実行された実験の全結果をJSON形式で保存する。

**構造:**
```json
{
  "experiment": {
    "model": "gemini-3-flash-preview",
    "num_docs": 10,
    "few_shot_doc": "スタッド (バンド)",
    "timestamp": "2026-02-04T13:41:23.882966"
  },
  "conditions": {
    "baseline": {
      "per_doc": [{"title": "...", "precision": 0.64, ...}, ...],
      "aggregate": {"precision": 0.30, "recall": 0.21, "f1": 0.25, ...}
    },
    "relation_split": {
      "per_doc": [{"title": "...", "precision": 0.14, ..., "stats": {...}}, ...],
      "aggregate": {"precision": 0.21, "recall": 0.16, "f1": 0.18, ...}
    }
  }
}
```

---

## 10. 参考文献

1. Ma, Y., Tanaka, J., & Araki, M. **"Building a Japanese Document-Level Relation Extraction Dataset Assisted by Cross-Lingual Transfer."** *Proceedings of LREC-COLING 2024.*
   - JacREDデータセットの構築論文。本実験のデータセット。

2. Yao, Y., Ye, D., Li, P., Han, X., Lin, Y., Liu, Z., Liu, Z., Huang, L., Zhou, J., & Sun, M. **"DocRED: A Large-Scale Document-Level Relation Extraction Dataset."** *Proceedings of ACL 2019.*
   - JacREDの基となった英語DocREデータセット。

3. Tan, C., Zhao, W., Wei, Z., & Huang, X. **"Document-level Relation Extraction: A Survey."** *arXiv preprint, 2023.*
   - 文書レベル関係抽出のサーベイ論文。

4. Li, D., Liu, Y., & Sun, M. **"A Survey on LLM-based Generative Information Extraction."** *arXiv preprint, 2024.*
   - LLMによる情報抽出のサーベイ論文。

5. Giorgi, J., Bader, G., & Wang, B. **"End-to-end Named Entity Recognition and Relation Extraction using Pre-trained Language Models."** *arXiv preprint, 2019.*
   - 事前学習言語モデルを用いたend-to-end NER+RE。

6. Dagdelen, J., Dunn, A., Lee, S., Walker, N., Rosen, A., Ceder, G., Persson, K., & Jain, A. **"Structured information extraction from scientific text with large language models."** *Nature Communications, 2024.*
   - LLMによる科学文献からの構造化情報抽出。

7. Willard, B., & Louf, R. **"Generating Structured Outputs from Language Models."** *arXiv preprint, 2025.*
   - 言語モデルからの構造化出力生成手法。

8. Harnoune, A., Rhanoui, M., Asri, B., Zellou, A., & Yousfi, S. **"Information extraction pipelines for knowledge graphs."** *Knowledge and Information Systems (Springer), 2022.*
   - 知識グラフ構築のための情報抽出パイプライン。

9. Mintz, M., Bills, S., Snow, R., & Jurafsky, D. **"Distant supervision for relation extraction without labeled data."** *Proceedings of ACL 2009.*
   - ラベルなしデータからの遠隔教師あり関係抽出。

10. Lu, Y., Liu, Z., & Huang, L. **"Cross-Lingual Structure Transfer for Relation and Event Extraction."** *Proceedings of ACL 2019.*
    - 言語間構造転移による関係・イベント抽出。
