# テストルール・ガイドライン

本ドキュメントは、`tests/` 配下のテストコードを作成・レビューする際に従うべきルールと規約を定義する。

---

## 1. ディレクトリ構成

テストディレクトリは **`app/` のレイヤー構造をミラーリング** する。

```
app/                              tests/
├── domain/                       ├── domain/
│   ├── llm_client/               │   ├── llm_client/
│   │   └── llm_client_base.py    │   │   └── test_llm_client_base.py
│   └── question_answer/          │   └── question_answer/
│       └── build_prompt.py       │       └── test_prompt.py
├── infra/                        ├── infra/
│   └── llm_client/               │   └── llm_client/
│       └── gemini.py             │       └── test_gemini.py
└── usecase/                      └── usecase/
    └── question_answer/              └── question_answer/
        ├── dto.py                        ├── test_dto.py
        └── usecase.py                    └── test_usecase.py
```

### ルール

- テストファイルは対応するソースファイルと **同じレイヤー・同じサブディレクトリ** に配置する
- テストファイル名は `test_<対応するモジュール名>.py` とする（例: `llm_client_base.py` → `test_llm_client_base.py`）

---

## 2. ファイル構成テンプレート

各テストファイルは以下の構造に従う。

```python
"""<モジュール概要>のテスト."""
# ruff: noqa: S101

import pytest
from pydantic import ValidationError  # Pydantic DTOのテスト時のみ

from app.<レイヤー>.<モジュールパス> import <テスト対象クラス>

# ================================
# <テスト対象クラス名>
# ================================


class Test<テスト対象クラス名>:
    """<テスト対象クラス名>のテスト"""

    def test_<振る舞いの説明>(self) -> None:
        """<日本語でテスト内容を説明するdocstring>"""
        ...
```

### ルール

| 項目 | ルール |
|---|---|
| モジュールdocstring | 日本語で `"""<内容>のテスト."""` と記述する。末尾にピリオドを付ける |
| ruff抑制 | ファイル先頭に `# ruff: noqa: S101` を記述し、`assert` の警告を抑制する |
| import順序 | 標準ライブラリ → サードパーティ（`pytest`, `pydantic`等） → プロジェクト内（`app.*`） |
| セクション区切り | テスト対象クラスごとに `# ================================` コメントで区切る |
| テストクラス名 | `Test` + テスト対象クラス名（例: `TestPrompt`, `TestLLMResult`） |
| テストメソッド名 | `test_` で始め、英語のスネークケースで振る舞いを記述する |
| テストdocstring | 日本語で期待する振る舞いを簡潔に記述する |
| 型アノテーション | テストメソッドの戻り値は `-> None` を必ず付ける |

---

## 3. テストパターン

### 3.1 Pydantic DTO のテスト

DTO（`BaseModel` 継承クラス）に対しては、以下の観点でテストを書く。

```python
class TestLLMResult:
    """LLMResult DTOのテスト"""

    def test_create_with_required_fields_only(self) -> None:
        """必須フィールドのみで生成できる"""
        result = LLMResult(output="回答です")
        assert result.output == "回答です"
        assert result.thinking is None  # Optional のデフォルト値

    def test_create_with_all_fields(self) -> None:
        """全フィールドを指定して生成できる"""
        result = LLMResult(output="回答です", thinking="思考過程です")
        assert result.output == "回答です"
        assert result.thinking == "思考過程です"

    def test_required_field_validation(self) -> None:
        """必須フィールドを省略するとValidationErrorになる"""
        with pytest.raises(ValidationError, match="output"):
            LLMResult()
```

#### 必須チェック項目

- [ ] 必須フィールドのみでインスタンス生成できること
- [ ] 全フィールド指定でインスタンス生成できること
- [ ] Optionalフィールドのデフォルト値が正しいこと
- [ ] 必須フィールドを省略した場合に `ValidationError` が発生すること
- [ ] `ValidationError` の `match` パラメータで対象フィールド名を検証すること

### 3.2 抽象基底クラス（ABC）のテスト

```python
class TestLLMClientBase:
    """LLMClientBase抽象クラスのテスト"""

    def test_cannot_instantiate_directly(self) -> None:
        """抽象クラスは直接インスタンス化するとTypeError"""
        with pytest.raises(TypeError, match="abstract"):
            LLMClientBase()  # type: ignore[abstract]

    def test_concrete_implementation(self) -> None:
        """具象実装では抽象メソッドが呼び出せる"""

        class StubClient(LLMClientBase):
            def predict(self, prompt: Prompt) -> LLMResult:
                return LLMResult(output=f"received: {prompt.text}")

        client = StubClient()
        result = client.predict(Prompt(text="テスト"))
        assert result.output == "received: テスト"
```

#### 必須チェック項目

- [ ] 抽象クラスを直接インスタンス化すると `TypeError` が発生すること
- [ ] `# type: ignore[abstract]` コメントで型チェッカーの警告を抑制すること
- [ ] テスト内でスタブ（Stub）の具象実装クラスを定義し、抽象メソッドの動作を検証すること

### 3.3 ユースケースのテスト

ユースケース層のテストでは、**依存する外部サービス（LLMクライアント等）をスタブに差し替えて** テストする。

```python
class TestQuestionAnswerUsecase:
    """QuestionAnswerUsecaseのテスト"""

    def test_execute_returns_answer(self) -> None:
        """質問を渡すと回答が返る"""

        class StubLLMClient(LLMClientBase):
            def predict(self, prompt: Prompt) -> LLMResult:
                return LLMResult(output="スタブ回答")

        usecase = QuestionAnswerUsecase(llm_client=StubLLMClient())
        response = usecase.execute("テスト質問")
        assert response.answer == "スタブ回答"
```

#### 必須チェック項目

- [ ] 外部依存（API、DB等）は **スタブ/モック** に差し替えること
- [ ] スタブはテストメソッド内またはテストクラス内で定義する
- [ ] ユースケースの入出力（DTO）の値を検証すること

### 3.4 インフラ層のテスト

外部APIに依存するインフラ層のテストでは、以下の方針に従う。

#### 必須チェック項目

- [ ] 外部API呼び出しは **モック/スタブ** に差し替え、実際のAPI呼び出しをしない
- [ ] レスポンスのパース処理が正しく動作することを検証する
- [ ] 異常系（APIエラー、不正レスポンス等）もテストする

---

## 4. テスト分類と方針

| レイヤー | テスト対象 | 外部依存 | 方針 |
|---|---|---|---|
| **domain** | DTO, 抽象クラス, プロンプト生成 | なし | 純粋なロジックテスト。外部依存なし |
| **usecase** | ユースケースクラス | あり（domain層の抽象に依存） | スタブを注入してテスト |
| **infra** | API クライアント等 | あり（外部API） | モックでAPI呼び出しを差し替えてテスト |

---

## 5. アサーション規約

| パターン | 書き方 |
|---|---|
| 値の等価検証 | `assert result.field == "期待値"` |
| 文字列の検証 | `assert result == "期待値"` （`in` による部分一致は使わない） |
| None検証 | `assert result.field is None` |
| 例外検証 | `with pytest.raises(ExceptionType, match="パターン"):` |
| 型検証 | `assert isinstance(result, ExpectedType)` |

### ルール

- 文字列の検証には `in`（部分一致）ではなく `==`（完全一致）を使用する。部分一致はプロンプトの構造変更を検知できないため、テストの信頼性が低下する
- `pytest.raises` には必ず `match` パラメータを指定し、例外メッセージの一部を検証する
- 複数のアサーションが必要な場合は、1つのテストメソッド内にまとめてよい（同一の振る舞いに関する検証であれば）

---

## 6. カバレッジ 100% ルール

**すべてのプロダクションコード（`app/` 配下）は、行カバレッジ 100% を維持する。**

### 基本方針

- 新しいモジュールを追加したら、対応するテストファイルで **全行・全分岐** をカバーするテストを書く
- PR / コミット前に必ずカバレッジを確認し、100% 未満であればテストを追加する
- カバレッジが 100% に満たない場合、CI は失敗として扱う（将来的なCI導入を想定）

### カバレッジ計測コマンド

`pyproject.toml` を変更せず、**コマンドラインオプションのみ** でカバレッジを制御する。

```bash
# 基本: カバレッジ計測 + 未カバー行の表示
pytest --cov=app --cov-report=term-missing

# 100% 未満で失敗させる（推奨: テスト作成後の最終確認）
pytest --cov=app --cov-report=term-missing --cov-fail-under=100

# HTML レポート生成（ブラウザで詳細確認）
pytest --cov=app --cov-report=term-missing --cov-report=html

# 特定レイヤーのみカバレッジ確認
pytest tests/domain/ --cov=app/domain --cov-report=term-missing --cov-fail-under=100
pytest tests/usecase/ --cov=app/usecase --cov-report=term-missing --cov-fail-under=100
pytest tests/infra/ --cov=app/infra --cov-report=term-missing --cov-fail-under=100
```

### カバレッジ対象と除外

| 対象 | カバレッジ必須? | 備考 |
|---|---|---|
| `app/domain/` | **必須** | 純粋ロジック。100% 達成が容易 |
| `app/usecase/` | **必須** | スタブ注入で全パスをカバー |
| `app/infra/` | **必須** | モックで外部API呼び出しを差し替え |
| `app/command/` | **必須** | エントリーポイント。必要に応じてモックを使う |
| `tests/` | 対象外 | テストコード自体はカバレッジ計測対象にしない |

### カバレッジが 100% にならない場合の対処法

#### 1. 未カバー行を特定する

```bash
pytest --cov=app --cov-report=term-missing
```

出力例:

```
Name                                      Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------
app/domain/llm_client/llm_client_base.py     10      0   100%
app/infra/llm_client/gemini.py               25      3    88%   44-46
-----------------------------------------------------------------------
TOTAL                                        35      3    91%
```

`Missing` 列に表示される行番号が未カバー行。該当行を通すテストケースを追加する。

#### 2. 分岐カバレッジも確認する

```bash
pytest --cov=app --cov-report=term-missing --cov-branch
```

`--cov-branch` を付けると、`if/else` や条件式の全分岐が通っているかまで検証できる。

#### 3. テスト困難なコードの対処

テスト困難なコードに対しては、以下の順に対処する。

| 優先度 | 対処法 | 説明 |
|---|---|---|
| 1 | **テストを書く** | まずはテストで網羅する方法を検討する |
| 2 | **設計を見直す** | テストしにくいコードは依存注入や責務分離で改善する |
| 3 | **`# pragma: no cover`** | 最終手段。使用時は理由をコメントに明記する |

`# pragma: no cover` を使う場合は **必ず理由を記述** する。

```python
if __name__ == "__main__":  # pragma: no cover  # CLIエントリーポイント: 統合テストで担保
    main()
```

### レイヤー別カバレッジ達成ガイド

#### domain 層

純粋なデータクラスとロジックのため、最も容易に 100% 達成可能。

- DTO: 全フィールドパターン + バリデーションエラー
- 抽象クラス: インスタンス化不可 + スタブ具象実装
- プロンプト生成: 入力値ごとの出力検証

#### usecase 層

依存をスタブに差し替えることで全パスをカバーする。

- 正常系: 期待する入出力の検証
- 異常系: 依存先が例外を投げた場合の伝播確認

#### infra 層

外部API呼び出しをモックに差し替え、レスポンスのパース処理を網羅する。

- 正常レスポンス: パース結果の検証
- 異常レスポンス: APIエラー、不正データ、空レスポンス等
- 条件分岐: フラグやオプションによる処理分岐を全パターンカバー

---

## 7. テスト実行

```bash
# 全テスト実行
pytest

# カバレッジ付き実行（100% 必須）
pytest --cov=app --cov-report=term-missing --cov-fail-under=100

# HTML レポート生成
pytest --cov=app --cov-report=term-missing --cov-report=html

# 特定ディレクトリのテスト実行
pytest tests/domain/

# 特定ファイルのテスト実行
pytest tests/domain/llm_client/test_llm_client_base.py

# 特定テストクラスの実行
pytest tests/domain/llm_client/test_llm_client_base.py::TestPrompt

# 特定テストメソッドの実行
pytest tests/domain/llm_client/test_llm_client_base.py::TestPrompt::test_create_with_text
```

---

## 8. チェックリスト（テスト作成時）

新しいテストファイルを作成する際は、以下を確認する。

### コード規約

- [ ] ファイルが `app/` と同じレイヤー・ディレクトリに配置されているか
- [ ] ファイル名が `test_<モジュール名>.py` になっているか
- [ ] モジュールdocstringが日本語で記述されているか
- [ ] `# ruff: noqa: S101` が記述されているか
- [ ] テストクラスに日本語のdocstringがあるか
- [ ] 各テストメソッドに日本語のdocstringがあるか
- [ ] テストメソッドの戻り値型 `-> None` が付いているか
- [ ] 外部依存がスタブ/モックに差し替えられているか
- [ ] `pytest.raises` に `match` パラメータが指定されているか

### カバレッジ

- [ ] `pytest --cov=app --cov-report=term-missing --cov-fail-under=100` が通るか
- [ ] 新規追加モジュールの全行がテストでカバーされているか
- [ ] 条件分岐（`if/else`, 三項演算子等）の全パスがカバーされているか
- [ ] `# pragma: no cover` を使用している場合、理由がコメントに明記されているか
