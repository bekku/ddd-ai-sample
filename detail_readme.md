# DDD AI Sample 詳細設計ガイド

本ドキュメントでは、このリポジトリの実際のコードを使って、各レイヤーがどのように実装されているかを解説する。
初めてこのプロジェクトに参加する開発者が「どこに何を書くのか」を迷わないことを目的としている。

---

## 目次

1. [全体像：レイヤーと依存ルール](#1-全体像レイヤーと依存ルール)
2. [domain 層 — ビジネスロジックの中核](#2-domain-層--ビジネスロジックの中核)
3. [usecase 層 — アプリケーションロジック](#3-usecase-層--アプリケーションロジック)
4. [infra 層 — 外部サービスの具象実装](#4-infra-層--外部サービスの具象実装)
5. [command 層 — エントリーポイントと DI](#5-command-層--エントリーポイントと-di)
6. [テスト — レイヤーごとのテスト戦略](#6-テスト--レイヤーごとのテスト戦略)
7. [DTO の配置ルールまとめ](#7-dto-の配置ルールまとめ)
8. [新機能を追加するときの手順](#8-新機能を追加するときの手順)

---

## 1. 全体像：レイヤーと依存ルール

### レイヤー構成

```
app/
├── command/     … エントリーポイント層（CLI / Lambda）
├── domain/      … ドメイン層（ビジネスロジック・抽象定義）
├── infra/       … インフラ層（外部 API の具象実装）
└── usecase/     … ユースケース層（アプリケーションロジック）
```

### 依存ルール（最重要）

**「誰が誰を import できるか」** を厳格に管理する。これが DDD の根幹。

```
command  ──→  usecase  ──→  domain  ←──  infra
   │                                        ↑
   └────────────────────────────────────────┘
```

- `usecase → domain` : usecase は domain 層の抽象・DTO・ドメインロジックを使う
- `infra → domain` : infra は domain 層の抽象インターフェースを **実装する**
- `command → usecase` : command は usecase を呼び出す
- `command → infra` : command は **具象クラスを生成して usecase に注入する（DI）**

| レイヤー | import できる対象 | import してはいけない対象 |
|---|---|---|
| **domain** | 標準ライブラリ, pydantic のみ | usecase, infra, command |
| **usecase** | domain | infra, command |
| **infra** | domain | usecase, command |
| **command** | usecase, infra（※） | — |

> **※ command と domain の関係:**
> command から domain を直接 import することはルール上許可されているが、
> 基本的には **usecase 経由で domain にアクセスする** のが望ましい。
> command が直接 import するのは **usecase と infra** だけに留めるのが原則。

> **なぜ command だけが infra を import できるのか？**
> → command 層で「どの具象実装を使うか」を決めて usecase に注入する（依存性の逆転 = DIP）ため。

---

## 2. domain 層 — ビジネスロジックの中核

domain 層には以下の 3 種類を配置する。

| 種類 | 説明 | 例 |
|---|---|---|
| **DTO** | レイヤー間のデータ受け渡し用 | `Prompt`, `LLMResult` |
| **抽象インターフェース** | 外部サービスの契約定義 | `LLMClientBase` |
| **ドメインロジック** | ビジネスルール・プロンプト生成等 | `QuestionAnswerPrompt` |

### 2.1 DTO（データ転送オブジェクト）

DTO は **Pydantic の `BaseModel`** を継承して定義する。バリデーション・イミュータブル性・シリアライズを自動で得られる。

**実際のコード: `app/domain/llm_client/llm_client_base.py`**

```python
from pydantic import BaseModel

class Prompt(BaseModel):
    """LLMに送信するプロンプトを表すイミュータブルなデータクラス"""

    text: str


class LLMResult(BaseModel):
    """LLMの応答結果を表すモデル"""

    output: str
    thinking: str | None = None  # Optional フィールドはデフォルト値を設定
```

**ポイント:**

- `Prompt` は LLM への入力を表す DTO。`text` が必須フィールド
- `LLMResult` は LLM からの出力を表す DTO。`thinking` は Optional（デフォルト `None`）
- この 2 つは domain 層に置く。なぜなら **「LLM にプロンプトを渡して結果を受け取る」というのはビジネス上の概念** だから
- domain 層の DTO は `pydantic` にしか依存しない。Gemini や OpenAI など **特定の外部サービスに一切依存しない**

### 2.2 抽象インターフェース（ABC）

外部サービスへの依存を **「抽象」** として定義する。具体的な API 呼び出しは書かない。

**実際のコード: `app/domain/llm_client/llm_client_base.py`**

```python
from abc import ABC, abstractmethod

class LLMClientBase(ABC):
    """LLMクライアントの抽象基底クラス"""

    @abstractmethod
    def predict(self, prompt: Prompt) -> LLMResult:
        """プロンプトを実行し、結果を返す"""
        ...
```

**ポイント:**

- `ABC` を継承し、`@abstractmethod` で契約を定義する
- 引数・戻り値は **domain 層の DTO（`Prompt`, `LLMResult`）** を使う
- `GeminiClient` や `OpenAIClient` など具体的な API 名は一切出てこない
- このインターフェースを infra 層で実装する（後述）

### 2.3 ドメインロジック（プロンプト生成）

ビジネスロジックとしてのプロンプト構築を担う。

**実際のコード: `app/domain/question_answer/build_prompt.py`**

```python
class QuestionAnswerPrompt:
    """質問に対する回答を生成するプロンプト"""

    def __init__(self, question: str) -> None:
        """初期化"""
        self.question = question

    def get_prompt(self) -> str:
        """プロンプトを取得"""
        return f"""あなたは,質問に対する回答を生成するためのAIアシスタントです。
        質問: {self.question}
        に対する回答を生成してください。
        """
```

**ポイント:**

- プロンプトのテンプレートは **ドメインロジック** として扱う
- 外部サービスに依存しない。純粋な文字列生成
- ユースケース層から呼び出される

### 2.4 `__init__.py` による公開制御

domain 層のモジュールは `__init__.py` で公開するクラスを制御する。

**実際のコード: `app/domain/llm_client/__init__.py`**

```python
from app.domain.llm_client.llm_client_base import LLMClientBase, LLMResult, Prompt

__all__ = ["LLMClientBase", "LLMResult", "Prompt"]
```

**ポイント:**

- 他のレイヤーから `from app.domain.llm_client import LLMClientBase, Prompt, LLMResult` と簡潔に import できるようになる
- `__all__` で公開範囲を明示する

---

## 3. usecase 層 — アプリケーションロジック

usecase 層は **domain 層のオブジェクトを組み合わせて、アプリケーション固有の処理フローを実行する**。

### 3.1 ユースケースの DTO

ユースケース層にも DTO を配置する。これは **ユースケースの出力（レスポンス）** を表す。

**実際のコード: `app/usecase/question_answer/dto.py`**

```python
from pydantic import BaseModel


class QuestionAnswerResponse(BaseModel):
    """質問に対する回答を生成するユースケースのDTO"""

    answer: str
```

**ポイント:**

- ユースケースの **出力の形** を定義する DTO
- domain 層の `LLMResult` とは別物。`LLMResult` は「LLM の生の出力」、`QuestionAnswerResponse` は「ユースケースとしての回答」
- この分離により、domain 層の内部構造が変わっても usecase の出力は影響を受けない

### 3.2 ユースケース本体

**実際のコード: `app/usecase/question_answer/usecase.py`**

```python
from app.domain.llm_client.llm_client_base import LLMClientBase, LLMResult, Prompt
from app.domain.question_answer.build_prompt import QuestionAnswerPrompt
from app.usecase.question_answer.dto import QuestionAnswerResponse


class QuestionAnswerUsecase:
    """質問に対する回答を生成するユースケース"""

    def __init__(self, llm_client: LLMClientBase) -> None:
        """初期化"""
        self.llm_client = llm_client

    def execute(self, question: str) -> QuestionAnswerResponse:
        """質問に対する回答を生成する"""
        # 1. プロンプトを組み立てる（domain 層のロジック）
        input_prompt: str = QuestionAnswerPrompt(question).get_prompt()

        # 2. LLM に問い合わせる（domain 層の抽象インターフェース経由）
        result: LLMResult = self.llm_client.predict(Prompt(text=input_prompt))

        # 3. ユースケースの DTO に変換して返す
        return QuestionAnswerResponse(answer=result.output)
```

**ポイント:**

- **import を確認 → `app.domain.*` のみ参照している**。`app.infra` は一切 import していない
- コンストラクタで `LLMClientBase`（抽象）を受け取る。**具体的に Gemini なのか OpenAI なのかは知らない**
- `execute()` メソッドが処理フロー:
  1. domain 層の `QuestionAnswerPrompt` でプロンプトを組み立てる
  2. domain 層の `LLMClientBase.predict()` で LLM に問い合わせる
  3. usecase 層の `QuestionAnswerResponse` DTO に変換して返す

> **初心者向け重要ポイント:**
> usecase 層は「何を使うか（抽象）」だけ知っていて、「どう実装されているか（具象）」は知らない。
> これが **依存性の逆転（DIP）** の核心。

---

## 4. infra 層 — 外部サービスの具象実装

infra 層は **domain 層で定義した抽象インターフェースを、具体的な外部サービスで実装する**。

**実際のコード: `app/infra/llm_client/gemini.py`**

```python
from google import genai

from app.domain.llm_client.llm_client_base import LLMClientBase, LLMResult, Prompt


class GeminiClient(LLMClientBase):       # ← domain 層の抽象を実装
    """Gemini APIを使用したLLMクライアントクラス"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        *,
        include_thoughts: bool = True,
    ) -> None:
        """初期化"""
        self.client = genai.Client(api_key=api_key)      # ← 外部 SDK はここだけ
        self.model = model
        self.include_thoughts = include_thoughts

    def predict(
        self,
        prompt: Prompt,           # ← domain 層の DTO を入力に受け取る
    ) -> LLMResult:               # ← domain 層の DTO を出力として返す
        """Gemini APIを使用してプロンプトを実行する"""
        contents: list[str] = [prompt.text]

        # Gemini API 固有の呼び出し
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(
                    include_thoughts=self.include_thoughts,
                ),
            ),
        )

        # Gemini 固有のレスポンスを domain 層の DTO に変換
        thinking_logs: list[str] = []
        outputs: list[str] = []

        parts = response.candidates[0].content.parts
        for part in parts:
            text = getattr(part, "text", None)
            if not text:
                continue

            if getattr(part, "thought", False):
                thinking_logs.append(text)
            else:
                outputs.append(text)

        thinking = "\n".join(thinking_logs).strip() if thinking_logs else None
        output = "\n".join(outputs).strip()

        return LLMResult(thinking=thinking, output=output)
```

**ポイント:**

- **import を確認 → `app.domain.*` のみ参照している**。`app.usecase` や `app.command` は import しない
- `LLMClientBase` を継承して `predict()` を実装する
- 引数は `Prompt`（domain DTO）、戻り値は `LLMResult`（domain DTO）
- Gemini API 固有のコード（`genai.Client`, `generate_content` 等）は **この層にのみ閉じ込める**
- 将来 OpenAI に切り替えたい場合は `OpenAIClient(LLMClientBase)` を新たに作るだけ。usecase 層は一切変更不要

> **初心者向け重要ポイント:**
> infra 層は「domain 層の抽象を実装する」のが仕事。
> 外部 SDK（`google-genai` 等）への依存はこの層に閉じ込める。

---

## 5. command 層 — エントリーポイントと DI

command 層は **アプリケーション全体を組み立てる唯一の場所**。ここで **依存性の注入（DI）** を行う。

### 5.1 ステータス Enum

**実際のコード: `app/command/status.py`**

```python
from enum import Enum


class Status(str, Enum):
    """APIレスポンスステータス"""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
```

### 5.2 メインコマンド（CLI / Lambda）

**実際のコード: `app/command/question.py`**

```python
import argparse
import os

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from app.command.status import Status
from app.infra.llm_client.gemini import GeminiClient                # ← infra を import
from app.usecase.question_answer.dto import QuestionAnswerResponse   # ← usecase を import
from app.usecase.question_answer.usecase import QuestionAnswerUsecase

load_dotenv()


# ── command 層専用の DTO ──

class Request(BaseModel):
    """質問に対する回答を生成するリクエスト"""
    question: str


class Response(BaseModel):
    """APIレスポンスラッパー"""
    status: Status
    data: QuestionAnswerResponse    # ← usecase 層の DTO を内包


# ── 環境変数バリデーション ──

def validate_env() -> None:
    """環境変数を検証"""
    if os.getenv("GEMINI_API_KEY") is None:
        msg = "GEMINI_API_KEY is required"
        raise ValueError(msg)


# ── ★ 依存性の注入（DI）はここで行う ── ★

def process(request: Request) -> Response:
    """ユースケースを実行する"""
    validate_env()

    # ★ ここが DI のポイント ★
    # command 層で「GeminiClient という具象」を生成し、usecase に注入する
    usecase = QuestionAnswerUsecase(
        llm_client=GeminiClient(                # ← 具象実装を選択
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.5-flash-lite",
        ),
    )

    try:
        response = usecase.execute(request.question)
    except Exception as e:
        error_message = f"Exception: {e}"
        raise ValueError(error_message) from e

    return Response(status=Status.SUCCESS, data=response)


# ── AWS Lambda エントリーポイント ──

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        req = Request.model_validate(event)
    except ValidationError as e:
        error_message = f"ValidationError: {e}"
        raise ValueError(error_message) from e

    resp = process(req)
    return resp.model_dump_json()


# ── CLI エントリーポイント ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="質問に対する回答を生成する")
    parser.add_argument("--question", required=True, help="質問内容")
    args = parser.parse_args()

    try:
        request = Request(question=args.question)
        response = process(request)
        print(response.model_dump_json())
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
```

**ポイント:**

- **import を確認 → `app.infra` と `app.usecase` の両方を import している**。これは command 層だけに許される
- `process()` 関数の中で `GeminiClient`（具象）を生成し、`QuestionAnswerUsecase` のコンストラクタに注入している
- command 層専用の DTO（`Request`, `Response`）を定義。これは API のリクエスト/レスポンスの形で、usecase 層の DTO とは分離されている
- Lambda と CLI の 2 つのエントリーポイントを持つ

> **初心者向け重要ポイント:**
> 「具象クラスをどこで選ぶか」が DDD の重要な設計判断。
> **答えは常に command 層**。usecase 層は抽象しか知らない。

---

## 6. テスト — レイヤーごとのテスト戦略

テストは `tests/` 配下に `app/` と同じディレクトリ構造でミラーリングする。

```
app/domain/llm_client/llm_client_base.py    →  tests/domain/llm_client/test_llm_client_base.py
app/domain/question_answer/build_prompt.py  →  tests/domain/question_answer/test_prompt.py
app/usecase/question_answer/dto.py          →  tests/usecase/question_answer/test_dto.py
app/usecase/question_answer/usecase.py      →  tests/usecase/question_answer/test_usecase.py
app/infra/llm_client/gemini.py              →  tests/infra/llm_client/test_gemini.py
```

### 6.1 domain 層のテスト — 外部依存なし・純粋ロジック

domain 層は外部に依存しないため、**モックもスタブも不要**。そのままテストできる。

#### DTO のテスト

**実際のコード: `tests/domain/llm_client/test_llm_client_base.py`（抜粋）**

```python
"""LLMクライアント基底クラスおよびDTOのテスト."""
# ruff: noqa: S101

import pytest
from pydantic import ValidationError

from app.domain.llm_client.llm_client_base import LLMClientBase, LLMResult, Prompt


class TestPrompt:
    """Prompt DTOのテスト"""

    def test_create_with_text(self) -> None:
        """textを渡すとPromptが生成される"""
        prompt = Prompt(text="こんにちは")
        assert prompt.text == "こんにちは"

    def test_text_required(self) -> None:
        """textを省略するとValidationErrorになる"""
        with pytest.raises(ValidationError, match="text"):
            Prompt()   # ← 必須フィールドを省略 → エラー


class TestLLMResult:
    """LLMResult DTOのテスト"""

    def test_create_with_output_only(self) -> None:
        """outputのみ指定した場合、thinkingはNone"""
        result = LLMResult(output="回答です")
        assert result.output == "回答です"
        assert result.thinking is None    # ← Optional のデフォルト値を確認

    def test_create_with_output_and_thinking(self) -> None:
        """outputとthinkingの両方を指定できる"""
        result = LLMResult(output="回答です", thinking="思考過程です")
        assert result.output == "回答です"
        assert result.thinking == "思考過程です"
```

**テスト観点:**

- 必須フィールドで生成できること
- Optional フィールドのデフォルト値が正しいこと
- 必須フィールド省略時に `ValidationError` が出ること（`match` で対象フィールド名を検証）

#### 抽象クラスのテスト

```python
class TestLLMClientBase:
    """LLMClientBase抽象クラスのテスト"""

    def test_cannot_instantiate_directly(self) -> None:
        """抽象クラスは直接インスタンス化するとTypeError"""
        with pytest.raises(TypeError, match="abstract"):
            LLMClientBase()  # type: ignore[abstract]

    def test_concrete_implementation_predict(self) -> None:
        """具象実装ではpredictが呼び出せる"""

        # テスト内でスタブの具象クラスを定義
        class StubLLMClient(LLMClientBase):
            def predict(self, prompt: Prompt) -> LLMResult:
                return LLMResult(
                    output=f"received: {prompt.text}",
                    thinking="stub thinking",
                )

        client = StubLLMClient()
        prompt = Prompt(text="テスト")
        result = client.predict(prompt)

        assert result.output == "received: テスト"
        assert result.thinking == "stub thinking"
```

**テスト観点:**

- 抽象クラスを直接インスタンス化すると `TypeError` になること
- テスト内で Stub 具象クラスを定義して、抽象メソッドが正しく動作すること

#### ドメインロジックのテスト

**実際のコード: `tests/domain/question_answer/test_prompt.py`**

```python
"""プロンプト定義のテスト."""
# ruff: noqa: S101

from app.domain.question_answer.build_prompt import QuestionAnswerPrompt


class TestQuestionAnswerPrompt:
    """QuestionAnswerPromptのテスト"""

    def test_init_sets_question(self) -> None:
        """初期化時にquestionが正しく設定される"""
        prompt = QuestionAnswerPrompt(question="テスト質問")
        assert prompt.question == "テスト質問"

    def test_get_prompt_returns_expected_string(self) -> None:
        """get_promptが期待通りのプロンプト文字列を返す"""
        prompt = QuestionAnswerPrompt(question="テスト質問")
        expected = (
            "あなたは,質問に対する回答を生成するためのAIアシスタントです。\n"
            "        質問: テスト質問\n"
            "        に対する回答を生成してください。\n"
            "        "
        )
        assert prompt.get_prompt() == expected   # ← 完全一致で検証（部分一致は NG）
```

**テスト観点:**

- 文字列の検証は **完全一致（`==`）** を使い、`in`（部分一致）は使わない
- プロンプトの構造変更を確実に検知するため

### 6.2 usecase 層のテスト — Stub で依存を差し替え

usecase 層のテストでは、**LLM クライアントをスタブに差し替える**。実際の API は呼ばない。

**実際のコード: `tests/usecase/question_answer/test_usecase.py`**

```python
"""質問に対する回答を生成するユースケースのテスト."""
# ruff: noqa: S101, ARG002, RUF001

from app.domain.llm_client.llm_client_base import LLMClientBase, LLMResult, Prompt
from app.domain.question_answer.build_prompt import QuestionAnswerPrompt
from app.usecase.question_answer.dto import QuestionAnswerResponse
from app.usecase.question_answer.usecase import QuestionAnswerUsecase


class TestQuestionAnswerUsecase:
    """QuestionAnswerUsecaseのテスト"""

    def test_execute_returns_answer(self) -> None:
        """質問を渡すと回答が返る"""

        # ★ テスト内で Stub を定義して注入する ★
        class StubLLMClient(LLMClientBase):
            def predict(self, prompt: Prompt) -> LLMResult:
                return LLMResult(output="スタブ回答")

        usecase = QuestionAnswerUsecase(llm_client=StubLLMClient())
        response = usecase.execute("テスト質問")
        assert response.answer == "スタブ回答"

    def test_execute_returns_question_answer_response(self) -> None:
        """executeの戻り値がQuestionAnswerResponseである"""

        class StubLLMClient(LLMClientBase):
            def predict(self, prompt: Prompt) -> LLMResult:
                return LLMResult(output="回答")

        usecase = QuestionAnswerUsecase(llm_client=StubLLMClient())
        response = usecase.execute("質問")
        assert isinstance(response, QuestionAnswerResponse)

    def test_execute_passes_prompt_to_llm_client(self) -> None:
        """LLMクライアントにプロンプトが正しく渡される"""

        # ★ Spy パターン: 引数をキャプチャして後で検証する ★
        captured_prompt: list[Prompt] = []

        class SpyLLMClient(LLMClientBase):
            def predict(self, prompt: Prompt) -> LLMResult:
                captured_prompt.append(prompt)
                return LLMResult(output="回答")

        usecase = QuestionAnswerUsecase(llm_client=SpyLLMClient())
        usecase.execute("銀行口座の開設方法は？")

        expected_prompt = QuestionAnswerPrompt("銀行口座の開設方法は？").get_prompt()
        assert len(captured_prompt) == 1
        assert captured_prompt[0].text == expected_prompt

    def test_init_stores_llm_client(self) -> None:
        """初期化時にllm_clientが保持される"""

        class StubLLMClient(LLMClientBase):
            def predict(self, prompt: Prompt) -> LLMResult:
                return LLMResult(output="回答")

        client = StubLLMClient()
        usecase = QuestionAnswerUsecase(llm_client=client)
        assert usecase.llm_client is client
```

**テスト観点:**

- **Stub パターン**: 固定値を返す偽の実装を注入して、usecase の出力を検証
- **Spy パターン**: 渡された引数をキャプチャして、usecase が正しくプロンプトを組み立てているかを検証
- `unittest.mock` は使わない。テスト内で `LLMClientBase` を継承した Stub / Spy クラスを定義する
- **infra 層のコード（`GeminiClient` 等）は import しない**。usecase のテストは domain 層の抽象だけで完結する

### 6.3 infra 層のテスト — Mock で外部 API を差し替え

infra 層のテストでは、**外部 SDK の呼び出しを `unittest.mock.patch` でモックする**。

**実際のコード: `tests/infra/llm_client/test_gemini.py`（抜粋）**

```python
"""Gemini APIを使用したLLMクライアントのテスト."""
# ruff: noqa: S101

from types import SimpleNamespace
from unittest.mock import patch

from app.domain.llm_client.llm_client_base import LLMResult, Prompt
from app.infra.llm_client.gemini import GeminiClient


class TestGeminiClient:
    """GeminiClientのテスト"""

    def test_init_default_parameters(self) -> None:
        """デフォルトパラメータで初期化できる"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            client = GeminiClient(api_key="test-api-key")

            mock_client_cls.assert_called_once_with(api_key="test-api-key")
            assert client.model == "gemini-2.5-flash-lite"
            assert client.include_thoughts is True

    def test_predict_with_thinking_and_output(self) -> None:
        """思考ログと出力の両方があるレスポンスを正しくパースする"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            # ★ SimpleNamespace で Gemini API のレスポンス構造を再現 ★
            thinking_part = SimpleNamespace(text="思考内容", thought=True)
            output_part = SimpleNamespace(text="回答テキスト", thought=False)
            response = SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[thinking_part, output_part]),
                    ),
                ],
            )
            mock_client_cls.return_value.models.generate_content.return_value = response

            client = GeminiClient(api_key="test-key")
            result = client.predict(Prompt(text="テスト質問"))

            assert result.output == "回答テキスト"
            assert result.thinking == "思考内容"

    def test_predict_calls_api_with_correct_parameters(self) -> None:
        """APIが正しいモデル名とプロンプト内容で呼び出される"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            part = SimpleNamespace(text="回答", thought=False)
            response = SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
            )
            mock_generate = mock_client_cls.return_value.models.generate_content
            mock_generate.return_value = response

            client = GeminiClient(api_key="test-key", model="gemini-pro")
            client.predict(Prompt(text="テストプロンプト"))

            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args
            assert call_kwargs.kwargs["model"] == "gemini-pro"
            assert call_kwargs.kwargs["contents"] == ["テストプロンプト"]
```

**テスト観点:**

- `unittest.mock.patch` で **外部 SDK（`genai.Client`）をモック** する。実際の API は呼ばない
- `SimpleNamespace` で Gemini API のレスポンス構造を再現する
- レスポンスのパース処理（thinking / output の分離）が正しく動作することを網羅的に検証
- API に渡されるパラメータ（モデル名、プロンプト内容）が正しいことを検証

> **Stub と Mock の使い分け:**
>
> | パターン | 使用場所 | 方法 | 目的 |
> |---|---|---|---|
> | **Stub** | usecase テスト | `LLMClientBase` を継承したクラスをテスト内で定義 | 固定値を返して usecase のロジックを検証 |
> | **Mock** | infra テスト | `unittest.mock.patch` で外部 SDK を差し替え | API 呼び出し自体をモックして infra のパース処理を検証 |

### 6.4 usecase DTO のテスト

**実際のコード: `tests/usecase/question_answer/test_dto.py`**

```python
"""質問に対する回答を生成するユースケースのDTOのテスト."""
# ruff: noqa: S101

import pytest
from pydantic import ValidationError

from app.usecase.question_answer.dto import QuestionAnswerResponse


class TestQuestionAnswerResponse:
    """QuestionAnswerResponse DTOのテスト"""

    def test_create_with_required_fields(self) -> None:
        """必須フィールドを指定して生成できる"""
        response = QuestionAnswerResponse(answer="回答です")
        assert response.answer == "回答です"

    def test_required_field_validation(self) -> None:
        """必須フィールドを省略するとValidationErrorになる"""
        with pytest.raises(ValidationError, match="answer"):
            QuestionAnswerResponse()  # type: ignore[call-arg]
```

### 6.5 テスト実行コマンド

```bash
# 全テスト実行
uv run pytest

# カバレッジ付き実行（100% 必須）
uv run pytest --cov=app --cov-report=term-missing --cov-fail-under=100

# 特定レイヤーのみ
uv run pytest tests/domain/
uv run pytest tests/usecase/
uv run pytest tests/infra/
```

---

## 7. DTO の配置ルールまとめ

DTO が「どのレイヤーに置かれるか」は初心者が最も迷うポイント。以下の表で整理する。

| DTO クラス | 配置場所 | 役割 | 誰が使うか |
|---|---|---|---|
| `Prompt` | **domain** (`llm_client/llm_client_base.py`) | LLM への入力 | usecase, infra |
| `LLMResult` | **domain** (`llm_client/llm_client_base.py`) | LLM からの出力 | usecase, infra |
| `QuestionAnswerResponse` | **usecase** (`question_answer/dto.py`) | ユースケースの出力 | usecase, command |
| `Request` | **command** (`question.py`) | API リクエスト | command のみ |
| `Response` | **command** (`question.py`) | API レスポンス | command のみ |
| `Status` | **command** (`status.py`) | レスポンスステータス | command のみ |

### 判断基準

```
「この DTO は誰のための概念か？」

  ├── ビジネス上の概念（LLM入出力など）      → domain 層
  ├── ユースケースの入出力                   → usecase 層
  └── API / CLI のリクエスト・レスポンス形式  → command 層
```

### データの流れと DTO の変換

```
[command 層]         [usecase 層]         [domain 層]           [infra 層]
Request              question (str)       QuestionAnswerPrompt  Prompt
  │                      │                      │                  │
  │  question を渡す     │  プロンプト生成       │  Prompt に変換     │
  ▼                      ▼                      ▼                  ▼
process()  ──→  usecase.execute()  ──→  build_prompt()  ──→  predict()
                                                                   │
  ▲                      ▲                                         │
  │  Response に変換     │  LLMResult → QuestionAnswerResponse    │
  │                      │                                         │
Response        QuestionAnswerResponse              LLMResult ◄────┘
```

---

## 8. 新機能を追加するときの手順

例：「文書要約機能」を追加するケースで説明する。

### Step 1: domain 層 — 抽象・DTO・ドメインロジックを定義

```python
# app/domain/summarize/build_prompt.py
class SummarizePrompt:
    """文書要約のプロンプト"""

    def __init__(self, document: str) -> None:
        self.document = document

    def get_prompt(self) -> str:
        return f"""以下の文書を要約してください。
        文書: {self.document}
        """
```

> LLM クライアントの抽象（`LLMClientBase`）は既に domain 層にあるので再利用する。

### Step 2: usecase 層 — ユースケースと DTO を作成

```python
# app/usecase/summarize/dto.py
from pydantic import BaseModel

class SummarizeResponse(BaseModel):
    """文書要約ユースケースのDTO"""
    summary: str
```

```python
# app/usecase/summarize/usecase.py
from app.domain.llm_client.llm_client_base import LLMClientBase, Prompt
from app.domain.summarize.build_prompt import SummarizePrompt
from app.usecase.summarize.dto import SummarizeResponse

class SummarizeUsecase:
    """文書要約ユースケース"""

    def __init__(self, llm_client: LLMClientBase) -> None:
        self.llm_client = llm_client

    def execute(self, document: str) -> SummarizeResponse:
        input_prompt = SummarizePrompt(document).get_prompt()
        result = self.llm_client.predict(Prompt(text=input_prompt))
        return SummarizeResponse(summary=result.output)
```

### Step 3: infra 層 — 既存の `GeminiClient` を再利用（追加不要）

LLM クライアントは既に infra 層に存在するので、新たなクラスは不要。

### Step 4: command 層 — エントリーポイントを作成

```python
# app/command/summarize.py
from app.infra.llm_client.gemini import GeminiClient          # ← DI のため infra を import
from app.usecase.summarize.usecase import SummarizeUsecase

def process(document: str) -> ...:
    usecase = SummarizeUsecase(
        llm_client=GeminiClient(api_key=os.getenv("GEMINI_API_KEY")),  # ← DI
    )
    return usecase.execute(document)
```

### Step 5: テスト — 各レイヤーのテストを作成

```
tests/domain/summarize/test_prompt.py        ← 純粋ロジックテスト
tests/usecase/summarize/test_dto.py          ← DTO バリデーション
tests/usecase/summarize/test_usecase.py      ← Stub で LLM を差し替え
```

### Step 6: カバレッジ確認

```bash
uv run pytest --cov=app --cov-report=term-missing --cov-fail-under=100
```

---

## まとめ：初心者向けチートシート

### 「どこに何を書くか」早見表

| 書きたいもの | 配置場所 | import ルール |
|---|---|---|
| LLM の抽象インターフェース | `app/domain/` | 外部依存なし |
| ビジネス DTO（Prompt, LLMResult 等） | `app/domain/` | 外部依存なし |
| プロンプト生成ロジック | `app/domain/` | 外部依存なし |
| ユースケースの DTO | `app/usecase/` | domain のみ |
| ユースケース本体 | `app/usecase/` | domain のみ |
| 外部 API クライアント実装 | `app/infra/` | domain のみ |
| CLI / Lambda エントリーポイント | `app/command/` | 全レイヤー可 |
| DI（具象クラスの選択と注入） | `app/command/` | infra + usecase |

### 「やってはいけないこと」チェックリスト

- [ ] usecase 層で `from app.infra.* import ...` していないか？
- [ ] domain 層で `from app.usecase.* import ...` していないか？
- [ ] infra 層で `from app.usecase.* import ...` していないか？
- [ ] usecase のテストで実際の API を呼んでいないか？（Stub を使っているか？）
- [ ] infra のテストで実際の API を呼んでいないか？（Mock を使っているか？）
- [ ] 新規モジュールのカバレッジが 100% か？
