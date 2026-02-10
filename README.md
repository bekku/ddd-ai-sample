# DDD AI Sample

DDD（ドメイン駆動設計）に基づいた AI アプリケーションのサンプルリポジトリ。
Gemini API を利用した質問応答システムを題材に、AI を活用したアプリケーションにおける DDD のレイヤー構成・テスト戦略・開発ルールを示す。

---

## 技術スタック

| カテゴリ | 技術 |
|---|---|
| 言語 | Python 3.12+ |
| パッケージマネージャ | [uv](https://docs.astral.sh/uv/) |
| LLM | Google Gemini API (`google-genai`) |
| バリデーション | Pydantic v2 |
| テスト | pytest / pytest-cov |
| リンター | Ruff |
| 実行環境 | CLI / AWS Lambda |

---

## ディレクトリ構成

```
.
├── app/                          # プロダクションコード
│   ├── command/                   #   エントリーポイント層（CLI / Lambda）
│   │   ├── __init__.py
│   │   ├── question.py            #     質問応答の CLI & Lambda ハンドラ
│   │   └── status.py              #     API レスポンスステータス Enum
│   ├── domain/                    #   ドメイン層（ビジネスロジック）
│   │   ├── llm_client/
│   │   │   ├── __init__.py
│   │   │   └── llm_client_base.py #     LLM クライアントの抽象基底クラス & DTO
│   │   └── question_answer/
│   │       ├── __init__.py
│   │       └── build_prompt.py    #     プロンプト生成ロジック
│   ├── infra/                     #   インフラ層（外部サービス連携）
│   │   └── llm_client/
│   │       └── gemini.py          #     Gemini API クライアント実装
│   └── usecase/                   #   ユースケース層（アプリケーションロジック）
│       └── question_answer/
│           ├── dto.py             #     ユースケースの入出力 DTO
│           └── usecase.py         #     質問応答ユースケース
├── tests/                        # テストコード（app/ のミラー構成）
│   ├── domain/
│   │   ├── llm_client/
│   │   │   └── test_llm_client_base.py
│   │   └── question_answer/
│   │       └── test_prompt.py
│   ├── infra/
│   │   └── llm_client/
│   │       └── test_gemini.py
│   ├── usecase/
│   │   └── question_answer/
│   │       ├── test_dto.py
│   │       └── test_usecase.py
│   └── README.md                 # テストルール・ガイドライン
├── .env.default                  # 環境変数テンプレート
├── .python-version               # Python バージョン指定
├── pyproject.toml                # プロジェクト設定
└── uv.lock                       # 依存関係ロックファイル
```

---

## アーキテクチャ（DDD レイヤー構成）

本プロジェクトは DDD のレイヤードアーキテクチャを採用している。

```
┌─────────────────────────────────────────┐
│              command 層                  │  ← エントリーポイント（CLI / Lambda）
│  リクエスト受付 → ユースケース呼び出し      │
├─────────────────────────────────────────┤
│              usecase 層                  │  ← アプリケーションロジック
│  ドメインオブジェクトを組み合わせて処理      │
├─────────────────────────────────────────┤
│              domain 層                   │  ← ビジネスロジック（中核）
│  抽象インターフェース / プロンプト生成       │
├─────────────────────────────────────────┤
│              infra 層                    │  ← 外部サービス連携
│  Gemini API クライアント等の具象実装        │
└─────────────────────────────────────────┘
```

### 各レイヤーの役割

| レイヤー | 役割 | 依存方向 |
|---|---|---|
| **command** | CLI / Lambda のエントリーポイント。リクエストの受付・バリデーション・レスポンス生成 | usecase, infra に依存 |
| **usecase** | アプリケーション固有のビジネスフローを実行。ドメインオブジェクトを組み合わせる | domain に依存 |
| **domain** | ビジネスロジックの中核。抽象インターフェース（`LLMClientBase`）・DTO（`Prompt`, `LLMResult`）・プロンプト生成を定義 | 他レイヤーに依存しない |
| **infra** | 外部 API（Gemini API）との連携を担う具象実装。domain 層の抽象を実装する | domain に依存 |

### データフロー

```
Request → command/question.py → QuestionAnswerUsecase.execute()
                                        │
                                        ├── QuestionAnswerPrompt.get_prompt()  [domain]
                                        │
                                        └── LLMClientBase.predict()            [domain → infra]
                                                │
                                                └── GeminiClient (Gemini API)  [infra]
                                                        │
                                                        └── LLMResult → QuestionAnswerResponse
```

---

## セットアップ

### 前提条件

- Python 3.12 以上
- [uv](https://docs.astral.sh/uv/) がインストールされていること

### 手順

```bash
# 1. リポジトリのクローン
git clone <repository-url>
cd deep-bank-ai-sample

# 2. 依存関係のインストール
uv sync

# 3. 環境変数の設定
cp .env.default .env
# .env ファイルを編集し、GEMINI_API_KEY を設定する
```

### 環境変数

| 変数名 | 必須 | 説明 |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API のキー |

---

## 実行方法

### CLI から実行

```bash
uv run python -m app.command.question --question "あなたの質問をここに入力"
```

### AWS Lambda として実行

`app/command/question.py` の `lambda_handler` 関数がエントリーポイントとなる。

```python
# Lambda イベント形式
{
    "question": "質問内容"
}
```

---

## 開発ルール

### コード規約

| 項目 | ルール |
|---|---|
| 言語バージョン | Python 3.12 以上 |
| 型アノテーション | 全ての関数・メソッドに型アノテーションを付与する |
| docstring | 日本語で記述する |
| バリデーション | DTO は Pydantic `BaseModel` を継承して定義する |
| リンター | Ruff を使用する |
| パッケージ管理 | uv を使用。依存関係は `pyproject.toml` で管理し、`uv.lock` でバージョンを固定する |

### DDD レイヤールール

| ルール | 説明 |
|---|---|
| **domain 層は外部に依存しない** | domain 層は他のレイヤーに依存してはならない。抽象インターフェースとドメインロジックのみを定義する |
| **依存性の逆転（DIP）** | infra 層は domain 層の抽象インターフェースを実装する。usecase 層は抽象に依存し、具象実装は command 層で注入する |
| **ディレクトリ構成はレイヤーに従う** | `app/domain/`, `app/usecase/`, `app/infra/`, `app/command/` の 4 レイヤー |
| **DTO の配置** | レイヤー間のデータ受け渡しは DTO（Pydantic BaseModel）を通じて行う |

### 新機能追加の手順

1. **domain 層**: 必要な抽象インターフェース・DTO・ドメインロジックを定義
2. **infra 層**: domain 層の抽象を実装する具象クラスを作成
3. **usecase 層**: ドメインオブジェクトを組み合わせたユースケースを作成
4. **command 層**: エントリーポイント（CLI / Lambda）を作成
5. **tests/**: 各レイヤーに対応するテストを作成（カバレッジ 100% 必須）

---

## テスト

テストの詳細なルール・ガイドラインは [`tests/README.md`](tests/README.md) を参照。

### テスト実行

```bash
# 全テスト実行
uv run pytest

# カバレッジ付き実行（100% 必須）
uv run pytest --cov=app --cov-report=term-missing --cov-fail-under=100

# HTML レポート生成
uv run pytest --cov=app --cov-report=term-missing --cov-report=html

# 特定レイヤーのテスト実行
uv run pytest tests/domain/
uv run pytest tests/usecase/
uv run pytest tests/infra/
```

### テスト方針

| レイヤー | テスト対象 | 外部依存の扱い | テスト方針 |
|---|---|---|---|
| **domain** | DTO, 抽象クラス, プロンプト生成 | なし | 純粋なロジックテスト |
| **usecase** | ユースケースクラス | スタブに差し替え | 依存を注入してテスト |
| **infra** | API クライアント等 | モックに差し替え | 実 API を呼ばずにテスト |
| **command** | エントリーポイント | モックに差し替え | E2E に近い形でテスト |

### カバレッジルール

- **`app/` 配下のすべてのプロダクションコードは行カバレッジ 100% を維持する**
- 新しいモジュールを追加したら、対応するテストで全行・全分岐をカバーする
- テストファイルは `app/` と同じレイヤー・ディレクトリ構成で `tests/` に配置する
- テストファイル名は `test_<対応するモジュール名>.py` とする

### テストファイルの規約

```python
"""<モジュール概要>のテスト."""
# ruff: noqa: S101

import pytest
from pydantic import ValidationError

from app.<レイヤー>.<モジュールパス> import <テスト対象クラス>

# ================================
# <テスト対象クラス名>
# ================================


class Test<テスト対象クラス名>:
    """<テスト対象クラス名>のテスト"""

    def test_<振る舞いの説明>(self) -> None:
        """<日本語でテスト内容を説明する docstring>"""
        ...
```

---

## 依存関係

| パッケージ | バージョン | 用途 |
|---|---|---|
| `google-genai` | 1.56.0 | Google Gemini API クライアント |
| `pydantic` | 2.12.5 | データバリデーション・DTO 定義 |
| `pytest` | >=9.0.2 | テストフレームワーク |
| `pytest-cov` | >=7.0.0 | カバレッジ計測 |
| `python-dotenv` | >=1.2.1 | `.env` ファイルからの環境変数読み込み |
