"""LLMクライアントインターフェースおよびその入出力DTOの定義"""

from abc import ABC, abstractmethod

from pydantic import BaseModel

# ================================
# 入出力DTO
# ================================


class Prompt(BaseModel):
    """LLMに送信するプロンプトを表すイミュータブルなデータクラス"""

    text: str


class LLMResult(BaseModel):
    """LLMの応答結果を表すモデル"""

    output: str
    thinking: str | None = None


# ================================
# インターフェース
# ================================


class LLMClientBase(ABC):
    """LLMクライアントの抽象基底クラス"""

    @abstractmethod
    def predict(self, prompt: Prompt) -> LLMResult:
        """プロンプトを実行し、結果を返す"""
        ...
