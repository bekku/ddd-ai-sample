"""LLMクライアント基底クラスおよびDTOのテスト."""
# ruff: noqa: S101

import pytest
from pydantic import ValidationError

from app.domain.llm_client.llm_client_base import (
    LLMClientBase,
    LLMResult,
    Prompt,
)

# ================================
# Prompt
# ================================


class TestPrompt:
    """Prompt DTOのテスト"""

    def test_create_with_text(self) -> None:
        """textを渡すとPromptが生成される"""
        prompt = Prompt(text="こんにちは")
        assert prompt.text == "こんにちは"

    def test_text_required(self) -> None:
        """textを省略するとValidationErrorになる"""
        with pytest.raises(ValidationError, match="text"):
            Prompt()


# ================================
# LLMResult
# ================================


class TestLLMResult:
    """LLMResult DTOのテスト"""

    def test_create_with_output_only(self) -> None:
        """outputのみ指定した場合、thinkingはNone"""
        result = LLMResult(output="回答です")
        assert result.output == "回答です"
        assert result.thinking is None

    def test_create_with_output_and_thinking(self) -> None:
        """outputとthinkingの両方を指定できる"""
        result = LLMResult(
            output="回答です",
            thinking="思考過程です",
        )
        assert result.output == "回答です"
        assert result.thinking == "思考過程です"


# ================================
# LLMClientBase
# ================================


class TestLLMClientBase:
    """LLMClientBase抽象クラスのテスト"""

    def test_cannot_instantiate_directly(self) -> None:
        """抽象クラスは直接インスタンス化するとTypeError"""
        with pytest.raises(TypeError, match="abstract"):
            LLMClientBase()  # type: ignore[abstract]

    def test_concrete_implementation_predict(self) -> None:
        """具象実装ではpredictが呼び出せる"""

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
