"""質問に対する回答を生成するユースケースのDTOのテスト."""
# ruff: noqa: S101

import pytest
from pydantic import ValidationError

from app.usecase.question_answer.dto import QuestionAnswerResponse

# ================================
# QuestionAnswerResponse
# ================================


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
