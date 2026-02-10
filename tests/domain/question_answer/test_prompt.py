"""プロンプト定義のテスト."""
# ruff: noqa: S101

from app.domain.question_answer.build_prompt import QuestionAnswerPrompt

# ================================
# QuestionAnswerPrompt
# ================================


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
        assert prompt.get_prompt() == expected
