"""質問に対する回答を生成するユースケースのテスト."""
# ruff: noqa: S101, ARG002, RUF001

from app.domain.llm_client.llm_client_base import LLMClientBase, LLMResult, Prompt
from app.domain.question_answer.build_prompt import QuestionAnswerPrompt
from app.usecase.question_answer.dto import QuestionAnswerResponse
from app.usecase.question_answer.usecase import QuestionAnswerUsecase

# ================================
# QuestionAnswerUsecase
# ================================


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
