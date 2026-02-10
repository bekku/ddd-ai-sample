"""質問に対する回答を生成するユースケース"""

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
        input_prompt: str = QuestionAnswerPrompt(question).get_prompt()
        result: LLMResult = self.llm_client.predict(Prompt(text=input_prompt))
        return QuestionAnswerResponse(answer=result.output)
