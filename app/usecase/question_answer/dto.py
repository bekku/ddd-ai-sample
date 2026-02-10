"""質問に対する回答を生成するユースケースのDTO"""

from pydantic import BaseModel


class QuestionAnswerResponse(BaseModel):
    """質問に対する回答を生成するユースケースのDTO"""

    answer: str
