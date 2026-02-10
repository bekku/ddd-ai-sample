"""質問に対する回答を生成するコマンド"""

import argparse
import os

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from app.command.status import Status
from app.infra.llm_client.gemini import GeminiClient
from app.usecase.question_answer.dto import QuestionAnswerResponse
from app.usecase.question_answer.usecase import QuestionAnswerUsecase

load_dotenv()


class Request(BaseModel):
    """質問に対する回答を生成するリクエスト"""

    question: str


class Response(BaseModel):
    """APIレスポンスラッパー"""

    status: Status
    data: QuestionAnswerResponse


def validate_env() -> None:
    """環境変数を検証"""
    # GEMINI_API_KEYが設定されているか確認
    if os.getenv("GEMINI_API_KEY") is None:
        msg = "GEMINI_API_KEY is required"
        raise ValueError(msg)


def process(request: Request) -> Response:
    """ユースケースを実行する"""
    # 0. 環境変数を検証
    validate_env()

    # 1. ユースケースを定義
    usecase = QuestionAnswerUsecase(
        llm_client=GeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.5-flash-lite",
        ),
    )

    # 2. ユースケースを実行
    try:
        response = usecase.execute(request.question)
    except Exception as e:
        error_message = f"Exception: {e}"
        raise ValueError(error_message) from e

    # 3. レスポンスを返す
    return Response(status=Status.SUCCESS, data=response)


def lambda_handler(event, context):  # noqa: ARG001, ANN201, ANN001
    """AWS Lambda handler function"""
    try:
        req = Request.model_validate(event)
    except ValidationError as e:
        error_message = f"ValidationError: {e}"
        raise ValueError(error_message) from e

    resp = process(req)

    return resp.model_dump_json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="質問に対する回答を生成する")
    parser.add_argument(
        "--question",
        required=True,
        help="質問内容",
    )

    args = parser.parse_args()

    try:
        # リクエストを作成
        request = Request(question=args.question)

        response = process(request)
        print(response.model_dump_json())
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        exit(1)  # noqa: PLR1722
