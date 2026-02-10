"""status enum"""

from enum import Enum


class Status(str, Enum):
    """APIレスポンスステータス"""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
