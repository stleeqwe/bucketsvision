"""
Logging configuration using loguru.

이 모듈은 프로젝트 전역 로깅 설정을 제공합니다.
"""

import sys
from pathlib import Path
from loguru import logger

# 기본 로거 설정 제거
logger.remove()

# 콘솔 출력 설정
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)


def setup_file_logging(log_dir: Path, level: str = "DEBUG") -> None:
    """
    파일 로깅 설정.

    Args:
        log_dir: 로그 파일 저장 디렉토리
        level: 로그 레벨
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # 일반 로그
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="00:00",  # 매일 자정 로테이션
        retention="30 days",
        compression="zip"
    )

    # 에러 전용 로그
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        compression="zip"
    )


# 전역 로거 export
__all__ = ["logger", "setup_file_logging"]
