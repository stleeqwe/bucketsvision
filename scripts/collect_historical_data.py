#!/usr/bin/env python3
"""
과거 시즌 데이터 수집 스크립트.

학습에 필요한 22-23, 23-24, 24-25 시즌 데이터를 수집합니다.

Usage:
    python scripts/collect_historical_data.py
    python scripts/collect_historical_data.py --seasons 2023 2024
    python scripts/collect_historical_data.py --include-boxscores
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.data_collection.collectors import HistoricalDataCollector
from src.utils.logger import logger, setup_file_logging


async def main(seasons: list, include_boxscores: bool = False):
    """메인 수집 함수"""

    # 로그 디렉토리 설정
    log_dir = settings.project_root / "logs"
    setup_file_logging(log_dir)

    logger.info("=" * 60)
    logger.info("NBA Score Predictor - Historical Data Collection")
    logger.info("=" * 60)
    logger.info(f"Seasons to collect: {seasons}")
    logger.info(f"Include boxscores: {include_boxscores}")
    logger.info(f"Data directory: {settings.data_dir}")

    # 수집기 초기화
    collector = HistoricalDataCollector(
        data_dir=settings.data_dir,
        api_key=settings.dnt_api_key
    )

    # 기존 수집 상태 확인
    status = collector.get_collection_status(seasons)
    logger.info("Current collection status:")
    for season, s in status.items():
        logger.info(f"  Season {season}: {s}")

    # 데이터 수집
    logger.info("Starting data collection...")

    try:
        results = await collector.collect_all(
            seasons=seasons,
            include_boxscores=include_boxscores
        )

        # 결과 요약
        logger.info("=" * 60)
        logger.info("Collection Summary")
        logger.info("=" * 60)

        for season, data in results.items():
            summary = data.summary()
            logger.info(f"Season {season}:")
            logger.info(f"  Team EPM records: {summary['team_epm_records']}")
            logger.info(f"  Season EPM records: {summary['season_epm_records']}")
            logger.info(f"  Games: {summary['games']}")
            if include_boxscores:
                logger.info(f"  Boxscores: {summary['boxscores']}")

        logger.info("=" * 60)
        logger.info("Data collection completed successfully!")

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Collect historical NBA data for model training"
    )

    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025],
        help="Seasons to collect (default: 2023 2024 2025)"
    )

    parser.add_argument(
        "--include-boxscores",
        action="store_true",
        help="Include detailed boxscore data (slower)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be collected without actually collecting"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dry_run:
        print(f"Would collect data for seasons: {args.seasons}")
        print(f"Include boxscores: {args.include_boxscores}")
        sys.exit(0)

    asyncio.run(main(
        seasons=args.seasons,
        include_boxscores=args.include_boxscores
    ))
