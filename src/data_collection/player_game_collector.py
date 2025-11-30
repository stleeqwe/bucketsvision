"""
Player Game Collector.

NBA Stats API를 사용하여 선수별 경기 출전 기록을 수집합니다.
On/Off 분석을 위한 데이터 기반을 제공합니다.

수집 데이터:
- 경기별 출전 선수 목록
- 출전 시간 (minutes)
- 선발 출전 여부 (started)
- 기본 스탯 (points, rebounds, assists)
"""

import asyncio
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data_collection.nba_stats_client import NBAStatsClient, NBAStatsClientConfig
from src.utils.logger import logger
from src.utils.helpers import get_nba_api_season_string, ensure_dir


class PlayerGameCollector:
    """
    선수별 경기 출전 기록 수집기.

    NBA Stats API의 playergamelogs 엔드포인트를 활용하여
    시즌별 모든 선수의 경기 출전 기록을 수집합니다.
    """

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Optional[Path] = None
    ):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
            cache_dir: API 캐시 디렉토리
        """
        self.data_dir = data_dir
        self.output_dir = data_dir / "raw" / "nba_stats" / "player_games"
        ensure_dir(self.output_dir)

        config = NBAStatsClientConfig(
            timeout=60,
            max_retries=3,
            delay_between_requests=0.8  # Rate limiting
        )
        self.client = NBAStatsClient(config, cache_dir)

    def collect_season_player_games(
        self,
        season: int,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        시즌 전체 선수 경기 출전 기록 수집.

        Args:
            season: 시즌 연도 (예: 2026 for 25-26)
            progress_callback: 진행률 콜백

        Returns:
            선수별 경기 출전 DataFrame
        """
        logger.info(f"Collecting player game logs for season {season}...")

        # NBA Stats API playergamelogs 호출
        player_logs = self.client.get_player_game_logs(
            season=season,
            season_type="Regular Season"
        )

        if player_logs.empty:
            logger.warning(f"No player game logs for season {season}")
            return pd.DataFrame()

        # 컬럼 정규화
        df = self._normalize_columns(player_logs)

        # 추가 정보 계산
        df = self._add_derived_columns(df, season)

        logger.info(f"Collected {len(df)} player game records for season {season}")
        logger.info(f"  Unique players: {df['player_id'].nunique()}")
        logger.info(f"  Unique games: {df['game_id'].nunique()}")

        if progress_callback:
            progress_callback(1.0)

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명 정규화"""
        column_mapping = {
            "PLAYER_ID": "player_id",
            "PLAYER_NAME": "player_name",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbr",
            "TEAM_NAME": "team_name",
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
            "MATCHUP": "matchup",
            "WL": "result",
            "MIN": "minutes_str",
            "PTS": "pts",
            "REB": "reb",
            "AST": "ast",
            "STL": "stl",
            "BLK": "blk",
            "TOV": "tov",
            "FGM": "fgm",
            "FGA": "fga",
            "FG3M": "fg3m",
            "FG3A": "fg3a",
            "FTM": "ftm",
            "FTA": "fta",
            "PLUS_MINUS": "plus_minus",
        }

        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        return df

    def _add_derived_columns(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """파생 컬럼 추가"""
        df = df.copy()

        # 시즌 정보
        df["season"] = season

        # 출전 시간 파싱 (MM:SS 형식)
        df["minutes"] = df["minutes_str"].apply(self._parse_minutes)

        # 출전 여부 (분 > 0)
        df["played"] = df["minutes"] > 0

        # 선발 출전 여부 추정 (분 >= 20이면 선발로 가정, 나중에 정확한 데이터로 업데이트)
        # 실제로는 boxscore에서 START_POSITION 정보로 확인 가능
        df["started"] = df["minutes"] >= 20

        # 홈/어웨이 파싱
        df["is_home"] = df["matchup"].apply(lambda x: "vs." in str(x) if pd.notna(x) else False)

        # 상대팀 추출
        df["opponent"] = df["matchup"].apply(self._extract_opponent)

        # 날짜 형식 정규화
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

        return df

    def _parse_minutes(self, minutes_str) -> float:
        """MM:SS 형식의 출전 시간을 float 분으로 변환"""
        if pd.isna(minutes_str) or minutes_str == "" or minutes_str is None:
            return 0.0

        try:
            if isinstance(minutes_str, (int, float)):
                return float(minutes_str)

            if ":" in str(minutes_str):
                parts = str(minutes_str).split(":")
                minutes = int(parts[0])
                seconds = int(parts[1]) if len(parts) > 1 else 0
                return minutes + seconds / 60.0

            return float(minutes_str)
        except (ValueError, TypeError):
            return 0.0

    def _extract_opponent(self, matchup: str) -> str:
        """매치업 문자열에서 상대팀 추출"""
        if pd.isna(matchup):
            return ""

        matchup = str(matchup)
        if "vs." in matchup:
            return matchup.split("vs.")[-1].strip()
        elif "@" in matchup:
            return matchup.split("@")[-1].strip()
        return ""

    def collect_multiple_seasons(
        self,
        seasons: List[int],
        save_individual: bool = True
    ) -> pd.DataFrame:
        """
        다중 시즌 선수 경기 기록 수집.

        Args:
            seasons: 수집할 시즌 리스트
            save_individual: 시즌별 개별 파일 저장 여부

        Returns:
            병합된 DataFrame
        """
        all_data = []

        for season in seasons:
            logger.info(f"\n{'='*50}")
            logger.info(f"Season {season}")
            logger.info(f"{'='*50}")

            df = self.collect_season_player_games(season)

            if df.empty:
                continue

            if save_individual:
                output_path = self.output_dir / f"season_{season}.parquet"
                df.to_parquet(output_path, index=False)
                logger.info(f"Saved to {output_path}")

            all_data.append(df)

            # Rate limiting between seasons
            time.sleep(2)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        # 전체 데이터 저장
        combined_path = self.output_dir / "all_seasons.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info(f"\nSaved combined data to {combined_path}")
        logger.info(f"Total records: {len(combined)}")

        return combined

    def load_season_player_games(self, season: int) -> pd.DataFrame:
        """저장된 시즌 데이터 로드"""
        path = self.output_dir / f"season_{season}.parquet"
        if not path.exists():
            logger.warning(f"No player games data for season {season}")
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_all_player_games(self) -> pd.DataFrame:
        """저장된 전체 데이터 로드"""
        path = self.output_dir / "all_seasons.parquet"
        if not path.exists():
            logger.warning("No combined player games data found")
            return pd.DataFrame()
        return pd.read_parquet(path)

    def get_player_game_matrix(
        self,
        season: int,
        min_minutes: float = 0.0
    ) -> pd.DataFrame:
        """
        선수-경기 출전 매트릭스 생성.

        선수별로 각 경기 출전 여부를 나타내는 피벗 테이블 생성.

        Args:
            season: 시즌 연도
            min_minutes: 출전으로 인정하는 최소 분

        Returns:
            피벗 테이블 (index: player_id, columns: game_id, values: played)
        """
        df = self.load_season_player_games(season)

        if df.empty:
            return pd.DataFrame()

        # 출전 여부 결정
        df["played"] = df["minutes"] > min_minutes

        # 피벗 테이블 생성
        matrix = df.pivot_table(
            index="player_id",
            columns="game_id",
            values="played",
            aggfunc="max",
            fill_value=False
        )

        return matrix

    def get_team_roster_by_game(
        self,
        season: int,
        team_id: int
    ) -> Dict[str, List[int]]:
        """
        팀의 경기별 출전 선수 목록.

        Args:
            season: 시즌 연도
            team_id: 팀 ID

        Returns:
            {game_id: [player_id1, player_id2, ...]}
        """
        df = self.load_season_player_games(season)

        if df.empty:
            return {}

        team_games = df[
            (df["team_id"] == team_id) &
            (df["played"] == True)
        ]

        roster_by_game = team_games.groupby("game_id")["player_id"].apply(list).to_dict()

        return roster_by_game

    def close(self):
        """리소스 정리"""
        self.client.close()


def collect_player_games_for_seasons(
    data_dir: Path,
    seasons: List[int],
    cache_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    선수 경기 출전 기록 수집 헬퍼 함수.

    Args:
        data_dir: 데이터 디렉토리
        seasons: 수집할 시즌 리스트
        cache_dir: 캐시 디렉토리

    Returns:
        병합된 DataFrame
    """
    collector = PlayerGameCollector(data_dir, cache_dir)

    try:
        return collector.collect_multiple_seasons(seasons)
    finally:
        collector.close()


# CLI 실행
if __name__ == "__main__":
    import argparse
    from config.settings import settings

    parser = argparse.ArgumentParser(description="Collect player game logs")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2025, 2026],
        help="Seasons to collect (e.g., 2025 2026)"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Player Game Collector")
    logger.info("=" * 70)
    logger.info(f"Seasons: {args.seasons}")

    data_dir = settings.data_dir
    cache_dir = data_dir / "cache" / "nba_stats"

    df = collect_player_games_for_seasons(
        data_dir=data_dir,
        seasons=args.seasons,
        cache_dir=cache_dir
    )

    if not df.empty:
        logger.info("\n" + "=" * 70)
        logger.info("Summary")
        logger.info("=" * 70)
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Players: {df['player_id'].nunique()}")
        logger.info(f"Games: {df['game_id'].nunique()}")

        # 시즌별 통계
        for season in args.seasons:
            season_df = df[df["season"] == season]
            logger.info(f"\nSeason {season}:")
            logger.info(f"  Records: {len(season_df)}")
            logger.info(f"  Players: {season_df['player_id'].nunique()}")
            logger.info(f"  Games: {season_df['game_id'].nunique()}")
