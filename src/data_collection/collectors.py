"""
Data collection orchestrators.

D&T API와 NBA Stats API를 통합하여 체계적으로 데이터를 수집합니다.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.data_collection.dnt_client import DNTClient, create_dnt_client
from src.data_collection.nba_stats_client import NBAStatsClient, create_nba_stats_client
from src.data_collection.validators import (
    validate_team_epm_data,
    validate_player_epm_data,
    validate_game_data,
    log_validation_report,
    ValidationReport
)
from src.utils.logger import logger
from src.utils.helpers import (
    format_date,
    parse_date,
    get_season_date_range,
    save_dataframe,
    ensure_dir
)
from config.settings import settings


@dataclass
class SeasonData:
    """시즌 데이터 컨테이너"""
    season: int
    team_epm: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_epm: pd.DataFrame = field(default_factory=pd.DataFrame)
    season_epm: pd.DataFrame = field(default_factory=pd.DataFrame)
    games: pd.DataFrame = field(default_factory=pd.DataFrame)
    boxscores: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """데이터 완전성 확인"""
        return (
            not self.team_epm.empty and
            not self.games.empty
        )

    def summary(self) -> Dict[str, int]:
        """데이터 요약"""
        return {
            "season": self.season,
            "team_epm_records": len(self.team_epm),
            "player_epm_records": len(self.player_epm),
            "season_epm_records": len(self.season_epm),
            "games": len(self.games),
            "boxscores": len(self.boxscores)
        }


class DataCollector:
    """
    통합 데이터 수집기.

    D&T API와 NBA Stats API를 사용하여 학습 데이터를 수집합니다.
    """

    def __init__(
        self,
        dnt_client: DNTClient,
        nba_client: NBAStatsClient,
        data_dir: Path
    ):
        """
        Args:
            dnt_client: D&T API 클라이언트
            nba_client: NBA Stats API 클라이언트
            data_dir: 데이터 저장 디렉토리
        """
        self.dnt = dnt_client
        self.nba = nba_client
        self.data_dir = data_dir

        # 저장 경로 설정
        self.dnt_dir = data_dir / "raw" / "dnt"
        self.nba_dir = data_dir / "raw" / "nba_stats"

        # 디렉토리 생성
        ensure_dir(self.dnt_dir / "team_epm")
        ensure_dir(self.dnt_dir / "player_epm")
        ensure_dir(self.dnt_dir / "season_epm")
        ensure_dir(self.nba_dir / "games")
        ensure_dir(self.nba_dir / "boxscores")

    async def collect_season(
        self,
        season: int,
        collect_boxscores: bool = False,
        progress_callback: Optional[callable] = None
    ) -> SeasonData:
        """
        단일 시즌 전체 데이터 수집.

        Args:
            season: 시즌 연도 (예: 2025 for 24-25)
            collect_boxscores: 박스스코어 수집 여부
            progress_callback: 진행률 콜백

        Returns:
            SeasonData 객체
        """
        logger.info(f"Starting data collection for season {season}")
        season_data = SeasonData(season=season)

        total_steps = 4 if collect_boxscores else 3
        current_step = 0

        try:
            # Step 1: Team EPM 수집
            logger.info(f"[{season}] Collecting team EPM...")
            team_epm_raw = await self.dnt.get_team_epm(season=season)

            # 검증
            report = validate_team_epm_data(team_epm_raw)
            log_validation_report(report, f"Season {season} Team EPM")

            if report.is_valid:
                season_data.team_epm = pd.DataFrame(team_epm_raw)
                self._save_data(
                    season_data.team_epm,
                    self.dnt_dir / "team_epm" / f"season_{season}.parquet"
                )

            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps)

            # Step 2: Season EPM 수집
            logger.info(f"[{season}] Collecting season EPM...")
            await asyncio.sleep(2)  # Rate limit 보호

            season_epm_raw = await self.dnt.get_season_epm(season=season)

            if season_epm_raw:
                season_data.season_epm = pd.DataFrame(season_epm_raw)
                self._save_data(
                    season_data.season_epm,
                    self.dnt_dir / "season_epm" / f"season_{season}.parquet"
                )

            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps)

            # Step 3: NBA Stats 경기 데이터 수집
            logger.info(f"[{season}] Collecting game data from NBA Stats...")
            games_df = self.nba.get_team_game_logs(season=season)

            report = validate_game_data(games_df)
            log_validation_report(report, f"Season {season} Games")

            if not games_df.empty:
                # 홈/어웨이 구분을 위한 처리
                games_df = self._process_games_data(games_df)
                season_data.games = games_df
                self._save_data(
                    games_df,
                    self.nba_dir / "games" / f"season_{season}.parquet"
                )

            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps)

            # Step 4: 박스스코어 수집 (선택적)
            if collect_boxscores and not games_df.empty:
                logger.info(f"[{season}] Collecting boxscores...")
                game_ids = games_df["game_id"].unique().tolist()

                boxscores = await self._collect_boxscores_batch(game_ids)
                season_data.boxscores = boxscores

                current_step += 1
                if progress_callback:
                    progress_callback(current_step / total_steps)

        except Exception as e:
            logger.error(f"Error collecting season {season}: {e}")
            raise

        logger.info(f"Season {season} collection complete: {season_data.summary()}")
        return season_data

    async def collect_all_seasons(
        self,
        seasons: List[int],
        collect_boxscores: bool = False
    ) -> Dict[int, SeasonData]:
        """
        다중 시즌 데이터 수집.

        Args:
            seasons: 시즌 리스트
            collect_boxscores: 박스스코어 수집 여부

        Returns:
            시즌별 데이터 딕셔너리
        """
        results = {}

        for season in seasons:
            try:
                data = await self.collect_season(
                    season,
                    collect_boxscores=collect_boxscores
                )
                results[season] = data

                # 시즌 간 대기 (API 부하 방지)
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Failed to collect season {season}: {e}")
                results[season] = SeasonData(season=season)

        return results

    async def collect_daily_data(
        self,
        game_date: str
    ) -> Dict[str, Any]:
        """
        특정 날짜 데이터 수집 (일일 예측용).

        Args:
            game_date: 날짜 (YYYY-MM-DD)

        Returns:
            일일 데이터 딕셔너리
        """
        logger.info(f"Collecting data for {game_date}")

        result = {
            "date": game_date,
            "team_epm": [],
            "player_epm": [],
            "games": None
        }

        try:
            # Team EPM
            team_epm = await self.dnt.get_team_epm(date=game_date)
            result["team_epm"] = team_epm

            # Player EPM (경기 있는 팀만)
            player_epm = await self.dnt.get_player_epm(date=game_date)
            result["player_epm"] = player_epm

            # 스코어보드
            scoreboard = self.nba.get_scoreboard(game_date)
            result["games"] = scoreboard.get("games")

        except Exception as e:
            logger.error(f"Error collecting daily data for {game_date}: {e}")

        return result

    async def _collect_boxscores_batch(
        self,
        game_ids: List[str],
        batch_size: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """박스스코어 배치 수집"""
        results = {}

        for i in range(0, len(game_ids), batch_size):
            batch = game_ids[i:i + batch_size]

            for game_id in batch:
                try:
                    boxscore = await self.nba.get_boxscore_async(game_id)
                    if boxscore["team"] is not None and not boxscore["team"].empty:
                        results[game_id] = boxscore["team"]
                except Exception as e:
                    logger.warning(f"Failed to get boxscore for {game_id}: {e}")

                await asyncio.sleep(0.6)  # Rate limiting

            logger.info(f"Collected {len(results)}/{len(game_ids)} boxscores")

        return results

    def _process_games_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        경기 데이터 처리.

        - 홈/어웨이 구분
        - 날짜 형식 표준화
        - 점수차 계산
        """
        if games_df.empty:
            return games_df

        df = games_df.copy()

        # 컬럼명이 대문자인 경우 처리
        if "MATCHUP" in df.columns:
            df.columns = df.columns.str.lower()

        # 홈/어웨이 구분 (matchup 컬럼 기준)
        if "matchup" in df.columns:
            df["is_home"] = df["matchup"].str.contains("vs.", regex=False)

        # 날짜 형식 표준화
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

        return df

    def _save_data(self, df: pd.DataFrame, path: Path) -> None:
        """데이터 저장"""
        if df.empty:
            logger.warning(f"Empty DataFrame, skipping save to {path}")
            return

        ensure_dir(path.parent)
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} records to {path}")

    def load_season_data(self, season: int) -> SeasonData:
        """저장된 시즌 데이터 로드"""
        season_data = SeasonData(season=season)

        # Team EPM
        team_epm_path = self.dnt_dir / "team_epm" / f"season_{season}.parquet"
        if team_epm_path.exists():
            season_data.team_epm = pd.read_parquet(team_epm_path)

        # Season EPM
        season_epm_path = self.dnt_dir / "season_epm" / f"season_{season}.parquet"
        if season_epm_path.exists():
            season_data.season_epm = pd.read_parquet(season_epm_path)

        # Games
        games_path = self.nba_dir / "games" / f"season_{season}.parquet"
        if games_path.exists():
            season_data.games = pd.read_parquet(games_path)

        return season_data


class HistoricalDataCollector:
    """
    과거 데이터 전체 수집기.

    학습에 필요한 모든 과거 시즌 데이터를 수집합니다.
    """

    def __init__(self, data_dir: Path, api_key: str):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
            api_key: D&T API 키
        """
        self.data_dir = data_dir
        self.api_key = api_key

    async def collect_all(
        self,
        seasons: List[int],
        include_boxscores: bool = False
    ) -> Dict[int, SeasonData]:
        """
        전체 과거 데이터 수집.

        Args:
            seasons: 수집할 시즌 리스트
            include_boxscores: 박스스코어 포함 여부

        Returns:
            시즌별 데이터 딕셔너리
        """
        cache_dir = self.data_dir / "cache"

        async with create_dnt_client(self.api_key, cache_dir) as dnt_client:
            nba_client = create_nba_stats_client(cache_dir / "nba")

            collector = DataCollector(dnt_client, nba_client, self.data_dir)

            results = await collector.collect_all_seasons(
                seasons,
                collect_boxscores=include_boxscores
            )

            nba_client.close()

        return results

    def get_collection_status(self, seasons: List[int]) -> Dict[int, Dict[str, bool]]:
        """수집 상태 확인"""
        status = {}

        for season in seasons:
            dnt_dir = self.data_dir / "raw" / "dnt"
            nba_dir = self.data_dir / "raw" / "nba_stats"

            status[season] = {
                "team_epm": (dnt_dir / "team_epm" / f"season_{season}.parquet").exists(),
                "season_epm": (dnt_dir / "season_epm" / f"season_{season}.parquet").exists(),
                "games": (nba_dir / "games" / f"season_{season}.parquet").exists(),
            }

        return status


# ===================
# Convenience Functions
# ===================

async def collect_training_data(
    api_key: str,
    data_dir: Path,
    seasons: Optional[List[int]] = None
) -> Dict[int, SeasonData]:
    """
    학습 데이터 수집 헬퍼 함수.

    Args:
        api_key: D&T API 키
        data_dir: 데이터 디렉토리
        seasons: 시즌 리스트 (기본: 설정의 training_seasons)

    Returns:
        시즌별 데이터 딕셔너리
    """
    if seasons is None:
        seasons = settings.training_seasons + [settings.validation_season]

    collector = HistoricalDataCollector(data_dir, api_key)
    return await collector.collect_all(seasons, include_boxscores=False)
