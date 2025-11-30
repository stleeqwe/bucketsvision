"""
NBA Stats API Client.

NBA Stats API와 통신하여 경기 일정, 박스스코어, 라인업 데이터를 수집합니다.
nba_api 라이브러리를 래핑하여 일관된 인터페이스를 제공합니다.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logger import logger
from src.utils.cache import DiskCache
from src.utils.helpers import (
    format_date,
    parse_date,
    get_nba_api_season_string,
    get_season_from_date
)


class NBAStatsAPIError(Exception):
    """NBA Stats API 에러"""
    pass


@dataclass
class NBAStatsClientConfig:
    """클라이언트 설정"""
    timeout: int = 30
    max_retries: int = 3
    delay_between_requests: float = 0.6  # API 요청 간 지연 (초)
    cache_ttl: Optional[int] = 86400  # 24시간 기본 캐시


class NBAStatsClient:
    """
    NBA Stats API 클라이언트.

    nba_api 대신 직접 HTTP 요청을 사용하여 더 세밀한 제어를 제공합니다.
    """

    BASE_URL = "https://stats.nba.com/stats"

    # 표준 헤더 (NBA API 접근에 필요)
    HEADERS = {
        "Host": "stats.nba.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }

    def __init__(
        self,
        config: Optional[NBAStatsClientConfig] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Args:
            config: 클라이언트 설정
            cache_dir: 캐시 디렉토리
        """
        self.config = config or NBAStatsClientConfig()

        # 세션 설정 (재시도 로직 포함)
        self._session = self._create_session()

        # 캐시 설정
        self._cache: Optional[DiskCache] = None
        if cache_dir:
            self._cache = DiskCache(cache_dir, default_ttl=self.config.cache_ttl)

        # 비동기 실행을 위한 ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=4)

        # 마지막 요청 시간 (rate limiting)
        self._last_request_time = 0.0

    def _create_session(self) -> requests.Session:
        """재시도 로직이 포함된 세션 생성"""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.headers.update(self.HEADERS)

        return session

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        return f"nba:{endpoint}:{param_str}"

    def _wait_for_rate_limit(self) -> None:
        """Rate limiting 대기"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.delay_between_requests:
            time.sleep(self.config.delay_between_requests - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        동기 API 요청.

        Args:
            endpoint: API 엔드포인트
            params: 쿼리 파라미터
            use_cache: 캐시 사용 여부

        Returns:
            API 응답 데이터
        """
        cache_key = self._get_cache_key(endpoint, params)

        # 캐시 확인
        if use_cache and self._cache:
            cached_data = self._cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_data

        # Rate limiting
        self._wait_for_rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = self._session.get(
                url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()

            # 캐시 저장
            if use_cache and self._cache:
                self._cache.set(cache_key, data)

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"NBA Stats API error for {endpoint}: {e}")
            raise NBAStatsAPIError(f"API request failed: {e}")

    async def _async_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """비동기 API 요청 (ThreadPoolExecutor 사용)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._make_request(endpoint, params, use_cache)
        )

    def _parse_response(
        self,
        data: Dict[str, Any],
        result_set_index: int = 0
    ) -> pd.DataFrame:
        """
        API 응답을 DataFrame으로 변환.

        Args:
            data: API 응답
            result_set_index: resultSets 인덱스

        Returns:
            DataFrame
        """
        try:
            result_sets = data.get("resultSets", [])
            if not result_sets or result_set_index >= len(result_sets):
                return pd.DataFrame()

            result_set = result_sets[result_set_index]
            headers = result_set.get("headers", [])
            rows = result_set.get("rowSet", [])

            return pd.DataFrame(rows, columns=headers)

        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing response: {e}")
            return pd.DataFrame()

    # ===================
    # Public API Methods
    # ===================

    def get_schedule(
        self,
        season: int,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        시즌 경기 일정 조회.

        Args:
            season: 시즌 연도 (예: 2025 for 24-25)
            season_type: 시즌 타입

        Returns:
            경기 일정 DataFrame
        """
        season_str = get_nba_api_season_string(season)

        params = {
            "LeagueID": "00",
            "Season": season_str,
            "SeasonType": season_type,
        }

        try:
            data = self._make_request("leaguegamefinder", params)
            df = self._parse_response(data)

            if df.empty:
                logger.warning(f"No schedule data for season {season}")
                return df

            # 컬럼명 정규화
            df = self._normalize_game_columns(df)

            return df

        except Exception as e:
            logger.error(f"Failed to get schedule for {season}: {e}")
            return pd.DataFrame()

    async def get_schedule_async(
        self,
        season: int,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """비동기 경기 일정 조회"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.get_schedule(season, season_type)
        )

    def get_boxscore(
        self,
        game_id: str
    ) -> Dict[str, pd.DataFrame]:
        """
        경기 박스스코어 조회.

        Args:
            game_id: NBA 경기 ID (예: "0022400001")

        Returns:
            {"team": 팀 스탯 DF, "player": 선수 스탯 DF}
        """
        params = {
            "GameID": game_id,
        }

        try:
            # Traditional boxscore
            data = self._make_request("boxscoretraditionalv2", params)

            result = {
                "player": self._parse_response(data, 0),  # PlayerStats
                "team": self._parse_response(data, 1),    # TeamStats
            }

            return result

        except Exception as e:
            logger.error(f"Failed to get boxscore for {game_id}: {e}")
            return {"player": pd.DataFrame(), "team": pd.DataFrame()}

    async def get_boxscore_async(self, game_id: str) -> Dict[str, pd.DataFrame]:
        """비동기 박스스코어 조회"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.get_boxscore(game_id)
        )

    def get_team_game_logs(
        self,
        season: int,
        team_id: Optional[int] = None,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        팀 경기 로그 조회.

        Args:
            season: 시즌 연도
            team_id: 팀 ID (None이면 전체 팀)
            season_type: 시즌 타입

        Returns:
            팀 경기 로그 DataFrame
        """
        season_str = get_nba_api_season_string(season)

        params = {
            "LeagueID": "00",
            "Season": season_str,
            "SeasonType": season_type,
        }

        if team_id:
            params["TeamID"] = team_id

        try:
            data = self._make_request("teamgamelogs", params)
            df = self._parse_response(data)

            return df

        except Exception as e:
            logger.error(f"Failed to get team game logs: {e}")
            return pd.DataFrame()

    def get_player_game_logs(
        self,
        season: int,
        player_id: Optional[int] = None,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        선수 경기 로그 조회.

        Args:
            season: 시즌 연도
            player_id: 선수 ID (None이면 전체 선수)
            season_type: 시즌 타입

        Returns:
            선수 경기 로그 DataFrame
        """
        season_str = get_nba_api_season_string(season)

        params = {
            "LeagueID": "00",
            "Season": season_str,
            "SeasonType": season_type,
        }

        if player_id:
            params["PlayerID"] = player_id

        try:
            data = self._make_request("playergamelogs", params)
            df = self._parse_response(data)

            return df

        except Exception as e:
            logger.error(f"Failed to get player game logs: {e}")
            return pd.DataFrame()

    def get_lineup_stats(
        self,
        season: int,
        team_id: int,
        group_quantity: int = 5,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        라인업 통계 조회.

        Args:
            season: 시즌 연도
            team_id: 팀 ID
            group_quantity: 라인업 인원 수 (2-5)
            season_type: 시즌 타입

        Returns:
            라인업 통계 DataFrame
        """
        season_str = get_nba_api_season_string(season)

        params = {
            "LeagueID": "00",
            "Season": season_str,
            "SeasonType": season_type,
            "TeamID": team_id,
            "GroupQuantity": group_quantity,
            "PerMode": "Totals",
            "MeasureType": "Advanced",
        }

        try:
            data = self._make_request("teamdashlineups", params)
            df = self._parse_response(data)

            return df

        except Exception as e:
            logger.error(f"Failed to get lineup stats for team {team_id}: {e}")
            return pd.DataFrame()

    def get_scoreboard(
        self,
        game_date: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        특정 날짜 스코어보드 조회.

        Args:
            game_date: 날짜 (YYYY-MM-DD)
            use_cache: 캐시 사용 여부 (라이브 경기는 False 권장)

        Returns:
            {"games": 경기 정보, "line_score": 라인 스코어}
        """
        # 날짜 형식 변환 (MM/DD/YYYY)
        dt = parse_date(game_date)
        date_str = dt.strftime("%m/%d/%Y")

        params = {
            "LeagueID": "00",
            "GameDate": date_str,
            "DayOffset": 0,
        }

        try:
            data = self._make_request("scoreboardv2", params, use_cache=use_cache)

            return {
                "games": self._parse_response(data, 0),  # GameHeader
                "line_score": self._parse_response(data, 1),  # LineScore
            }

        except Exception as e:
            logger.error(f"Failed to get scoreboard for {game_date}: {e}")
            return {"games": pd.DataFrame(), "line_score": pd.DataFrame()}

    # ===================
    # Batch Collection Methods
    # ===================

    async def collect_season_boxscores(
        self,
        season: int,
        game_ids: List[str],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        시즌 전체 박스스코어 수집.

        Args:
            season: 시즌 연도
            game_ids: 경기 ID 리스트
            progress_callback: 진행률 콜백

        Returns:
            게임 ID별 박스스코어 딕셔너리
        """
        results = {}
        total = len(game_ids)

        for i, game_id in enumerate(game_ids):
            try:
                boxscore = await self.get_boxscore_async(game_id)
                results[game_id] = boxscore

            except Exception as e:
                logger.error(f"Failed to get boxscore for {game_id}: {e}")
                results[game_id] = {"player": pd.DataFrame(), "team": pd.DataFrame()}

            if progress_callback:
                progress_callback((i + 1) / total)

            # Rate limiting
            await asyncio.sleep(self.config.delay_between_requests)

        logger.info(f"Collected {len(results)} boxscores for season {season}")
        return results

    # ===================
    # Data Normalization
    # ===================

    def _normalize_game_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """경기 데이터 컬럼 정규화"""
        column_mapping = {
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbr",
            "TEAM_NAME": "team_name",
            "MATCHUP": "matchup",
            "WL": "result",
            "PTS": "pts",
            "FGM": "fgm",
            "FGA": "fga",
            "FG_PCT": "fg_pct",
            "FG3M": "fg3m",
            "FG3A": "fg3a",
            "FG3_PCT": "fg3_pct",
            "FTM": "ftm",
            "FTA": "fta",
            "FT_PCT": "ft_pct",
            "OREB": "oreb",
            "DREB": "dreb",
            "REB": "reb",
            "AST": "ast",
            "STL": "stl",
            "BLK": "blk",
            "TOV": "tov",
            "PF": "pf",
            "PLUS_MINUS": "plus_minus",
        }

        # 존재하는 컬럼만 매핑
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        return df

    def close(self) -> None:
        """리소스 정리"""
        self._session.close()
        self._executor.shutdown(wait=False)


# ===================
# Factory Function
# ===================

def create_nba_stats_client(
    cache_dir: Optional[Path] = None,
    **kwargs
) -> NBAStatsClient:
    """
    NBA Stats 클라이언트 팩토리 함수.

    Args:
        cache_dir: 캐시 디렉토리
        **kwargs: 추가 설정

    Returns:
        NBAStatsClient 인스턴스
    """
    config = NBAStatsClientConfig(**kwargs)
    return NBAStatsClient(config, cache_dir)
