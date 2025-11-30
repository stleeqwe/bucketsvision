"""
Rest and Fatigue Features.

휴식일, 이동거리, 일정 밀도 등 피로 관련 피처를 생성합니다.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math

import pandas as pd
import numpy as np

from src.features.base import (
    BaseFeature,
    FeatureContext,
    FeatureResult,
    safe_divide,
    compute_diff,
    clip_feature
)
from src.utils.logger import logger
from src.utils.helpers import parse_date, haversine_distance
from config.feature_config import NBA_TEAMS, TIMEZONE_OFFSETS


class RestFatigueFeature(BaseFeature):
    """
    휴식/피로 피처.

    생성 피처:
    - rest_days_home: 홈팀 휴식일 (1, 2, 3, 4+)
    - rest_days_away: 어웨이팀 휴식일
    - rest_diff: 휴식일 차이
    - back_to_back: B2B 상태 (0: 없음, 1: 어웨이만, 2: 홈만, 3: 양팀)
    - travel_km_away: 어웨이팀 이동거리 (km)
    - games_7d_diff: 7일간 경기 수 차이
    - timezone_shift_away: 어웨이팀 시간대 이동
    """

    def __init__(self, arena_coords: Optional[pd.DataFrame] = None):
        """
        Args:
            arena_coords: 경기장 좌표 DataFrame (team_id, lat, lon)
        """
        self.arena_coords = arena_coords

    @property
    def name(self) -> str:
        return "rest_fatigue"

    @property
    def feature_names(self) -> List[str]:
        return [
            "rest_days_home",
            "rest_days_away",
            "rest_diff",
            "back_to_back",
            "travel_km_away",
            "games_7d_diff",
            "timezone_shift_away"
        ]

    @property
    def required_context(self) -> List[str]:
        return ["games_history"]

    def compute(self, context: FeatureContext) -> FeatureResult:
        """휴식/피로 피처 계산"""
        features = {}
        warnings = []

        game_date = parse_date(context.game_date)

        # 휴식일 계산
        home_rest = self._calculate_rest_days(
            context, context.home_team_id, game_date
        )
        away_rest = self._calculate_rest_days(
            context, context.away_team_id, game_date
        )

        features["rest_days_home"] = clip_feature(home_rest, 0, 4)
        features["rest_days_away"] = clip_feature(away_rest, 0, 4)
        features["rest_diff"] = compute_diff(home_rest, away_rest)

        # Back-to-back 상태
        features["back_to_back"] = self._calculate_b2b_status(home_rest, away_rest)

        # 이동거리 (어웨이팀)
        travel_km = self._calculate_travel_distance(
            context, context.away_team_id, game_date
        )
        features["travel_km_away"] = clip_feature(travel_km, 0, 5000)

        # 7일간 경기 수 차이
        home_games_7d = self._count_games_in_period(
            context, context.home_team_id, game_date, days=7
        )
        away_games_7d = self._count_games_in_period(
            context, context.away_team_id, game_date, days=7
        )
        features["games_7d_diff"] = compute_diff(away_games_7d, home_games_7d)

        # 시간대 이동 (어웨이팀)
        tz_shift = self._calculate_timezone_shift(
            context, context.away_team_id, context.home_team_id
        )
        features["timezone_shift_away"] = clip_feature(tz_shift, -3, 3)

        return FeatureResult(
            features=features,
            is_valid=True,
            warnings=warnings
        )

    def _calculate_rest_days(
        self,
        context: FeatureContext,
        team_id: int,
        game_date
    ) -> int:
        """
        휴식일 계산.

        Args:
            context: 피처 컨텍스트
            team_id: 팀 ID
            game_date: 경기 날짜

        Returns:
            휴식일 (0=B2B, 1=1일 휴식, ..., 4+=4일 이상)
        """
        if context.games_history is None or context.games_history.empty:
            return 2  # 기본값

        # 해당 팀의 이전 경기 찾기
        team_games = context.games_history[
            context.games_history["team_id"] == team_id
        ].copy()

        if team_games.empty:
            return 2

        # 날짜 변환
        if "game_date" in team_games.columns:
            team_games["game_date"] = pd.to_datetime(team_games["game_date"])

        # 현재 경기 이전의 가장 최근 경기
        previous_games = team_games[
            team_games["game_date"] < pd.Timestamp(game_date)
        ].sort_values("game_date", ascending=False)

        if previous_games.empty:
            return 4  # 첫 경기는 충분한 휴식

        last_game_date = previous_games.iloc[0]["game_date"]

        if isinstance(last_game_date, str):
            last_game_date = parse_date(last_game_date)
        elif hasattr(last_game_date, 'date'):
            last_game_date = last_game_date.date()

        rest_days = (game_date - last_game_date).days - 1  # 경기 다음날부터 카운트

        return min(4, max(0, rest_days))

    def _calculate_b2b_status(self, home_rest: int, away_rest: int) -> int:
        """
        Back-to-back 상태 계산.

        Returns:
            0: 없음, 1: 어웨이만, 2: 홈만, 3: 양팀
        """
        home_b2b = home_rest == 0
        away_b2b = away_rest == 0

        if not home_b2b and not away_b2b:
            return 0
        elif away_b2b and not home_b2b:
            return 1
        elif home_b2b and not away_b2b:
            return 2
        else:
            return 3

    def _calculate_travel_distance(
        self,
        context: FeatureContext,
        team_id: int,
        game_date
    ) -> float:
        """
        어웨이팀 이동거리 계산.

        Args:
            context: 피처 컨텍스트
            team_id: 팀 ID
            game_date: 경기 날짜

        Returns:
            이동거리 (km)
        """
        if self.arena_coords is None or self.arena_coords.empty:
            return 0.0

        if context.games_history is None or context.games_history.empty:
            return 0.0

        # 이전 경기 위치 찾기
        team_games = context.games_history[
            context.games_history["team_id"] == team_id
        ].copy()

        if team_games.empty:
            return 0.0

        team_games["game_date"] = pd.to_datetime(team_games["game_date"])
        previous_games = team_games[
            team_games["game_date"] < pd.Timestamp(game_date)
        ].sort_values("game_date", ascending=False)

        if previous_games.empty:
            return 0.0

        # 이전 경기 위치 (홈/어웨이에 따라)
        last_game = previous_games.iloc[0]

        # 현재 경기장과 이전 경기장 좌표로 거리 계산
        # (실제 구현에서는 arena_coords에서 좌표를 조회)
        # 여기서는 간단한 추정값 반환
        return self._estimate_travel_distance(team_id, context.home_team_id)

    def _estimate_travel_distance(
        self,
        from_team_id: int,
        to_team_id: int
    ) -> float:
        """
        팀 간 대략적인 이동거리 추정.

        실제 구현에서는 경기장 좌표를 사용해야 합니다.
        """
        # 간단한 추정: 같은 시간대면 가까움
        from_tz = NBA_TEAMS.get(from_team_id, {}).get("timezone", "ET")
        to_tz = NBA_TEAMS.get(to_team_id, {}).get("timezone", "ET")

        from_offset = TIMEZONE_OFFSETS.get(from_tz, 0)
        to_offset = TIMEZONE_OFFSETS.get(to_tz, 0)

        tz_diff = abs(from_offset - to_offset)

        # 시간대 차이에 따른 대략적 거리 (km)
        distance_map = {0: 500, 1: 1500, 2: 2500, 3: 4000}
        return distance_map.get(tz_diff, 2000)

    def _count_games_in_period(
        self,
        context: FeatureContext,
        team_id: int,
        end_date,
        days: int = 7
    ) -> int:
        """
        특정 기간 내 경기 수 계산.

        Args:
            context: 피처 컨텍스트
            team_id: 팀 ID
            end_date: 종료 날짜
            days: 기간 (일)

        Returns:
            경기 수
        """
        if context.games_history is None or context.games_history.empty:
            return 0

        team_games = context.games_history[
            context.games_history["team_id"] == team_id
        ].copy()

        if team_games.empty:
            return 0

        team_games["game_date"] = pd.to_datetime(team_games["game_date"])

        start_date = pd.Timestamp(end_date) - timedelta(days=days)
        end_timestamp = pd.Timestamp(end_date)

        games_in_period = team_games[
            (team_games["game_date"] >= start_date) &
            (team_games["game_date"] < end_timestamp)
        ]

        return len(games_in_period)

    def _calculate_timezone_shift(
        self,
        context: FeatureContext,
        away_team_id: int,
        home_team_id: int
    ) -> int:
        """
        시간대 이동 계산.

        동쪽 → 서쪽: 양수 (시간을 "벌음")
        서쪽 → 동쪽: 음수 (시간을 "잃음")

        Returns:
            시간대 차이 (-3 ~ +3)
        """
        away_tz = NBA_TEAMS.get(away_team_id, {}).get("timezone", "ET")
        home_tz = NBA_TEAMS.get(home_team_id, {}).get("timezone", "ET")

        away_offset = TIMEZONE_OFFSETS.get(away_tz, 0)
        home_offset = TIMEZONE_OFFSETS.get(home_tz, 0)

        # 어웨이팀 입장에서 시간대 변화
        # 동→서 이동 (offset 감소) = 양수 (유리)
        return away_offset - home_offset


class ScheduleDensityFeature(BaseFeature):
    """
    일정 밀도 피처.

    향후 일정 부담을 고려한 피처.
    """

    @property
    def name(self) -> str:
        return "schedule_density"

    @property
    def feature_names(self) -> List[str]:
        return [
            "games_next_7d_diff",
            "avg_rest_5g_diff"
        ]

    @property
    def required_context(self) -> List[str]:
        return ["schedules"]

    def compute(self, context: FeatureContext) -> FeatureResult:
        """일정 밀도 피처 계산"""
        features = {}

        # 기본값 반환 (스케줄 데이터 없는 경우)
        features["games_next_7d_diff"] = 0.0
        features["avg_rest_5g_diff"] = 0.0

        return FeatureResult(
            features=features,
            is_valid=True,
            warnings=[]
        )
