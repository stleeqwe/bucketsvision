"""
Team Strength Features using D&T EPM and SOS.

D&T API의 team-epm 데이터를 활용한 팀 강도 피처를 생성합니다.
"""

from typing import Dict, List, Optional

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


class TeamStrengthFeature(BaseFeature):
    """
    팀 강도 피처 (D&T EPM/SOS 기반).

    생성 피처:
    - team_epm_diff: 팀 EPM 차이
    - sos_diff: 상대강도 총합 차이
    - sos_o_diff: 공격 상대강도 차이
    - sos_d_diff: 수비 상대강도 차이
    - win_pct_10G_diff: 최근 10경기 승률 차이
    - home_record: 홈팀 홈경기 승률
    - away_record: 어웨이팀 원정 승률
    """

    @property
    def name(self) -> str:
        return "team_strength"

    @property
    def feature_names(self) -> List[str]:
        return [
            "team_epm_diff",
            "sos_diff",
            "sos_o_diff",
            "sos_d_diff",
            "win_pct_10G_diff",
            "home_record",
            "away_record"
        ]

    @property
    def required_context(self) -> List[str]:
        return ["team_epm"]

    def compute(self, context: FeatureContext) -> FeatureResult:
        """팀 강도 피처 계산"""
        features = {}
        warnings = []

        # D&T EPM 데이터 조회
        home_epm = context.get_team_epm_on_date(
            context.home_team_id,
            context.game_date
        )
        away_epm = context.get_team_epm_on_date(
            context.away_team_id,
            context.game_date
        )

        if home_epm is None:
            warnings.append(f"No EPM data for home team {context.home_team_id}")
            home_epm = {}

        if away_epm is None:
            warnings.append(f"No EPM data for away team {context.away_team_id}")
            away_epm = {}

        # EPM 차이 계산
        features["team_epm_diff"] = compute_diff(
            home_epm.get("team_epm", np.nan),
            away_epm.get("team_epm", np.nan)
        )

        # SOS 차이 계산
        features["sos_diff"] = compute_diff(
            home_epm.get("sos", np.nan),
            away_epm.get("sos", np.nan)
        )

        features["sos_o_diff"] = compute_diff(
            home_epm.get("sos_o", np.nan),
            away_epm.get("sos_o", np.nan)
        )

        features["sos_d_diff"] = compute_diff(
            home_epm.get("sos_d", np.nan),
            away_epm.get("sos_d", np.nan)
        )

        # 승률 계산 (경기 이력 필요)
        home_win_pct, away_win_pct = self._calculate_win_pct(context, n_games=10)
        features["win_pct_10G_diff"] = compute_diff(home_win_pct, away_win_pct)

        # 홈/원정 기록
        features["home_record"] = self._calculate_home_record(context)
        features["away_record"] = self._calculate_away_record(context)

        # 값 클리핑
        features["team_epm_diff"] = clip_feature(features["team_epm_diff"], -20, 20)
        features["sos_diff"] = clip_feature(features["sos_diff"], -5, 5)
        features["sos_o_diff"] = clip_feature(features["sos_o_diff"], -5, 5)
        features["sos_d_diff"] = clip_feature(features["sos_d_diff"], -5, 5)

        return FeatureResult(
            features=features,
            is_valid=len(warnings) == 0,
            warnings=warnings
        )

    def _calculate_win_pct(
        self,
        context: FeatureContext,
        n_games: int = 10
    ) -> tuple:
        """최근 N경기 승률 계산"""
        home_win_pct = np.nan
        away_win_pct = np.nan

        if context.games_history is not None and not context.games_history.empty:
            # 홈팀 승률
            home_history = context.get_team_history(
                context.home_team_id,
                context.game_date,
                n_games
            )
            if not home_history.empty and "result" in home_history.columns:
                wins = (home_history["result"] == "W").sum()
                home_win_pct = safe_divide(wins, len(home_history), 0.5)

            # 어웨이팀 승률
            away_history = context.get_team_history(
                context.away_team_id,
                context.game_date,
                n_games
            )
            if not away_history.empty and "result" in away_history.columns:
                wins = (away_history["result"] == "W").sum()
                away_win_pct = safe_divide(wins, len(away_history), 0.5)

        return home_win_pct, away_win_pct

    def _calculate_home_record(self, context: FeatureContext) -> float:
        """홈팀의 홈경기 승률"""
        if context.games_history is None or context.games_history.empty:
            return 0.5

        home_games = context.games_history[
            (context.games_history["team_id"] == context.home_team_id) &
            (context.games_history["game_date"] < context.game_date) &
            (context.games_history.get("is_home", pd.Series([True] * len(context.games_history))))
        ]

        if home_games.empty or "result" not in home_games.columns:
            return 0.5

        wins = (home_games["result"] == "W").sum()
        return safe_divide(wins, len(home_games), 0.5)

    def _calculate_away_record(self, context: FeatureContext) -> float:
        """어웨이팀의 원정 승률"""
        if context.games_history is None or context.games_history.empty:
            return 0.5

        away_games = context.games_history[
            (context.games_history["team_id"] == context.away_team_id) &
            (context.games_history["game_date"] < context.game_date) &
            (~context.games_history.get("is_home", pd.Series([True] * len(context.games_history))))
        ]

        if away_games.empty or "result" not in away_games.columns:
            return 0.5

        wins = (away_games["result"] == "W").sum()
        return safe_divide(wins, len(away_games), 0.5)


class AdvancedTeamStrengthFeature(BaseFeature):
    """
    고급 팀 강도 피처.

    D&T의 다양한 EPM 버전과 추가 메트릭을 활용합니다.
    """

    @property
    def name(self) -> str:
        return "advanced_team_strength"

    @property
    def feature_names(self) -> List[str]:
        return [
            "team_oepm_diff",
            "team_depm_diff",
            "team_epm_smoothed_diff",
            "team_epm_full_diff",
            "sos_matchup"
        ]

    @property
    def required_context(self) -> List[str]:
        return ["team_epm"]

    def compute(self, context: FeatureContext) -> FeatureResult:
        """고급 팀 강도 피처 계산"""
        features = {}
        warnings = []

        home_epm = context.get_team_epm_on_date(
            context.home_team_id,
            context.game_date
        )
        away_epm = context.get_team_epm_on_date(
            context.away_team_id,
            context.game_date
        )

        if home_epm is None or away_epm is None:
            return FeatureResult(
                features={n: np.nan for n in self.feature_names},
                is_valid=False,
                warnings=["Missing EPM data"]
            )

        # 공수 분리 EPM
        features["team_oepm_diff"] = compute_diff(
            home_epm.get("team_oepm", np.nan),
            away_epm.get("team_oepm", np.nan)
        )

        features["team_depm_diff"] = compute_diff(
            home_epm.get("team_depm", np.nan),
            away_epm.get("team_depm", np.nan)
        )

        # Smoothed/Full EPM (있는 경우)
        features["team_epm_smoothed_diff"] = compute_diff(
            home_epm.get("team_epm_smoothed", np.nan),
            away_epm.get("team_epm_smoothed", np.nan)
        )

        features["team_epm_full_diff"] = compute_diff(
            home_epm.get("team_epm_full", np.nan),
            away_epm.get("team_epm_full", np.nan)
        )

        # SOS 매치업: 홈 공격 상대강도 vs 어웨이 수비 상대강도
        features["sos_matchup"] = compute_diff(
            home_epm.get("sos_o", np.nan),
            away_epm.get("sos_d", np.nan)
        )

        return FeatureResult(
            features=features,
            is_valid=True,
            warnings=warnings
        )
