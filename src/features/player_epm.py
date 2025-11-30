"""
Player EPM Features using D&T API.

D&T API의 선수별 EPM 데이터를 활용한 피처를 생성합니다.
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


class PlayerEPMFeature(BaseFeature):
    """
    선수 EPM 피처.

    생성 피처:
    - top5_epm_diff: 상위 5명 선수 EPM 차이 (예상 출전시간 가중)
    - top8_epm_diff: 8인 로테이션 EPM 차이
    - bench_epm_diff: 벤치 선수(6-10번째) EPM 차이
    """

    @property
    def name(self) -> str:
        return "player_epm"

    @property
    def feature_names(self) -> List[str]:
        return [
            "top5_epm_diff",
            "top8_epm_diff",
            "bench_epm_diff"
        ]

    @property
    def required_context(self) -> List[str]:
        return ["player_epm"]

    def compute(self, context: FeatureContext) -> FeatureResult:
        """선수 EPM 피처 계산"""
        features = {}
        warnings = []

        if context.player_epm is None or context.player_epm.empty:
            return FeatureResult(
                features={n: np.nan for n in self.feature_names},
                is_valid=False,
                warnings=["No player EPM data available"]
            )

        # 홈팀 선수 EPM 계산
        home_top5 = self._get_team_weighted_epm(
            context.player_epm,
            context.home_team_id,
            top_n=5
        )
        home_top8 = self._get_team_weighted_epm(
            context.player_epm,
            context.home_team_id,
            top_n=8
        )
        home_bench = self._get_bench_epm(
            context.player_epm,
            context.home_team_id
        )

        # 어웨이팀 선수 EPM 계산
        away_top5 = self._get_team_weighted_epm(
            context.player_epm,
            context.away_team_id,
            top_n=5
        )
        away_top8 = self._get_team_weighted_epm(
            context.player_epm,
            context.away_team_id,
            top_n=8
        )
        away_bench = self._get_bench_epm(
            context.player_epm,
            context.away_team_id
        )

        # 차이 계산
        features["top5_epm_diff"] = compute_diff(home_top5, away_top5)
        features["top8_epm_diff"] = compute_diff(home_top8, away_top8)
        features["bench_epm_diff"] = compute_diff(home_bench, away_bench)

        # 값 클리핑
        features["top5_epm_diff"] = clip_feature(features["top5_epm_diff"], -30, 30)
        features["top8_epm_diff"] = clip_feature(features["top8_epm_diff"], -40, 40)
        features["bench_epm_diff"] = clip_feature(features["bench_epm_diff"], -15, 15)

        return FeatureResult(
            features=features,
            is_valid=True,
            warnings=warnings
        )

    def _get_team_weighted_epm(
        self,
        player_epm: pd.DataFrame,
        team_id: int,
        top_n: int = 5
    ) -> float:
        """
        팀의 상위 N명 선수 가중 EPM 계산.

        가중치: 예상 출전시간 비율 (p_mp_48 / 48)

        Args:
            player_epm: 선수 EPM DataFrame
            team_id: 팀 ID
            top_n: 상위 N명

        Returns:
            가중 EPM 합계
        """
        team_players = player_epm[player_epm["team_id"] == team_id].copy()

        if team_players.empty:
            return np.nan

        # 예상 출전시간 기준 정렬
        mp_col = "p_mp_48" if "p_mp_48" in team_players.columns else "mpg"

        if mp_col not in team_players.columns:
            # 출전시간 컬럼이 없으면 단순 평균
            return team_players.nlargest(top_n, "epm")["epm"].sum()

        top_players = team_players.nlargest(top_n, mp_col)

        # 가중 EPM: EPM * (출전시간 / 48)
        if "epm" not in top_players.columns:
            if "tot" in top_players.columns:
                epm_col = "tot"
            else:
                return np.nan
        else:
            epm_col = "epm"

        weighted_epm = (
            top_players[epm_col] *
            top_players[mp_col] / 48.0
        ).sum()

        return weighted_epm

    def _get_bench_epm(
        self,
        player_epm: pd.DataFrame,
        team_id: int
    ) -> float:
        """
        벤치 선수 (6-10번째) EPM 계산.

        Args:
            player_epm: 선수 EPM DataFrame
            team_id: 팀 ID

        Returns:
            벤치 가중 EPM 합계
        """
        team_players = player_epm[player_epm["team_id"] == team_id].copy()

        if len(team_players) < 6:
            return np.nan

        mp_col = "p_mp_48" if "p_mp_48" in team_players.columns else "mpg"

        if mp_col not in team_players.columns:
            return np.nan

        # 6-10번째 선수
        sorted_players = team_players.nlargest(10, mp_col)
        bench_players = sorted_players.iloc[5:10] if len(sorted_players) > 5 else pd.DataFrame()

        if bench_players.empty:
            return np.nan

        epm_col = "epm" if "epm" in bench_players.columns else "tot"
        if epm_col not in bench_players.columns:
            return np.nan

        weighted_epm = (
            bench_players[epm_col] *
            bench_players[mp_col] / 48.0
        ).sum()

        return weighted_epm


class DepthFeature(BaseFeature):
    """
    로스터 깊이 피처.

    팀의 로테이션 깊이를 평가합니다.
    """

    @property
    def name(self) -> str:
        return "depth"

    @property
    def feature_names(self) -> List[str]:
        return [
            "depth_diff",
            "rotation_size_diff",
            "star_reliance_diff"
        ]

    @property
    def required_context(self) -> List[str]:
        return ["player_epm"]

    def compute(self, context: FeatureContext) -> FeatureResult:
        """깊이 피처 계산"""
        features = {}
        warnings = []

        if context.player_epm is None or context.player_epm.empty:
            return FeatureResult(
                features={n: np.nan for n in self.feature_names},
                is_valid=False,
                warnings=["No player EPM data"]
            )

        # 8인 평균 EPM
        home_depth = self._calculate_depth(context.player_epm, context.home_team_id)
        away_depth = self._calculate_depth(context.player_epm, context.away_team_id)
        features["depth_diff"] = compute_diff(home_depth, away_depth)

        # 로테이션 크기 (15분 이상 출전 예상 선수 수)
        home_rotation = self._count_rotation_players(
            context.player_epm, context.home_team_id
        )
        away_rotation = self._count_rotation_players(
            context.player_epm, context.away_team_id
        )
        features["rotation_size_diff"] = compute_diff(home_rotation, away_rotation)

        # 스타 의존도 (Top 2 선수의 EPM 비중)
        home_reliance = self._calculate_star_reliance(
            context.player_epm, context.home_team_id
        )
        away_reliance = self._calculate_star_reliance(
            context.player_epm, context.away_team_id
        )
        features["star_reliance_diff"] = compute_diff(home_reliance, away_reliance)

        return FeatureResult(
            features=features,
            is_valid=True,
            warnings=warnings
        )

    def _calculate_depth(
        self,
        player_epm: pd.DataFrame,
        team_id: int
    ) -> float:
        """8인 로테이션 평균 EPM"""
        team_players = player_epm[player_epm["team_id"] == team_id]

        if len(team_players) < 8:
            return np.nan

        mp_col = "p_mp_48" if "p_mp_48" in team_players.columns else "mpg"
        epm_col = "epm" if "epm" in team_players.columns else "tot"

        if mp_col not in team_players.columns or epm_col not in team_players.columns:
            return np.nan

        top8 = team_players.nlargest(8, mp_col)
        return top8[epm_col].mean()

    def _count_rotation_players(
        self,
        player_epm: pd.DataFrame,
        team_id: int,
        min_minutes: float = 15.0
    ) -> int:
        """15분 이상 출전 예상 선수 수"""
        team_players = player_epm[player_epm["team_id"] == team_id]
        mp_col = "p_mp_48" if "p_mp_48" in team_players.columns else "mpg"

        if mp_col not in team_players.columns:
            return 8  # 기본값

        return (team_players[mp_col] >= min_minutes).sum()

    def _calculate_star_reliance(
        self,
        player_epm: pd.DataFrame,
        team_id: int
    ) -> float:
        """Top 2 선수의 EPM 비중"""
        team_players = player_epm[player_epm["team_id"] == team_id]

        if len(team_players) < 2:
            return np.nan

        epm_col = "epm" if "epm" in team_players.columns else "tot"
        mp_col = "p_mp_48" if "p_mp_48" in team_players.columns else "mpg"

        if epm_col not in team_players.columns:
            return np.nan

        # 출전시간 기준 상위 8명
        top8 = team_players.nlargest(8, mp_col)
        top2 = top8.nlargest(2, epm_col)

        total_epm = top8[epm_col].sum()
        top2_epm = top2[epm_col].sum()

        return safe_divide(top2_epm, total_epm, 0.5)
