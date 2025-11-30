"""
Lineup Features.

스타터 라인업, 벤치 유닛, 베스트 듀오 등 조합 관련 피처를 생성합니다.
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


class LineupFeature(BaseFeature):
    """
    라인업 피처.

    생성 피처:
    - starter_netrtg_diff: 스타터 5인 Net Rating 차이
    - bench_netrtg_diff: 벤치 유닛 Net Rating 차이
    - depth_diff: 8인 로테이션 깊이 차이
    - best_duo_diff: 베스트 2인조 Net Rating 차이
    """

    @property
    def name(self) -> str:
        return "lineup"

    @property
    def feature_names(self) -> List[str]:
        return [
            "starter_netrtg_diff",
            "bench_netrtg_diff",
            "depth_diff",
            "best_duo_diff"
        ]

    @property
    def required_context(self) -> List[str]:
        # 라인업 데이터가 없으면 선수 EPM으로 대체
        return ["player_epm"]

    def compute(self, context: FeatureContext) -> FeatureResult:
        """라인업 피처 계산"""
        features = {}
        warnings = []

        # 스타터 Net Rating (선수 EPM 합으로 대체)
        home_starter = self._estimate_starter_netrtg(
            context.player_epm, context.home_team_id
        )
        away_starter = self._estimate_starter_netrtg(
            context.player_epm, context.away_team_id
        )
        features["starter_netrtg_diff"] = compute_diff(home_starter, away_starter)

        # 벤치 Net Rating
        home_bench = self._estimate_bench_netrtg(
            context.player_epm, context.home_team_id
        )
        away_bench = self._estimate_bench_netrtg(
            context.player_epm, context.away_team_id
        )
        features["bench_netrtg_diff"] = compute_diff(home_bench, away_bench)

        # 깊이 (8인 평균 EPM)
        home_depth = self._calculate_depth(
            context.player_epm, context.home_team_id
        )
        away_depth = self._calculate_depth(
            context.player_epm, context.away_team_id
        )
        features["depth_diff"] = compute_diff(home_depth, away_depth)

        # 베스트 듀오 (상위 2인 합)
        home_duo = self._estimate_best_duo(
            context.player_epm, context.home_team_id
        )
        away_duo = self._estimate_best_duo(
            context.player_epm, context.away_team_id
        )
        features["best_duo_diff"] = compute_diff(home_duo, away_duo)

        # 값 클리핑
        features["starter_netrtg_diff"] = clip_feature(
            features["starter_netrtg_diff"], -50, 50
        )
        features["bench_netrtg_diff"] = clip_feature(
            features["bench_netrtg_diff"], -50, 50
        )
        features["depth_diff"] = clip_feature(
            features["depth_diff"], -10, 10
        )
        features["best_duo_diff"] = clip_feature(
            features["best_duo_diff"], -50, 50
        )

        return FeatureResult(
            features=features,
            is_valid=True,
            warnings=warnings
        )

    def _estimate_starter_netrtg(
        self,
        player_epm: Optional[pd.DataFrame],
        team_id: int
    ) -> float:
        """
        스타터 Net Rating 추정.

        실제 라인업 데이터가 없으면 상위 5인 EPM 합으로 대체.
        """
        if player_epm is None or player_epm.empty:
            return np.nan

        team_players = player_epm[player_epm["team_id"] == team_id]

        if team_players.empty:
            return np.nan

        # 출전시간 기준 상위 5명
        mp_col = "p_mp_48" if "p_mp_48" in team_players.columns else "mpg"
        epm_col = "epm" if "epm" in team_players.columns else "tot"

        if mp_col not in team_players.columns or epm_col not in team_players.columns:
            return np.nan

        starters = team_players.nlargest(5, mp_col)

        # EPM 합 * 5를 Net Rating으로 근사
        # (EPM은 100 포제션당 +/- 기여도이므로)
        return starters[epm_col].sum() * 5

    def _estimate_bench_netrtg(
        self,
        player_epm: Optional[pd.DataFrame],
        team_id: int
    ) -> float:
        """
        벤치 유닛 Net Rating 추정.

        6-8번째 선수의 EPM 합.
        """
        if player_epm is None or player_epm.empty:
            return np.nan

        team_players = player_epm[player_epm["team_id"] == team_id]

        if len(team_players) < 8:
            return np.nan

        mp_col = "p_mp_48" if "p_mp_48" in team_players.columns else "mpg"
        epm_col = "epm" if "epm" in team_players.columns else "tot"

        if mp_col not in team_players.columns or epm_col not in team_players.columns:
            return np.nan

        sorted_players = team_players.nlargest(8, mp_col)
        bench = sorted_players.iloc[5:8]  # 6-8번째

        return bench[epm_col].sum() * 5

    def _calculate_depth(
        self,
        player_epm: Optional[pd.DataFrame],
        team_id: int
    ) -> float:
        """8인 로테이션 평균 EPM"""
        if player_epm is None or player_epm.empty:
            return np.nan

        team_players = player_epm[player_epm["team_id"] == team_id]

        if len(team_players) < 8:
            return np.nan

        mp_col = "p_mp_48" if "p_mp_48" in team_players.columns else "mpg"
        epm_col = "epm" if "epm" in team_players.columns else "tot"

        if mp_col not in team_players.columns or epm_col not in team_players.columns:
            return np.nan

        top8 = team_players.nlargest(8, mp_col)
        return top8[epm_col].mean()

    def _estimate_best_duo(
        self,
        player_epm: Optional[pd.DataFrame],
        team_id: int
    ) -> float:
        """
        베스트 듀오 Net Rating 추정.

        상위 2인의 EPM 합 * 스케일링.
        """
        if player_epm is None or player_epm.empty:
            return np.nan

        team_players = player_epm[player_epm["team_id"] == team_id]

        if len(team_players) < 2:
            return np.nan

        mp_col = "p_mp_48" if "p_mp_48" in team_players.columns else "mpg"
        epm_col = "epm" if "epm" in team_players.columns else "tot"

        if mp_col not in team_players.columns or epm_col not in team_players.columns:
            return np.nan

        # 출전시간 기준 상위 2명
        top2 = team_players.nlargest(2, mp_col)

        # 시너지 보정 (실제로는 2인조 Net Rating 사용해야 함)
        return top2[epm_col].sum() * 3
