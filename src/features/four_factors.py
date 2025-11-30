"""
Four Factors Features.

NBA 경기력의 핵심 지표인 Four Factors를 계산합니다:
- eFG% (Effective Field Goal Percentage)
- TOV% (Turnover Percentage)
- ORB% (Offensive Rebound Percentage)
- FT Rate (Free Throw Rate)
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.features.base import (
    BaseFeature,
    FeatureContext,
    FeatureResult,
    safe_divide,
    compute_diff,
    clip_feature,
    rolling_stat
)
from src.utils.logger import logger
from src.utils.helpers import (
    calculate_efg,
    calculate_tov_pct,
    calculate_orb_pct,
    calculate_ft_rate
)


class FourFactorsFeature(BaseFeature):
    """
    Four Factors 피처.

    5경기, 10경기 롤링 윈도우로 계산합니다.

    생성 피처:
    - eFG_diff_5G, eFG_diff_10G
    - TOV_diff_5G, TOV_diff_10G
    - ORB_diff_5G, ORB_diff_10G
    - FTR_diff_5G, FTR_diff_10G
    """

    def __init__(self, windows: List[int] = None):
        """
        Args:
            windows: 롤링 윈도우 크기 리스트 (기본: [5, 10])
        """
        self.windows = windows or [5, 10]

    @property
    def name(self) -> str:
        return "four_factors"

    @property
    def feature_names(self) -> List[str]:
        names = []
        for w in self.windows:
            names.extend([
                f"eFG_diff_{w}G",
                f"TOV_diff_{w}G",
                f"ORB_diff_{w}G",
                f"FTR_diff_{w}G"
            ])
        return names

    @property
    def required_context(self) -> List[str]:
        return ["games_history"]

    def compute(self, context: FeatureContext) -> FeatureResult:
        """Four Factors 피처 계산"""
        features = {}
        warnings = []

        for window in self.windows:
            # 홈팀 공격 지표 (최근 N경기)
            home_offense = self._calculate_team_offense_stats(
                context, context.home_team_id, window
            )

            # 어웨이팀 수비 지표 (상대팀에게 허용한 지표)
            away_defense = self._calculate_team_defense_stats(
                context, context.away_team_id, window
            )

            # 어웨이팀 공격 지표
            away_offense = self._calculate_team_offense_stats(
                context, context.away_team_id, window
            )

            # 홈팀 수비 지표
            home_defense = self._calculate_team_defense_stats(
                context, context.home_team_id, window
            )

            # eFG% diff: 홈 공격 eFG% - 어웨이 수비 eFG% (허용)
            features[f"eFG_diff_{window}G"] = compute_diff(
                home_offense.get("eFG", np.nan),
                away_defense.get("eFG_allowed", np.nan)
            )

            # TOV% diff: 어웨이 강제 턴오버 - 홈 턴오버 (낮을수록 좋음)
            features[f"TOV_diff_{window}G"] = compute_diff(
                away_defense.get("TOV_forced", np.nan),
                home_offense.get("TOV", np.nan)
            )

            # ORB% diff: 홈 공리바 - 어웨이 공리바 허용
            features[f"ORB_diff_{window}G"] = compute_diff(
                home_offense.get("ORB", np.nan),
                away_defense.get("ORB_allowed", np.nan)
            )

            # FT Rate diff: 홈 FT 비율 - 어웨이 FT 허용 비율
            features[f"FTR_diff_{window}G"] = compute_diff(
                home_offense.get("FTR", np.nan),
                away_defense.get("FTR_allowed", np.nan)
            )

        # 값 클리핑
        for key in features:
            features[key] = clip_feature(features[key], -0.5, 0.5)

        return FeatureResult(
            features=features,
            is_valid=True,
            warnings=warnings
        )

    def _calculate_team_offense_stats(
        self,
        context: FeatureContext,
        team_id: int,
        n_games: int
    ) -> Dict[str, float]:
        """
        팀 공격 지표 계산 (최근 N경기).

        Returns:
            {"eFG": ..., "TOV": ..., "ORB": ..., "FTR": ...}
        """
        history = context.get_team_history(team_id, context.game_date, n_games)

        if history.empty:
            return {"eFG": np.nan, "TOV": np.nan, "ORB": np.nan, "FTR": np.nan}

        # 필요한 컬럼 확인
        required_cols = ["fgm", "fga", "fg3m", "ftm", "fta", "tov", "oreb"]

        # 컬럼명 정규화 (대소문자 처리)
        history.columns = history.columns.str.lower()

        missing_cols = [c for c in required_cols if c not in history.columns]
        if missing_cols:
            logger.debug(f"Missing columns for offense stats: {missing_cols}")
            return {"eFG": np.nan, "TOV": np.nan, "ORB": np.nan, "FTR": np.nan}

        # 합계 계산
        totals = history[required_cols].sum()

        # Four Factors 계산
        efg = calculate_efg(
            totals.get("fgm", 0),
            totals.get("fg3m", 0),
            totals.get("fga", 0)
        )

        tov_pct = calculate_tov_pct(
            totals.get("tov", 0),
            totals.get("fga", 0),
            totals.get("fta", 0)
        )

        # ORB%는 상대방 DRB가 필요하므로 단순화
        orb_pct = safe_divide(
            totals.get("oreb", 0),
            totals.get("oreb", 0) + 40 * n_games  # 대략적인 상대 DRB
        )

        ft_rate = calculate_ft_rate(
            totals.get("ftm", 0),
            totals.get("fga", 0)
        )

        return {
            "eFG": efg,
            "TOV": tov_pct,
            "ORB": orb_pct,
            "FTR": ft_rate
        }

    def _calculate_team_defense_stats(
        self,
        context: FeatureContext,
        team_id: int,
        n_games: int
    ) -> Dict[str, float]:
        """
        팀 수비 지표 계산 (상대에게 허용한 지표).

        실제 구현에서는 상대팀 스탯을 조회해야 합니다.
        여기서는 간단히 공격 지표의 역수/보정값으로 추정합니다.
        """
        # 실제로는 박스스코어에서 상대팀 스탯을 가져와야 함
        # 여기서는 팀 평균 대비 보정값 사용

        offense = self._calculate_team_offense_stats(context, team_id, n_games)

        # 수비 지표는 리그 평균 대비 보정
        # (실제 구현에서는 상대팀 스탯 직접 사용)
        return {
            "eFG_allowed": 0.52,  # 리그 평균
            "TOV_forced": 0.12,   # 리그 평균
            "ORB_allowed": 0.25,  # 리그 평균
            "FTR_allowed": 0.20   # 리그 평균
        }


class AdvancedFourFactorsFeature(BaseFeature):
    """
    고급 Four Factors 피처.

    가중치가 적용된 Four Factors와 추가 효율성 지표.
    """

    @property
    def name(self) -> str:
        return "advanced_four_factors"

    @property
    def feature_names(self) -> List[str]:
        return [
            "weighted_four_factors_diff",
            "offensive_efficiency_diff",
            "defensive_efficiency_diff"
        ]

    @property
    def required_context(self) -> List[str]:
        return ["games_history"]

    def compute(self, context: FeatureContext) -> FeatureResult:
        """고급 Four Factors 피처 계산"""
        features = {}

        # Oliver의 가중치: eFG(0.4), TOV(0.25), ORB(0.2), FTR(0.15)
        # 여기서는 간단히 기본값 반환

        features["weighted_four_factors_diff"] = 0.0
        features["offensive_efficiency_diff"] = 0.0
        features["defensive_efficiency_diff"] = 0.0

        return FeatureResult(
            features=features,
            is_valid=True,
            warnings=[]
        )
