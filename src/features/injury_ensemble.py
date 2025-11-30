"""
Injury Impact Ensemble.

여러 분석 방법론의 결과를 앙상블하여
최종 부상 영향도를 계산합니다.

앙상블 구성요소:
1. On/Off 분석 (Adjusted)
2. Mixed Effects Regression
3. Bayesian Hierarchical Model
4. EPM 기반 추정 (Fallback)

가중치 결정:
- 데이터 품질 (표본 크기, 신뢰구간 폭)
- 방법론 신뢰도
- 통계적 유의성
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.injury_impact import InjuryImpactCalculator
from src.utils.logger import logger


@dataclass
class EnsembleImpact:
    """앙상블 영향도"""
    player_id: int
    player_name: str
    team_id: int
    team_abbr: str

    # 앙상블 결과
    ensemble_impact: float  # 최종 영향도
    ensemble_std: float  # 불확실성

    # 개별 방법론 결과
    on_off_impact: Optional[float]
    mixed_effects_impact: Optional[float]
    bayesian_impact: Optional[float]
    epm_impact: Optional[float]

    # 가중치
    weights: Dict[str, float]

    # 메타 정보
    sources_used: List[str]
    confidence: float  # 0-1


class InjuryImpactEnsemble:
    """
    부상 영향도 앙상블 계산기.

    여러 분석 방법론의 결과를 통합하여
    신뢰도 높은 최종 영향도를 계산합니다.
    """

    # 기본 가중치
    DEFAULT_WEIGHTS = {
        "on_off": 0.35,
        "mixed_effects": 0.25,
        "bayesian": 0.25,
        "epm": 0.15,
    }

    def __init__(
        self,
        on_off_df: Optional[pd.DataFrame] = None,
        mixed_effects_df: Optional[pd.DataFrame] = None,
        bayesian_df: Optional[pd.DataFrame] = None,
        epm_calculator: Optional[InjuryImpactCalculator] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            on_off_df: On/Off 분석 결과
            mixed_effects_df: Mixed Effects 결과
            bayesian_df: Bayesian 추정 결과
            epm_calculator: EPM 기반 계산기 (fallback)
            weights: 방법론별 가중치
        """
        self.on_off_df = on_off_df
        self.mixed_effects_df = mixed_effects_df
        self.bayesian_df = bayesian_df
        self.epm_calculator = epm_calculator

        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # 인덱스 구축
        self._build_indices()

        logger.info("InjuryImpactEnsemble initialized:")
        logger.info(f"  On/Off: {len(self.on_off_df) if self.on_off_df is not None else 0} players")
        logger.info(f"  Mixed Effects: {len(self.mixed_effects_df) if self.mixed_effects_df is not None else 0} players")
        logger.info(f"  Bayesian: {len(self.bayesian_df) if self.bayesian_df is not None else 0} players")
        logger.info(f"  EPM fallback: {'Yes' if self.epm_calculator else 'No'}")

    def _build_indices(self):
        """빠른 조회를 위한 인덱스 구축"""
        self._on_off_idx = {}
        self._mixed_idx = {}
        self._bayesian_idx = {}

        if self.on_off_df is not None and not self.on_off_df.empty:
            for _, row in self.on_off_df.iterrows():
                self._on_off_idx[row["player_id"]] = row.to_dict()

        if self.mixed_effects_df is not None and not self.mixed_effects_df.empty:
            for _, row in self.mixed_effects_df.iterrows():
                self._mixed_idx[row["player_id"]] = row.to_dict()

        if self.bayesian_df is not None and not self.bayesian_df.empty:
            for _, row in self.bayesian_df.iterrows():
                self._bayesian_idx[row["player_id"]] = row.to_dict()

    def get_player_ensemble(
        self,
        player_id: int,
        player_name: Optional[str] = None,
        team_abbr: Optional[str] = None
    ) -> EnsembleImpact:
        """
        선수별 앙상블 영향도 계산.

        Args:
            player_id: 선수 ID
            player_name: 선수 이름 (EPM fallback용)
            team_abbr: 팀 약어 (EPM fallback용)

        Returns:
            EnsembleImpact
        """
        impacts = {}
        stds = {}
        weights = {}
        sources_used = []

        # 1. On/Off 결과
        if player_id in self._on_off_idx:
            row = self._on_off_idx[player_id]
            # On/Off에서 adjusted_impact는 margin_on - margin_off
            # 부재 시 영향 = -adjusted_impact
            impact = -row.get("adjusted_impact", 0)
            impacts["on_off"] = impact
            stds["on_off"] = (row.get("ci_upper", 0) - row.get("ci_lower", 0)) / 3.92  # ~95% CI
            weights["on_off"] = self._calculate_weight_on_off(row)
            sources_used.append("on_off")

            if player_name is None:
                player_name = row.get("player_name", f"Player_{player_id}")
            if team_abbr is None:
                team_abbr = row.get("team_abbr", "")

        # 2. Mixed Effects 결과
        if player_id in self._mixed_idx:
            row = self._mixed_idx[player_id]
            impact = row.get("effect", 0)  # 이미 부재 시 효과로 정의됨
            impacts["mixed_effects"] = impact
            stds["mixed_effects"] = row.get("std_error", 0)
            weights["mixed_effects"] = self._calculate_weight_mixed(row)
            sources_used.append("mixed_effects")

            if player_name is None:
                player_name = row.get("player_name", f"Player_{player_id}")

        # 3. Bayesian 결과
        if player_id in self._bayesian_idx:
            row = self._bayesian_idx[player_id]
            impact = row.get("posterior_mean", 0)  # 부재 시 영향
            impacts["bayesian"] = impact
            stds["bayesian"] = row.get("posterior_std", 0)
            weights["bayesian"] = self._calculate_weight_bayesian(row)
            sources_used.append("bayesian")

            if player_name is None:
                player_name = row.get("player_name", f"Player_{player_id}")

        # 4. EPM Fallback
        if self.epm_calculator is not None and player_name:
            epm_impact = self.epm_calculator.calculate_player_impact(
                player_name, team_abbr or ""
            )
            if epm_impact != 0:
                # EPM 기반은 양수가 팀에 불리, 부호 변환 필요
                impacts["epm"] = -epm_impact
                stds["epm"] = abs(epm_impact) * 0.5  # 대략적 불확실성
                weights["epm"] = self.weights.get("epm", 0.15)
                sources_used.append("epm")

        # 가중 평균 계산
        if not impacts:
            return EnsembleImpact(
                player_id=player_id,
                player_name=player_name or f"Player_{player_id}",
                team_id=0,
                team_abbr=team_abbr or "",
                ensemble_impact=0.0,
                ensemble_std=5.0,  # 기본 불확실성
                on_off_impact=None,
                mixed_effects_impact=None,
                bayesian_impact=None,
                epm_impact=None,
                weights={},
                sources_used=[],
                confidence=0.0,
            )

        # 가중치 정규화
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # 가중 평균
        ensemble_impact = sum(
            impacts[k] * normalized_weights[k]
            for k in impacts.keys()
        )

        # 가중 표준편차 (분산의 가중 합의 제곱근)
        ensemble_var = sum(
            (stds[k] ** 2) * (normalized_weights[k] ** 2)
            for k in stds.keys()
        )
        ensemble_std = np.sqrt(ensemble_var)

        # 신뢰도 계산
        confidence = self._calculate_confidence(
            sources_used, weights, stds
        )

        return EnsembleImpact(
            player_id=player_id,
            player_name=player_name or f"Player_{player_id}",
            team_id=self._get_team_id(player_id),
            team_abbr=team_abbr or "",
            ensemble_impact=ensemble_impact,
            ensemble_std=ensemble_std,
            on_off_impact=impacts.get("on_off"),
            mixed_effects_impact=impacts.get("mixed_effects"),
            bayesian_impact=impacts.get("bayesian"),
            epm_impact=impacts.get("epm"),
            weights=normalized_weights,
            sources_used=sources_used,
            confidence=confidence,
        )

    def _calculate_weight_on_off(self, row: dict) -> float:
        """On/Off 가중치 계산"""
        base = self.weights.get("on_off", 0.35)

        # 표본 품질 보정
        quality = row.get("sample_quality", "low")
        if quality == "high":
            base *= 1.2
        elif quality == "low":
            base *= 0.7

        # 통계적 유의성 보정
        if row.get("is_significant", False):
            base *= 1.1

        return base

    def _calculate_weight_mixed(self, row: dict) -> float:
        """Mixed Effects 가중치 계산"""
        base = self.weights.get("mixed_effects", 0.25)

        # p-value 기반 보정
        p_value = row.get("p_value", 1.0)
        if p_value < 0.01:
            base *= 1.3
        elif p_value < 0.05:
            base *= 1.1
        elif p_value > 0.1:
            base *= 0.8

        return base

    def _calculate_weight_bayesian(self, row: dict) -> float:
        """Bayesian 가중치 계산"""
        base = self.weights.get("bayesian", 0.25)

        # 불확실성 감소 기반 보정
        uncertainty_reduction = row.get("uncertainty_reduction", 0)
        base *= (1 + uncertainty_reduction * 0.3)

        # 데이터 양 기반 보정
        n_games = row.get("n_games_on", 0) + row.get("n_games_off", 0)
        if n_games >= 50:
            base *= 1.1
        elif n_games < 20:
            base *= 0.9

        return base

    def _calculate_confidence(
        self,
        sources: List[str],
        weights: Dict[str, float],
        stds: Dict[str, float]
    ) -> float:
        """신뢰도 계산 (0-1)"""
        if not sources:
            return 0.0

        # 소스 다양성 (여러 방법론 합의)
        diversity_score = min(len(sources) / 3, 1.0) * 0.4

        # 데이터 기반 소스 비율
        data_sources = [s for s in sources if s in ["on_off", "mixed_effects", "bayesian"]]
        data_ratio = len(data_sources) / len(sources) * 0.3

        # 평균 불확실성 (낮을수록 좋음)
        if stds:
            avg_std = np.mean(list(stds.values()))
            uncertainty_score = max(0, 1 - avg_std / 10) * 0.3
        else:
            uncertainty_score = 0.0

        return diversity_score + data_ratio + uncertainty_score

    def _get_team_id(self, player_id: int) -> int:
        """선수의 팀 ID 조회"""
        for idx in [self._on_off_idx, self._mixed_idx, self._bayesian_idx]:
            if player_id in idx:
                return idx[player_id].get("team_id", 0)
        return 0

    def calculate_game_adjustment(
        self,
        home_out_players: List[Tuple[int, str, str]],  # [(player_id, name, team_abbr), ...]
        away_out_players: List[Tuple[int, str, str]]
    ) -> Tuple[float, float, Dict]:
        """
        경기별 부상 조정값 계산.

        Args:
            home_out_players: 홈팀 결장 선수 [(id, name, team_abbr), ...]
            away_out_players: 원정팀 결장 선수

        Returns:
            (adjustment, uncertainty, details)
        """
        home_impacts = []
        away_impacts = []
        details = {"home": [], "away": []}

        for player_id, name, team_abbr in home_out_players:
            ensemble = self.get_player_ensemble(player_id, name, team_abbr)
            home_impacts.append((ensemble.ensemble_impact, ensemble.ensemble_std))
            details["home"].append({
                "player_name": ensemble.player_name,
                "impact": ensemble.ensemble_impact,
                "confidence": ensemble.confidence,
                "sources": ensemble.sources_used,
            })

        for player_id, name, team_abbr in away_out_players:
            ensemble = self.get_player_ensemble(player_id, name, team_abbr)
            away_impacts.append((ensemble.ensemble_impact, ensemble.ensemble_std))
            details["away"].append({
                "player_name": ensemble.player_name,
                "impact": ensemble.ensemble_impact,
                "confidence": ensemble.confidence,
                "sources": ensemble.sources_used,
            })

        # 총 영향 (부재 시 영향이 음수 = 팀에 불리)
        home_total = sum(imp for imp, _ in home_impacts)
        away_total = sum(imp for imp, _ in away_impacts)

        # 홈팀 주전이 빠지면(home_total < 0) 마진 감소
        # 원정팀 주전이 빠지면(away_total < 0) 마진 증가
        adjustment = home_total - away_total

        # 불확실성 (분산 합)
        home_var = sum(std ** 2 for _, std in home_impacts)
        away_var = sum(std ** 2 for _, std in away_impacts)
        uncertainty = np.sqrt(home_var + away_var)

        return adjustment, uncertainty, details

    def get_all_ensemble_impacts(self) -> pd.DataFrame:
        """전체 선수 앙상블 결과"""
        all_player_ids = set()

        if self.on_off_df is not None:
            all_player_ids.update(self.on_off_df["player_id"].unique())
        if self.mixed_effects_df is not None:
            all_player_ids.update(self.mixed_effects_df["player_id"].unique())
        if self.bayesian_df is not None:
            all_player_ids.update(self.bayesian_df["player_id"].unique())

        results = []
        for player_id in all_player_ids:
            ensemble = self.get_player_ensemble(player_id)
            results.append({
                "player_id": ensemble.player_id,
                "player_name": ensemble.player_name,
                "team_id": ensemble.team_id,
                "team_abbr": ensemble.team_abbr,
                "ensemble_impact": ensemble.ensemble_impact,
                "ensemble_std": ensemble.ensemble_std,
                "on_off_impact": ensemble.on_off_impact,
                "mixed_effects_impact": ensemble.mixed_effects_impact,
                "bayesian_impact": ensemble.bayesian_impact,
                "epm_impact": ensemble.epm_impact,
                "confidence": ensemble.confidence,
                "sources_count": len(ensemble.sources_used),
            })

        df = pd.DataFrame(results)
        return df.sort_values("ensemble_impact")

    @classmethod
    def load(
        cls,
        data_dir: Path,
        season: int
    ) -> "InjuryImpactEnsemble":
        """
        저장된 데이터에서 앙상블 로드.

        Args:
            data_dir: 데이터 디렉토리
            season: 시즌 연도

        Returns:
            InjuryImpactEnsemble 인스턴스
        """
        # On/Off 결과
        on_off_path = data_dir / "processed" / "player_impact" / f"season_{season}.parquet"
        on_off_df = pd.read_parquet(on_off_path) if on_off_path.exists() else None

        # Mixed Effects 결과
        mixed_path = data_dir / "processed" / "mixed_effects" / f"season_{season}.parquet"
        mixed_df = pd.read_parquet(mixed_path) if mixed_path.exists() else None

        # Bayesian 결과
        bayesian_path = data_dir / "processed" / "bayesian_impact" / f"season_{season}.parquet"
        bayesian_df = pd.read_parquet(bayesian_path) if bayesian_path.exists() else None

        # EPM Calculator
        from src.features.injury_impact import load_player_epm, InjuryImpactCalculator
        try:
            player_epm = load_player_epm(data_dir, season)
            epm_calc = InjuryImpactCalculator(player_epm)
        except FileNotFoundError:
            epm_calc = None

        return cls(
            on_off_df=on_off_df,
            mixed_effects_df=mixed_df,
            bayesian_df=bayesian_df,
            epm_calculator=epm_calc
        )

    def save_ensemble(self, output_path: Path):
        """앙상블 결과 저장"""
        df = self.get_all_ensemble_impacts()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved ensemble results to {output_path}")
