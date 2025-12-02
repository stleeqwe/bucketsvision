"""
BucketsVision V5.2 예측 서비스 모듈.

V5.2 모델: 11개 피처 + XGBoost
- EPM 핵심 (5개): team_epm_diff, player_rotation_epm_diff, bench_strength_diff, top5_epm_diff
- Four Factors (3개): efg_pct_diff, ft_rate_diff, orb_pct_diff
- 모멘텀 (2개): last3_win_pct_diff, last5_win_pct_diff
- 피로도 (2개): b2b_diff, rest_days_diff (V5.2 신규)

V5.2는 B2B와 휴식일을 모델 피처로 통합하여 학습에 반영합니다.
부상 영향은 후행 지표(post-prediction adjustment)로 적용됩니다.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np


@dataclass
class V5GamePrediction:
    """V5.2 경기 예측 결과"""
    game_id: str
    game_date: str
    game_time: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_win_prob: float  # 기본 예측 확률
    injury_adjusted_prob: float  # 부상 조정 후 확률
    features: Dict[str, float]
    home_injury_shift: float = 0.0  # 홈팀 부상 영향 (%)
    away_injury_shift: float = 0.0  # 어웨이팀 부상 영향 (%)


class V5PredictionService:
    """V5.2 예측 서비스"""

    # 부상 보정 최대 한도
    MAX_INJURY_SHIFT = 0.10  # ±10%p

    def __init__(self, model_dir: Path):
        """
        Args:
            model_dir: V5 모델 디렉토리 (bucketsvision_v4/models)
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None

        self._load_model()

    def _load_model(self) -> None:
        """모델 및 메타데이터 로드"""
        # 모델 로드
        model_path = self.model_dir / "v5_2_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

        # Scaler 로드
        scaler_path = self.model_dir / "v5_2_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # 피처명 로드
        feature_path = self.model_dir / "v5_2_feature_names.json"
        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_names = json.load(f)

        # 메타데이터 로드
        meta_path = self.model_dir / "v5_2_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        홈팀 승리 확률 예측 (기본, 부상 조정 전).

        Args:
            features: 피처 딕셔너리 (11개)

        Returns:
            홈팀 승리 확률 (0-1)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded")

        # 피처 순서 맞추기
        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)

        # 확률 예측
        proba = self.model.predict_proba(X_scaled)[0, 1]

        return float(proba)

    def apply_injury_adjustment(
        self,
        base_prob: float,
        home_prob_shift: float,
        away_prob_shift: float
    ) -> float:
        """
        부상 영향력 보정 적용.

        Args:
            base_prob: 기본 예측 확률 (홈팀 승리)
            home_prob_shift: 홈팀 부상으로 인한 승률 감소 (% 단위, 양수)
            away_prob_shift: 원정팀 부상으로 인한 승률 감소 (% 단위, 양수)

        Returns:
            부상 보정된 확률

        공식:
            - 홈팀 부상 → 홈팀 승률 감소 → base_prob 감소
            - 원정팀 부상 → 원정팀 승률 감소 → base_prob 증가
            - 최종 보정 = (away_shift - home_shift) / 100
        """
        # % 단위를 소수로 변환 (3.0% → 0.03)
        home_shift = max(home_prob_shift, 0) / 100.0
        away_shift = max(away_prob_shift, 0) / 100.0

        # 부상 영향 차이 (양수 = 원정팀이 더 불리 = 홈팀 유리)
        net_shift = away_shift - home_shift

        if net_shift == 0:
            return base_prob

        # 최대 보정 한도 적용
        net_shift = max(-self.MAX_INJURY_SHIFT, min(self.MAX_INJURY_SHIFT, net_shift))

        # 확률 조정 (확률 경계 유지)
        adjusted_prob = base_prob + net_shift
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))

        return adjusted_prob

    def predict_game(
        self,
        game_id: str,
        game_date: str,
        game_time: str,
        home_team: str,
        away_team: str,
        home_team_id: int,
        away_team_id: int,
        features: Dict[str, float],
        home_prob_shift: float = 0.0,
        away_prob_shift: float = 0.0
    ) -> V5GamePrediction:
        """
        단일 경기 예측 (부상 조정 포함).

        Args:
            game_id: 경기 ID
            game_date: 경기 날짜
            game_time: 경기 시간
            home_team: 홈팀 약어
            away_team: 원정팀 약어
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            features: V5.2 피처 딕셔너리 (11개)
            home_prob_shift: 홈팀 부상 영향 (%, optional)
            away_prob_shift: 어웨이팀 부상 영향 (%, optional)

        Returns:
            V5GamePrediction
        """
        # 기본 예측
        base_prob = self.predict_proba(features)

        # 부상 조정 적용
        adjusted_prob = self.apply_injury_adjustment(
            base_prob, home_prob_shift, away_prob_shift
        )

        return V5GamePrediction(
            game_id=game_id,
            game_date=game_date,
            game_time=game_time,
            home_team=home_team,
            away_team=away_team,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_win_prob=round(base_prob, 3),
            injury_adjusted_prob=round(adjusted_prob, 3),
            features=features,
            home_injury_shift=home_prob_shift,
            away_injury_shift=away_prob_shift
        )

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_type": "V5.2 XGBoost + B2B + Rest Days",
            "model_version": self.metadata.get("version", "5.2.0") if self.metadata else "5.2.0",
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "training_date": self.metadata.get("training_date") if self.metadata else None,
            "low_conf_accuracy": self.metadata.get("metrics", {}).get("low_conf_accuracy") if self.metadata else None,
            "overall_accuracy": self.metadata.get("metrics", {}).get("overall_accuracy") if self.metadata else None,
            "trailing_indicators": ["injury_adjustment"],
            "model_features": ["b2b_diff", "rest_days_diff"],
        }
