"""
BucketsVision V5.4 예측 서비스 모듈.

V5.4 모델: 5개 피처 + Logistic Regression (C=0.01)
- team_epm_diff: 팀 EPM 차이
- sos_diff: Strength of Schedule 차이 (신규)
- bench_strength_diff: 벤치 선수 EPM 차이
- top5_epm_diff: 상위 5인 EPM 차이
- ft_rate_diff: Free Throw Rate 차이

V5.4는 최적화된 5개 피처로 78.05% 정확도를 달성합니다.
확률 범위: 8% ~ 95% (압축 없음)
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

from app.services.injury_adjuster import InjuryAdjuster


@dataclass
class V5GamePrediction:
    """V5.4 경기 예측 결과"""
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
    """V5.4 예측 서비스"""

    # 부상 보정 최대 한도
    MAX_INJURY_SHIFT = 0.10  # ±10%p

    # 캘리브레이션 설정 (Under-confident 보정)
    CALIBRATION_ENABLED = True
    CALIBRATION_STRETCH_FACTOR = 1.15  # 50% 기준 확률 확장 계수

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
        model_path = self.model_dir / "v5_4_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

        # Scaler 로드
        scaler_path = self.model_dir / "v5_4_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # 피처명 로드
        feature_path = self.model_dir / "v5_4_feature_names.json"
        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_names = json.load(f)

        # 메타데이터 로드
        meta_path = self.model_dir / "v5_4_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        홈팀 승리 확률 예측 (기본, 부상 조정 전).

        Args:
            features: 피처 딕셔너리 (5개)

        Returns:
            홈팀 승리 확률 (0-1), 캘리브레이션 적용됨
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded")

        # 피처 순서 맞추기
        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)

        # 확률 예측
        raw_proba = self.model.predict_proba(X_scaled)[0, 1]

        # 캘리브레이션 적용 (Under-confident 보정)
        if self.CALIBRATION_ENABLED:
            return self._calibrate(raw_proba)

        return float(raw_proba)

    def _calibrate(self, prob: float) -> float:
        """
        Under-confident 보정을 위한 확률 스트레칭.

        원리: 50% 기준으로 확률을 양방향으로 확장
        - 60% → 61.5% (더 확신)
        - 40% → 38.5% (더 확신)
        - 50% → 50% (변화 없음)

        Args:
            prob: 원본 예측 확률 (0-1)

        Returns:
            보정된 확률 (1%-99% 범위로 클리핑)
        """
        calibrated = 0.5 + (prob - 0.5) * self.CALIBRATION_STRETCH_FACTOR
        return max(0.01, min(0.99, calibrated))

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

        Note:
            InjuryAdjuster로 로직 위임 (리팩토링 Phase 1).
        """
        return InjuryAdjuster.apply(base_prob, home_prob_shift, away_prob_shift)

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
            features: V5.4 피처 딕셔너리 (5개)
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
            "model_type": "V5.4 Logistic Regression (C=0.01)",
            "model_version": self.metadata.get("version", "5.4.0") if self.metadata else "5.4.0",
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "training_date": self.metadata.get("training_date") if self.metadata else None,
            "low_conf_accuracy": self.metadata.get("metrics", {}).get("low_conf_accuracy") if self.metadata else None,
            "high_conf_accuracy": self.metadata.get("metrics", {}).get("high_conf_accuracy") if self.metadata else None,
            "overall_accuracy": self.metadata.get("metrics", {}).get("overall_accuracy") if self.metadata else None,
            "trailing_indicators": ["injury_adjustment"],
            "prob_range": self.metadata.get("metrics", {}).get("prob_range") if self.metadata else [0.08, 0.95],
            "calibration_enabled": self.CALIBRATION_ENABLED,
            "calibration_factor": self.CALIBRATION_STRETCH_FACTOR if self.CALIBRATION_ENABLED else None,
        }
