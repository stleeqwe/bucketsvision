"""
BucketsVision V4.3 예측 서비스 모듈.

V4.3 모델: 13개 피처 + Logistic Regression (C=0.001) + Isotonic Calibration
- EPM 핵심 (4개): team_epm_diff, team_oepm_diff, team_depm_diff, sos_diff
- Four Factors (2개): efg_pct_diff, ft_rate_diff
- 모멘텀 (3개): last5_win_pct_diff, streak_diff, margin_ewma_diff
- 컨텍스트 (1개): away_road_strength
- 리바운드 (1개): orb_diff
- 선수 EPM (2개): player_rotation_epm_diff, bench_strength_diff (V4.3 신규)

V4.3은 선수 개별 EPM을 활용하여 로스터 깊이를 반영합니다.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd


@dataclass
class V4GamePrediction:
    """V4 경기 예측 결과"""
    game_id: str
    game_date: str
    game_time: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_win_prob: float
    features: Dict[str, float]


class V4PredictionService:
    """V4.3 예측 서비스"""

    def __init__(self, model_dir: Path, version: str = "4.3"):
        """
        Args:
            model_dir: V4 모델 디렉토리 (bucketsvision_v4/models)
            version: 모델 버전 ("4.3" 또는 "4.2")
        """
        self.model_dir = model_dir
        self.version = version
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None

        self._load_model()

    def _load_model(self) -> None:
        """모델 및 메타데이터 로드"""
        # 버전에 따른 파일명 결정
        if self.version == "4.3":
            prefix = "v4_3"
        else:
            prefix = "v4"

        # Calibrated 모델 로드
        model_path = self.model_dir / f"{prefix}_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

        # Scaler 로드
        scaler_path = self.model_dir / f"{prefix}_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # 피처명 로드
        feature_path = self.model_dir / f"{prefix}_feature_names.json"
        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_names = json.load(f)

        # 메타데이터 로드
        meta_path = self.model_dir / f"{prefix}_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        홈팀 승리 확률 예측.

        CalibratedClassifierCV를 사용하여 직접 확률을 출력합니다.

        Args:
            features: 피처 딕셔너리 (V4.3: 13개, V4.2: 11개)

        Returns:
            홈팀 승리 확률 (0-1)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded")

        # 피처 순서 맞추기
        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)

        # 확률 예측 (CalibratedClassifierCV는 직접 확률 출력)
        proba = self.model.predict_proba(X_scaled)[0, 1]

        return float(proba)

    def predict_game(
        self,
        game_id: str,
        game_date: str,
        game_time: str,
        home_team: str,
        away_team: str,
        home_team_id: int,
        away_team_id: int,
        features: Dict[str, float]
    ) -> V4GamePrediction:
        """
        단일 경기 예측.

        Args:
            game_id: 경기 ID
            game_date: 경기 날짜
            game_time: 경기 시간
            home_team: 홈팀 약어
            away_team: 원정팀 약어
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            features: V4 피처 딕셔너리

        Returns:
            V4GamePrediction
        """
        win_prob = self.predict_proba(features)

        return V4GamePrediction(
            game_id=game_id,
            game_date=game_date,
            game_time=game_time,
            home_team=home_team,
            away_team=away_team,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_win_prob=round(win_prob, 3),
            features=features
        )

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        model_type = "V4.3 Logistic + Player EPM + Isotonic Calibration" if self.version == "4.3" else "V4.2 Logistic + Isotonic Calibration"
        return {
            "model_type": model_type,
            "model_version": self.metadata.get("version", f"{self.version}.0") if self.metadata else f"{self.version}.0",
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "training_date": self.metadata.get("training_date") if self.metadata else None,
            "accuracy": self.metadata.get("metrics", {}).get("accuracy") if self.metadata else None,
            "brier_score": self.metadata.get("metrics", {}).get("brier_score") if self.metadata else None,
            "auc_roc": self.metadata.get("metrics", {}).get("auc_roc") if self.metadata else None,
        }
