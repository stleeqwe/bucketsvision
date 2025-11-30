"""
Evaluation Metrics.

NBA 점수 예측 모델의 성능 평가 지표를 계산합니다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

from src.utils.logger import logger


@dataclass
class MetricsReport:
    """
    종합 성능 보고서.

    회귀 메트릭, 분류 메트릭(승패 예측), 베팅 관련 메트릭을 포함합니다.
    """
    # 기본 회귀 메트릭
    rmse: float
    mae: float
    mse: float
    r2: float
    explained_variance: float

    # 분류 메트릭 (승패 예측)
    win_accuracy: float
    win_precision: float
    win_recall: float

    # 범위 내 예측 정확도
    within_3_accuracy: float
    within_5_accuracy: float
    within_7_accuracy: float
    within_10_accuracy: float

    # 오차 분포
    mean_error: float  # Bias
    std_error: float
    median_error: float
    error_percentiles: Dict[str, float] = field(default_factory=dict)

    # 추가 메트릭
    n_samples: int = 0
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mse": self.mse,
            "r2": self.r2,
            "explained_variance": self.explained_variance,
            "win_accuracy": self.win_accuracy,
            "win_precision": self.win_precision,
            "win_recall": self.win_recall,
            "within_3_accuracy": self.within_3_accuracy,
            "within_5_accuracy": self.within_5_accuracy,
            "within_7_accuracy": self.within_7_accuracy,
            "within_10_accuracy": self.within_10_accuracy,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "median_error": self.median_error,
            "error_percentiles": self.error_percentiles,
            "n_samples": self.n_samples,
            "model_name": self.model_name
        }

    def summary(self) -> str:
        """요약 문자열"""
        return (
            f"RMSE: {self.rmse:.3f}, MAE: {self.mae:.3f}, "
            f"Win Acc: {self.win_accuracy:.1%}, "
            f"Within 5: {self.within_5_accuracy:.1%}"
        )

    def detailed_summary(self) -> str:
        """상세 요약 문자열"""
        lines = [
            f"{'='*50}",
            f"Model: {self.model_name}" if self.model_name else "",
            f"N Samples: {self.n_samples}",
            f"{'='*50}",
            f"",
            f"Regression Metrics:",
            f"  RMSE: {self.rmse:.4f}",
            f"  MAE:  {self.mae:.4f}",
            f"  R²:   {self.r2:.4f}",
            f"",
            f"Classification Metrics (Win/Loss):",
            f"  Accuracy:  {self.win_accuracy:.2%}",
            f"  Precision: {self.win_precision:.2%}",
            f"  Recall:    {self.win_recall:.2%}",
            f"",
            f"Prediction Accuracy:",
            f"  Within 3 pts: {self.within_3_accuracy:.2%}",
            f"  Within 5 pts: {self.within_5_accuracy:.2%}",
            f"  Within 7 pts: {self.within_7_accuracy:.2%}",
            f"  Within 10 pts: {self.within_10_accuracy:.2%}",
            f"",
            f"Error Distribution:",
            f"  Mean (Bias): {self.mean_error:+.3f}",
            f"  Std:         {self.std_error:.3f}",
            f"  Median:      {self.median_error:+.3f}",
            f"{'='*50}"
        ]
        return "\n".join(line for line in lines if line or line == "")

    def save(self, path: Path) -> None:
        """결과 저장"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def meets_criteria(
        self,
        max_rmse: float = 11.5,
        min_win_accuracy: float = 0.66
    ) -> bool:
        """성공 기준 충족 여부"""
        return self.rmse <= max_rmse and self.win_accuracy >= min_win_accuracy


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = ""
) -> MetricsReport:
    """
    종합 메트릭 계산.

    Args:
        y_true: 실제값 (점수차)
        y_pred: 예측값 (점수차)
        model_name: 모델 이름

    Returns:
        MetricsReport 객체
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # 기본 회귀 메트릭
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)

    # 승패 예측 메트릭
    true_wins = y_true > 0
    pred_wins = y_pred > 0

    win_accuracy = np.mean(true_wins == pred_wins)

    # Precision/Recall
    tp = np.sum(pred_wins & true_wins)
    fp = np.sum(pred_wins & ~true_wins)
    fn = np.sum(~pred_wins & true_wins)

    win_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    win_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 범위 내 정확도
    within_3 = np.mean(abs_errors <= 3)
    within_5 = np.mean(abs_errors <= 5)
    within_7 = np.mean(abs_errors <= 7)
    within_10 = np.mean(abs_errors <= 10)

    # 오차 분포
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)

    percentiles = {
        "p10": np.percentile(errors, 10),
        "p25": np.percentile(errors, 25),
        "p50": np.percentile(errors, 50),
        "p75": np.percentile(errors, 75),
        "p90": np.percentile(errors, 90)
    }

    return MetricsReport(
        rmse=rmse,
        mae=mae,
        mse=mse,
        r2=r2,
        explained_variance=ev,
        win_accuracy=win_accuracy,
        win_precision=win_precision,
        win_recall=win_recall,
        within_3_accuracy=within_3,
        within_5_accuracy=within_5,
        within_7_accuracy=within_7,
        within_10_accuracy=within_10,
        mean_error=mean_error,
        std_error=std_error,
        median_error=median_error,
        error_percentiles=percentiles,
        n_samples=len(y_true),
        model_name=model_name
    )


def calculate_betting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spreads: Optional[np.ndarray] = None,
    odds: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    베팅 관련 메트릭 계산.

    Args:
        y_true: 실제 점수차
        y_pred: 예측 점수차
        spreads: 스프레드 라인 (홈팀 기준)
        odds: 배당률 (데시멀 오즈)

    Returns:
        베팅 메트릭 딕셔너리
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    results = {
        "moneyline_accuracy": 0.0,
        "ats_accuracy": 0.0,  # Against The Spread
        "roi": 0.0,
        "kelly_fraction": 0.0
    }

    # 머니라인 정확도 (승패 예측)
    true_wins = y_true > 0
    pred_wins = y_pred > 0
    results["moneyline_accuracy"] = np.mean(true_wins == pred_wins)

    # ATS (스프레드) 정확도
    if spreads is not None:
        spreads = np.asarray(spreads).flatten()

        # 스프레드 커버 여부
        # spread가 -5면 홈팀이 5점 이상 이기면 커버
        true_covers = y_true > spreads
        pred_covers = y_pred > spreads

        results["ats_accuracy"] = np.mean(true_covers == pred_covers)

        # 단순 베팅 ROI (-110 가정, 승리 시 0.909 배당)
        correct_bets = true_covers == pred_covers
        n_bets = len(correct_bets)
        wins = np.sum(correct_bets)
        losses = n_bets - wins

        # 표준 -110 베팅 기준
        profit = wins * 0.909 - losses * 1.0
        results["roi"] = profit / n_bets if n_bets > 0 else 0.0

    # 켈리 기준 (단순화)
    if results["ats_accuracy"] > 0.5:
        # Kelly = (bp - q) / b, where b = 0.909, p = win_rate, q = 1-p
        b = 0.909
        p = results["ats_accuracy"]
        q = 1 - p
        kelly = (b * p - q) / b
        results["kelly_fraction"] = max(0, kelly)

    return results


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    예측 보정(Calibration) 메트릭 계산.

    예측값과 실제값의 분포가 얼마나 일치하는지 측정합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값
        n_bins: 빈 개수

    Returns:
        보정 메트릭 딕셔너리
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # 예측값 기준 빈 분할
    bin_edges = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    calibration_data = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mean_pred = np.mean(y_pred[mask])
            mean_true = np.mean(y_true[mask])
            count = np.sum(mask)

            calibration_data.append({
                "bin": i,
                "mean_predicted": mean_pred,
                "mean_actual": mean_true,
                "count": count,
                "error": mean_pred - mean_true
            })

    # Expected Calibration Error (ECE)
    ece = 0.0
    total = len(y_true)

    for bin_data in calibration_data:
        weight = bin_data["count"] / total
        ece += weight * abs(bin_data["error"])

    # Maximum Calibration Error (MCE)
    mce = max(abs(d["error"]) for d in calibration_data) if calibration_data else 0.0

    return {
        "ece": ece,
        "mce": mce,
        "calibration_bins": calibration_data
    }


def calculate_margin_based_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_thresholds: List[float] = None
) -> Dict[str, Any]:
    """
    예측 마진(confidence) 기반 메트릭.

    예측 확신도가 높을 때의 정확도를 측정합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값
        confidence_thresholds: 확신도 임계값 리스트

    Returns:
        마진 기반 메트릭
    """
    if confidence_thresholds is None:
        confidence_thresholds = [3.0, 5.0, 7.0, 10.0]

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    results = {}

    for threshold in confidence_thresholds:
        # 예측 마진이 임계값 이상인 경기만 선택
        confident_mask = np.abs(y_pred) >= threshold

        if np.sum(confident_mask) > 0:
            confident_true = y_true[confident_mask]
            confident_pred = y_pred[confident_mask]

            # 승패 예측 정확도
            true_wins = confident_true > 0
            pred_wins = confident_pred > 0
            accuracy = np.mean(true_wins == pred_wins)

            results[f"conf_{threshold}_accuracy"] = accuracy
            results[f"conf_{threshold}_coverage"] = np.mean(confident_mask)
            results[f"conf_{threshold}_count"] = np.sum(confident_mask)
        else:
            results[f"conf_{threshold}_accuracy"] = np.nan
            results[f"conf_{threshold}_coverage"] = 0.0
            results[f"conf_{threshold}_count"] = 0

    return results


def calculate_home_away_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    홈/어웨이 구분 메트릭.

    Args:
        y_true: 실제 점수차 (홈 - 어웨이)
        y_pred: 예측 점수차

    Returns:
        홈/어웨이 구분 메트릭
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # 홈팀 승리 경기
    home_win_mask = y_true > 0
    away_win_mask = y_true < 0

    results = {}

    # 홈팀 승리 경기 메트릭
    if np.sum(home_win_mask) > 0:
        results["home_win_rmse"] = np.sqrt(mean_squared_error(
            y_true[home_win_mask], y_pred[home_win_mask]
        ))
        results["home_win_mae"] = mean_absolute_error(
            y_true[home_win_mask], y_pred[home_win_mask]
        )

    # 어웨이팀 승리 경기 메트릭
    if np.sum(away_win_mask) > 0:
        results["away_win_rmse"] = np.sqrt(mean_squared_error(
            y_true[away_win_mask], y_pred[away_win_mask]
        ))
        results["away_win_mae"] = mean_absolute_error(
            y_true[away_win_mask], y_pred[away_win_mask]
        )

    # 접전 경기 (5점 이내)
    close_game_mask = np.abs(y_true) <= 5
    if np.sum(close_game_mask) > 0:
        results["close_game_rmse"] = np.sqrt(mean_squared_error(
            y_true[close_game_mask], y_pred[close_game_mask]
        ))
        results["close_game_win_accuracy"] = np.mean(
            (y_true[close_game_mask] > 0) == (y_pred[close_game_mask] > 0)
        )

    # 대차이 경기 (15점 이상)
    blowout_mask = np.abs(y_true) >= 15
    if np.sum(blowout_mask) > 0:
        results["blowout_rmse"] = np.sqrt(mean_squared_error(
            y_true[blowout_mask], y_pred[blowout_mask]
        ))
        results["blowout_win_accuracy"] = np.mean(
            (y_true[blowout_mask] > 0) == (y_pred[blowout_mask] > 0)
        )

    return results


def compare_models_report(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    여러 모델 비교 리포트 생성.

    Args:
        y_true: 실제값
        predictions: 모델명 -> 예측값 딕셔너리

    Returns:
        비교 DataFrame
    """
    reports = []

    for model_name, y_pred in predictions.items():
        metrics = calculate_metrics(y_true, y_pred, model_name)
        reports.append({
            "model": model_name,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "r2": metrics.r2,
            "win_accuracy": metrics.win_accuracy,
            "within_5": metrics.within_5_accuracy,
            "within_10": metrics.within_10_accuracy,
            "bias": metrics.mean_error
        })

    df = pd.DataFrame(reports)
    df = df.sort_values("rmse")

    return df
