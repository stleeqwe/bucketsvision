"""
Model Analysis Module.

모델 예측 오차 분석, 피처 중요도 분석, 성능 분해를 위한 도구.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.logger import logger


@dataclass
class ErrorAnalysisResult:
    """오차 분석 결과"""
    overall_stats: Dict[str, float]
    by_margin_stats: Dict[str, Dict[str, float]]
    by_feature_correlation: Dict[str, float]
    worst_predictions: pd.DataFrame
    best_predictions: pd.DataFrame


class ErrorAnalyzer:
    """
    예측 오차 분석기.

    오차의 패턴을 분석하여 모델 개선 방향을 제시합니다.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: Optional[pd.DataFrame] = None,
        metadata: Optional[pd.DataFrame] = None
    ):
        """
        Args:
            y_true: 실제값
            y_pred: 예측값
            features: 피처 DataFrame (오차 상관 분석용)
            metadata: 메타데이터 (game_id, team_id 등)
        """
        self.y_true = np.asarray(y_true).flatten()
        self.y_pred = np.asarray(y_pred).flatten()
        self.errors = self.y_pred - self.y_true
        self.abs_errors = np.abs(self.errors)
        self.features = features
        self.metadata = metadata

    def analyze(self) -> ErrorAnalysisResult:
        """종합 오차 분석"""
        # 전체 통계
        overall_stats = {
            "mean_error": np.mean(self.errors),
            "std_error": np.std(self.errors),
            "median_error": np.median(self.errors),
            "mean_abs_error": np.mean(self.abs_errors),
            "max_abs_error": np.max(self.abs_errors),
            "skewness": self._calculate_skewness(),
            "kurtosis": self._calculate_kurtosis()
        }

        # 마진별 분석
        by_margin_stats = self._analyze_by_margin()

        # 피처 상관 분석
        by_feature_correlation = {}
        if self.features is not None:
            by_feature_correlation = self._analyze_feature_correlation()

        # 최악/최선 예측
        worst_df, best_df = self._get_extreme_predictions()

        return ErrorAnalysisResult(
            overall_stats=overall_stats,
            by_margin_stats=by_margin_stats,
            by_feature_correlation=by_feature_correlation,
            worst_predictions=worst_df,
            best_predictions=best_df
        )

    def _calculate_skewness(self) -> float:
        """왜도 계산"""
        mean = np.mean(self.errors)
        std = np.std(self.errors)
        if std == 0:
            return 0.0
        return np.mean(((self.errors - mean) / std) ** 3)

    def _calculate_kurtosis(self) -> float:
        """첨도 계산"""
        mean = np.mean(self.errors)
        std = np.std(self.errors)
        if std == 0:
            return 0.0
        return np.mean(((self.errors - mean) / std) ** 4) - 3

    def _analyze_by_margin(self) -> Dict[str, Dict[str, float]]:
        """실제 점수차 구간별 분석"""
        results = {}

        # 접전 (5점 이내)
        close_mask = np.abs(self.y_true) <= 5
        if np.sum(close_mask) > 0:
            results["close_games"] = {
                "n_games": int(np.sum(close_mask)),
                "rmse": np.sqrt(np.mean(self.errors[close_mask] ** 2)),
                "mae": np.mean(self.abs_errors[close_mask]),
                "win_accuracy": np.mean(
                    (self.y_true[close_mask] > 0) == (self.y_pred[close_mask] > 0)
                )
            }

        # 중간 (6-14점)
        mid_mask = (np.abs(self.y_true) > 5) & (np.abs(self.y_true) <= 14)
        if np.sum(mid_mask) > 0:
            results["mid_margin"] = {
                "n_games": int(np.sum(mid_mask)),
                "rmse": np.sqrt(np.mean(self.errors[mid_mask] ** 2)),
                "mae": np.mean(self.abs_errors[mid_mask]),
                "win_accuracy": np.mean(
                    (self.y_true[mid_mask] > 0) == (self.y_pred[mid_mask] > 0)
                )
            }

        # 대차이 (15점 이상)
        blowout_mask = np.abs(self.y_true) >= 15
        if np.sum(blowout_mask) > 0:
            results["blowout"] = {
                "n_games": int(np.sum(blowout_mask)),
                "rmse": np.sqrt(np.mean(self.errors[blowout_mask] ** 2)),
                "mae": np.mean(self.abs_errors[blowout_mask]),
                "win_accuracy": np.mean(
                    (self.y_true[blowout_mask] > 0) == (self.y_pred[blowout_mask] > 0)
                )
            }

        return results

    def _analyze_feature_correlation(self) -> Dict[str, float]:
        """피처와 오차의 상관관계"""
        correlations = {}

        for col in self.features.columns:
            if self.features[col].dtype in [np.float64, np.int64]:
                valid_mask = ~(np.isnan(self.features[col]) | np.isnan(self.errors))
                if np.sum(valid_mask) > 10:
                    corr = np.corrcoef(
                        self.features[col].values[valid_mask],
                        self.errors[valid_mask]
                    )[0, 1]
                    correlations[col] = corr

        # 절대값 기준 정렬
        correlations = dict(sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))

        return correlations

    def _get_extreme_predictions(
        self,
        n: int = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """최악/최선 예측 추출"""
        indices = np.argsort(self.abs_errors)

        data = {
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "error": self.errors,
            "abs_error": self.abs_errors
        }

        if self.metadata is not None:
            for col in self.metadata.columns:
                data[col] = self.metadata[col].values

        df = pd.DataFrame(data)

        worst = df.iloc[indices[-n:]].sort_values("abs_error", ascending=False)
        best = df.iloc[indices[:n]].sort_values("abs_error")

        return worst, best

    def get_systematic_bias(self) -> Dict[str, Any]:
        """체계적 편향 분석"""
        bias_analysis = {
            "overall_bias": np.mean(self.errors),
            "bias_significance": self._test_bias_significance()
        }

        # 홈팀 승리 vs 어웨이 승리 편향
        home_win_mask = self.y_true > 0
        if np.sum(home_win_mask) > 0 and np.sum(~home_win_mask) > 0:
            bias_analysis["home_win_bias"] = np.mean(self.errors[home_win_mask])
            bias_analysis["away_win_bias"] = np.mean(self.errors[~home_win_mask])

        # 예측 방향별 편향
        pred_home_mask = self.y_pred > 0
        if np.sum(pred_home_mask) > 0 and np.sum(~pred_home_mask) > 0:
            bias_analysis["when_pred_home_bias"] = np.mean(self.errors[pred_home_mask])
            bias_analysis["when_pred_away_bias"] = np.mean(self.errors[~pred_home_mask])

        return bias_analysis

    def _test_bias_significance(self) -> Dict[str, float]:
        """편향 유의성 검정"""
        from scipy import stats

        n = len(self.errors)
        mean = np.mean(self.errors)
        se = np.std(self.errors) / np.sqrt(n)

        t_stat = mean / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05
        }


class PredictionAnalyzer:
    """
    예측 패턴 분석기.

    모델 예측의 특성과 패턴을 분석합니다.
    """

    def __init__(
        self,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ):
        self.y_pred = np.asarray(y_pred).flatten()
        self.y_true = np.asarray(y_true).flatten() if y_true is not None else None

    def analyze_distribution(self) -> Dict[str, Any]:
        """예측값 분포 분석"""
        return {
            "mean": np.mean(self.y_pred),
            "std": np.std(self.y_pred),
            "min": np.min(self.y_pred),
            "max": np.max(self.y_pred),
            "percentiles": {
                "p5": np.percentile(self.y_pred, 5),
                "p25": np.percentile(self.y_pred, 25),
                "p50": np.percentile(self.y_pred, 50),
                "p75": np.percentile(self.y_pred, 75),
                "p95": np.percentile(self.y_pred, 95)
            },
            "home_win_rate": np.mean(self.y_pred > 0),
            "confident_predictions": {
                "high_conf_rate": np.mean(np.abs(self.y_pred) > 7),
                "mid_conf_rate": np.mean((np.abs(self.y_pred) > 3) & (np.abs(self.y_pred) <= 7)),
                "low_conf_rate": np.mean(np.abs(self.y_pred) <= 3)
            }
        }

    def analyze_agreement_with_actual(self) -> Dict[str, Any]:
        """실제값과의 일치도 분석"""
        if self.y_true is None:
            return {}

        return {
            "correlation": np.corrcoef(self.y_pred, self.y_true)[0, 1],
            "direction_agreement": np.mean(
                (self.y_pred > 0) == (self.y_true > 0)
            ),
            "magnitude_ratio": np.mean(np.abs(self.y_pred)) / np.mean(np.abs(self.y_true)),
            "variance_ratio": np.var(self.y_pred) / np.var(self.y_true) if np.var(self.y_true) > 0 else np.nan
        }

    def analyze_extreme_predictions(
        self,
        threshold: float = 15.0
    ) -> Dict[str, Any]:
        """극단적 예측 분석"""
        extreme_mask = np.abs(self.y_pred) >= threshold

        result = {
            "extreme_rate": np.mean(extreme_mask),
            "n_extreme": int(np.sum(extreme_mask))
        }

        if self.y_true is not None and np.sum(extreme_mask) > 0:
            result["extreme_accuracy"] = np.mean(
                (self.y_true[extreme_mask] > 0) == (self.y_pred[extreme_mask] > 0)
            )
            result["extreme_mae"] = np.mean(
                np.abs(self.y_pred[extreme_mask] - self.y_true[extreme_mask])
            )

        return result


class FeatureImportanceAnalyzer:
    """
    피처 중요도 분석기.

    여러 모델의 피처 중요도를 비교하고 분석합니다.
    """

    def __init__(self, models: List[BaseModel]):
        """
        Args:
            models: 분석할 모델 리스트
        """
        self.models = models

    def get_combined_importance(
        self,
        method: str = "mean"
    ) -> Dict[str, float]:
        """
        여러 모델의 피처 중요도 결합.

        Args:
            method: 결합 방법 ('mean', 'max', 'rank_mean')

        Returns:
            결합된 피처 중요도
        """
        all_importance = {}

        for model in self.models:
            importance = model.get_feature_importance()

            for feature, value in importance.items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(value)

        combined = {}

        for feature, values in all_importance.items():
            if method == "mean":
                combined[feature] = np.mean(values)
            elif method == "max":
                combined[feature] = np.max(values)
            elif method == "rank_mean":
                # 각 모델 내에서 순위 평균
                ranks = []
                for model in self.models:
                    imp = model.get_feature_importance()
                    sorted_features = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                    feature_ranks = {f: i + 1 for i, (f, _) in enumerate(sorted_features)}
                    ranks.append(feature_ranks.get(feature, len(imp) + 1))
                combined[feature] = np.mean(ranks)
            else:
                combined[feature] = np.mean(values)

        # 정렬
        if method == "rank_mean":
            combined = dict(sorted(combined.items(), key=lambda x: x[1]))
        else:
            combined = dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))

        return combined

    def get_importance_dataframe(self) -> pd.DataFrame:
        """피처 중요도 DataFrame"""
        data = []

        for model in self.models:
            importance = model.get_feature_importance()
            for feature, value in importance.items():
                data.append({
                    "model": model.name,
                    "feature": feature,
                    "importance": value
                })

        df = pd.DataFrame(data)

        # 피벗 테이블로 변환
        pivot = df.pivot(index="feature", columns="model", values="importance")
        pivot["mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("mean", ascending=False)

        return pivot

    def analyze_consistency(self) -> Dict[str, Any]:
        """모델 간 피처 중요도 일관성 분석"""
        importance_df = self.get_importance_dataframe()

        # 모델 간 상관관계
        model_cols = [c for c in importance_df.columns if c != "mean"]

        correlations = {}
        for i, m1 in enumerate(model_cols):
            for m2 in model_cols[i + 1:]:
                valid = ~(importance_df[m1].isna() | importance_df[m2].isna())
                if valid.sum() > 0:
                    corr = np.corrcoef(
                        importance_df.loc[valid, m1],
                        importance_df.loc[valid, m2]
                    )[0, 1]
                    correlations[f"{m1}_vs_{m2}"] = corr

        # 상위 10개 피처 일관성
        top_features_per_model = {}
        for model in self.models:
            top = model.get_top_features(10)
            top_features_per_model[model.name] = set(f for f, _ in top)

        # 모든 모델에서 공통으로 상위인 피처
        if top_features_per_model:
            common_top = set.intersection(*top_features_per_model.values())
        else:
            common_top = set()

        return {
            "model_correlations": correlations,
            "common_top_features": list(common_top),
            "n_common_top": len(common_top)
        }

    def get_feature_ranking(self) -> pd.DataFrame:
        """피처 순위 테이블"""
        rankings = {}

        for model in self.models:
            importance = model.get_feature_importance()
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            for rank, (feature, _) in enumerate(sorted_features, 1):
                if feature not in rankings:
                    rankings[feature] = {}
                rankings[feature][model.name] = rank

        df = pd.DataFrame(rankings).T
        df["mean_rank"] = df.mean(axis=1)
        df["std_rank"] = df.std(axis=1)
        df = df.sort_values("mean_rank")

        return df
