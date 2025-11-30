"""
Statistical Validator.

On/Off 분석 결과의 통계적 유의성을 검증하고
신뢰구간, 효과 크기 등을 계산합니다.

검증 항목:
- Independent t-test (Welch's)
- Bootstrap confidence interval
- Cohen's d effect size
- Power analysis
- Multiple comparison correction (FDR)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ValidationResult:
    """통계 검증 결과"""
    # 기본 통계
    mean_diff: float
    std_on: float
    std_off: float

    # t-test
    t_statistic: float
    p_value: float
    df: float  # degrees of freedom

    # 효과 크기
    cohens_d: float
    hedges_g: float  # small sample correction

    # 신뢰구간
    ci_95: Tuple[float, float]
    ci_99: Tuple[float, float]
    bootstrap_ci: Tuple[float, float]

    # 검정력 분석
    achieved_power: float
    required_n_for_80_power: int

    # 유의성 판정
    is_significant_05: bool
    is_significant_01: bool
    is_practically_significant: bool  # |d| >= 0.5

    # 품질 평가
    quality_score: float  # 0-1


class StatisticalValidator:
    """
    통계적 유의성 검증기.

    On/Off 분석 결과에 대한 다양한 통계적 검증을 수행합니다.
    """

    # 효과 크기 해석 기준 (Cohen's d)
    EFFECT_SMALL = 0.2
    EFFECT_MEDIUM = 0.5
    EFFECT_LARGE = 0.8

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        random_seed: int = 42
    ):
        """
        Args:
            alpha: 유의 수준
            n_bootstrap: Bootstrap 반복 횟수
            random_seed: 랜덤 시드
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        np.random.seed(random_seed)

    def validate(
        self,
        margins_on: List[float],
        margins_off: List[float]
    ) -> ValidationResult:
        """
        종합 통계 검증.

        Args:
            margins_on: 출전 경기 마진 리스트
            margins_off: 미출전 경기 마진 리스트

        Returns:
            ValidationResult
        """
        margins_on = np.array(margins_on)
        margins_off = np.array(margins_off)

        n_on = len(margins_on)
        n_off = len(margins_off)

        if n_on < 2 or n_off < 2:
            return self._empty_result()

        # 기본 통계
        mean_on = np.mean(margins_on)
        mean_off = np.mean(margins_off)
        mean_diff = mean_on - mean_off
        std_on = np.std(margins_on, ddof=1)
        std_off = np.std(margins_off, ddof=1)

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(margins_on, margins_off, equal_var=False)

        # Degrees of freedom (Welch-Satterthwaite)
        var_on = std_on ** 2
        var_off = std_off ** 2
        df = ((var_on / n_on + var_off / n_off) ** 2) / (
            (var_on / n_on) ** 2 / (n_on - 1) +
            (var_off / n_off) ** 2 / (n_off - 1)
        )

        # 효과 크기
        cohens_d = self._cohens_d(margins_on, margins_off)
        hedges_g = self._hedges_g(margins_on, margins_off)

        # 신뢰구간
        ci_95 = self._confidence_interval(margins_on, margins_off, 0.95)
        ci_99 = self._confidence_interval(margins_on, margins_off, 0.99)
        bootstrap_ci = self._bootstrap_ci(margins_on, margins_off)

        # 검정력 분석
        achieved_power = self._achieved_power(n_on, n_off, cohens_d)
        required_n = self._required_sample_size(cohens_d)

        # 품질 점수 계산
        quality_score = self._calculate_quality_score(
            n_on, n_off, p_value, cohens_d, achieved_power
        )

        return ValidationResult(
            mean_diff=mean_diff,
            std_on=std_on,
            std_off=std_off,
            t_statistic=t_stat,
            p_value=p_value,
            df=df,
            cohens_d=cohens_d,
            hedges_g=hedges_g,
            ci_95=ci_95,
            ci_99=ci_99,
            bootstrap_ci=bootstrap_ci,
            achieved_power=achieved_power,
            required_n_for_80_power=required_n,
            is_significant_05=p_value < 0.05,
            is_significant_01=p_value < 0.01,
            is_practically_significant=abs(cohens_d) >= self.EFFECT_MEDIUM,
            quality_score=quality_score,
        )

    def _cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> float:
        """Cohen's d 효과 크기 계산"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        )

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def _hedges_g(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> float:
        """Hedges' g (small sample corrected Cohen's d)"""
        d = self._cohens_d(group1, group2)
        n = len(group1) + len(group2)

        # Small sample correction factor
        correction = 1 - (3 / (4 * (n - 2) - 1))

        return d * correction

    def _confidence_interval(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence: float
    ) -> Tuple[float, float]:
        """평균 차이의 신뢰구간 (Welch's method)"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        se = np.sqrt(var1 / n1 + var2 / n2)

        # Welch-Satterthwaite degrees of freedom
        df = ((var1 / n1 + var2 / n2) ** 2) / (
            (var1 / n1) ** 2 / (n1 - 1) +
            (var2 / n2) ** 2 / (n2 - 1)
        )

        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        margin = t_crit * se

        mean_diff = mean1 - mean2
        return (mean_diff - margin, mean_diff + margin)

    def _bootstrap_ci(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Bootstrap 신뢰구간"""
        boot_diffs = []

        for _ in range(self.n_bootstrap):
            boot1 = np.random.choice(group1, size=len(group1), replace=True)
            boot2 = np.random.choice(group2, size=len(group2), replace=True)
            boot_diffs.append(np.mean(boot1) - np.mean(boot2))

        alpha = 1 - confidence
        lower = np.percentile(boot_diffs, 100 * alpha / 2)
        upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

        return (lower, upper)

    def _achieved_power(
        self,
        n1: int,
        n2: int,
        effect_size: float,
        alpha: float = 0.05
    ) -> float:
        """
        달성된 검정력 계산.

        Two-sample t-test의 사후 검정력.
        """
        if effect_size == 0 or n1 < 2 or n2 < 2:
            return 0.0

        # Harmonic mean of sample sizes
        n_harmonic = 2 * n1 * n2 / (n1 + n2)

        # Non-centrality parameter
        ncp = abs(effect_size) * np.sqrt(n_harmonic / 2)

        # Critical value
        df = n1 + n2 - 2
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        # Power = P(reject H0 | H1 is true)
        # Using non-central t-distribution
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

        return min(1.0, max(0.0, power))

    def _required_sample_size(
        self,
        effect_size: float,
        power: float = 0.80,
        alpha: float = 0.05
    ) -> int:
        """
        목표 검정력을 위한 필요 표본 크기.

        각 그룹당 필요한 표본 크기 반환.
        """
        if effect_size == 0:
            return float('inf')

        # Approximation formula
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return max(2, int(np.ceil(n)))

    def _calculate_quality_score(
        self,
        n_on: int,
        n_off: int,
        p_value: float,
        effect_size: float,
        power: float
    ) -> float:
        """
        종합 품질 점수 계산 (0-1).

        고려 요소:
        - 표본 크기
        - p-value
        - 효과 크기
        - 검정력
        """
        # 표본 크기 점수 (0-0.3)
        min_n = min(n_on, n_off)
        size_score = min(0.3, 0.3 * min(min_n / 30, 1))

        # p-value 점수 (0-0.3)
        if p_value < 0.01:
            p_score = 0.3
        elif p_value < 0.05:
            p_score = 0.2
        elif p_value < 0.10:
            p_score = 0.1
        else:
            p_score = 0.0

        # 효과 크기 점수 (0-0.2)
        abs_d = abs(effect_size)
        if abs_d >= self.EFFECT_LARGE:
            effect_score = 0.2
        elif abs_d >= self.EFFECT_MEDIUM:
            effect_score = 0.15
        elif abs_d >= self.EFFECT_SMALL:
            effect_score = 0.1
        else:
            effect_score = 0.05

        # 검정력 점수 (0-0.2)
        power_score = 0.2 * min(power / 0.8, 1)

        return size_score + p_score + effect_score + power_score

    def _empty_result(self) -> ValidationResult:
        """빈 결과 반환 (데이터 부족 시)"""
        return ValidationResult(
            mean_diff=np.nan,
            std_on=np.nan,
            std_off=np.nan,
            t_statistic=np.nan,
            p_value=np.nan,
            df=np.nan,
            cohens_d=np.nan,
            hedges_g=np.nan,
            ci_95=(np.nan, np.nan),
            ci_99=(np.nan, np.nan),
            bootstrap_ci=(np.nan, np.nan),
            achieved_power=0.0,
            required_n_for_80_power=float('inf'),
            is_significant_05=False,
            is_significant_01=False,
            is_practically_significant=False,
            quality_score=0.0,
        )

    def validate_batch(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        배치 검증 (여러 선수 동시 처리).

        Args:
            results_df: On/Off 분석 결과 DataFrame

        Returns:
            검증 결과가 추가된 DataFrame
        """
        # Multiple comparison correction (FDR)
        if "p_value" in results_df.columns:
            p_values = results_df["p_value"].values
            # Benjamini-Hochberg procedure
            fdr_adjusted = self._benjamini_hochberg(p_values)
            results_df = results_df.copy()
            results_df["p_value_fdr"] = fdr_adjusted
            results_df["is_significant_fdr"] = fdr_adjusted < self.alpha

        return results_df

    def _benjamini_hochberg(
        self,
        p_values: np.ndarray
    ) -> np.ndarray:
        """Benjamini-Hochberg FDR correction"""
        n = len(p_values)
        if n == 0:
            return np.array([])

        # Handle NaN
        valid_mask = ~np.isnan(p_values)
        valid_p = p_values[valid_mask]

        if len(valid_p) == 0:
            return p_values

        # Sort and rank
        sorted_idx = np.argsort(valid_p)
        sorted_p = valid_p[sorted_idx]

        # BH adjustment
        ranks = np.arange(1, len(valid_p) + 1)
        adjusted = sorted_p * len(valid_p) / ranks

        # Ensure monotonicity
        for i in range(len(adjusted) - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        adjusted = np.minimum(adjusted, 1.0)

        # Unsort
        unsorted = np.empty_like(adjusted)
        unsorted[sorted_idx] = adjusted

        # Restore with NaN
        result = np.full_like(p_values, np.nan)
        result[valid_mask] = unsorted

        return result

    def interpret_effect_size(self, d: float) -> str:
        """효과 크기 해석"""
        abs_d = abs(d)
        if abs_d < self.EFFECT_SMALL:
            return "negligible"
        elif abs_d < self.EFFECT_MEDIUM:
            return "small"
        elif abs_d < self.EFFECT_LARGE:
            return "medium"
        else:
            return "large"

    def generate_report(
        self,
        result: ValidationResult,
        player_name: str = ""
    ) -> str:
        """검증 결과 리포트 생성"""
        lines = [
            f"Statistical Validation Report",
            f"{'='*50}",
            f"Player: {player_name}" if player_name else "",
            "",
            "1. Basic Statistics",
            f"   Mean difference: {result.mean_diff:.3f}",
            f"   Std (on): {result.std_on:.3f}, Std (off): {result.std_off:.3f}",
            "",
            "2. Hypothesis Test (Welch's t-test)",
            f"   t-statistic: {result.t_statistic:.3f}",
            f"   p-value: {result.p_value:.4f}",
            f"   df: {result.df:.1f}",
            f"   Significant (α=0.05): {'Yes' if result.is_significant_05 else 'No'}",
            "",
            "3. Effect Size",
            f"   Cohen's d: {result.cohens_d:.3f} ({self.interpret_effect_size(result.cohens_d)})",
            f"   Hedges' g: {result.hedges_g:.3f}",
            f"   Practically significant: {'Yes' if result.is_practically_significant else 'No'}",
            "",
            "4. Confidence Intervals",
            f"   95% CI: [{result.ci_95[0]:.3f}, {result.ci_95[1]:.3f}]",
            f"   99% CI: [{result.ci_99[0]:.3f}, {result.ci_99[1]:.3f}]",
            f"   Bootstrap CI: [{result.bootstrap_ci[0]:.3f}, {result.bootstrap_ci[1]:.3f}]",
            "",
            "5. Power Analysis",
            f"   Achieved power: {result.achieved_power:.3f}",
            f"   Required n for 80% power: {result.required_n_for_80_power}",
            "",
            "6. Quality Assessment",
            f"   Quality score: {result.quality_score:.2f}/1.00",
            f"{'='*50}",
        ]

        return "\n".join(filter(None, lines))
