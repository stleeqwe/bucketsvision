#!/usr/bin/env python3
"""
Phase 2: 통일된 평가 프레임워크.

V1, V2, V3 모델을 동일한 메트릭으로 평가합니다.

실행: python scripts/benchmark/unified_evaluator.py
"""

import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional


@dataclass
class ModelMetrics:
    """모델 평가 메트릭"""
    model_name: str
    n_samples: int

    # 분류 메트릭
    accuracy: float
    precision: float
    recall: float
    f1: float

    # 확률 메트릭
    brier_score: float
    log_loss_value: float
    auc_roc: float

    # 캘리브레이션
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error

    # 신뢰도별 정확도
    conf_55_acc: float
    conf_60_acc: float
    conf_65_acc: float
    conf_70_acc: float
    conf_75_acc: float

    conf_55_coverage: float
    conf_60_coverage: float
    conf_65_coverage: float
    conf_70_coverage: float
    conf_75_coverage: float


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE 계산"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return ece


def maximum_calibration_error(y_true, y_prob, n_bins=10):
    """MCE 계산"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            mce = max(mce, abs(bin_acc - bin_conf))

    return mce


def confidence_accuracy(y_true, y_prob, threshold):
    """특정 신뢰도 이상 예측의 정확도와 커버리지"""
    mask = (y_prob >= threshold) | (y_prob <= 1 - threshold)
    if mask.sum() == 0:
        return np.nan, 0.0

    y_pred = (y_prob[mask] >= 0.5).astype(int)
    acc = accuracy_score(y_true[mask], y_pred)
    coverage = mask.mean()
    return acc, coverage


def evaluate_model(y_true, y_prob, model_name="model"):
    """모델 종합 평가"""
    y_pred = (y_prob >= 0.5).astype(int)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)

    # 기본 메트릭
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 확률 메트릭
    brier = brier_score_loss(y_true, y_prob_clipped)
    ll = log_loss(y_true, y_prob_clipped)
    auc = roc_auc_score(y_true, y_prob_clipped)

    # 캘리브레이션
    ece = expected_calibration_error(y_true, y_prob_clipped)
    mce = maximum_calibration_error(y_true, y_prob_clipped)

    # 신뢰도별
    conf_55_acc, conf_55_cov = confidence_accuracy(y_true, y_prob_clipped, 0.55)
    conf_60_acc, conf_60_cov = confidence_accuracy(y_true, y_prob_clipped, 0.60)
    conf_65_acc, conf_65_cov = confidence_accuracy(y_true, y_prob_clipped, 0.65)
    conf_70_acc, conf_70_cov = confidence_accuracy(y_true, y_prob_clipped, 0.70)
    conf_75_acc, conf_75_cov = confidence_accuracy(y_true, y_prob_clipped, 0.75)

    return ModelMetrics(
        model_name=model_name,
        n_samples=len(y_true),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        brier_score=brier,
        log_loss_value=ll,
        auc_roc=auc,
        ece=ece,
        mce=mce,
        conf_55_acc=conf_55_acc,
        conf_60_acc=conf_60_acc,
        conf_65_acc=conf_65_acc,
        conf_70_acc=conf_70_acc,
        conf_75_acc=conf_75_acc,
        conf_55_coverage=conf_55_cov,
        conf_60_coverage=conf_60_cov,
        conf_65_coverage=conf_65_cov,
        conf_70_coverage=conf_70_cov,
        conf_75_coverage=conf_75_cov,
    )


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar Test: 두 모델 예측 차이의 통계적 유의성"""
    # contingency table
    # b = A correct, B incorrect
    # c = A incorrect, B correct
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = np.sum(correct_a & ~correct_b)  # A만 맞춤
    c = np.sum(~correct_a & correct_b)  # B만 맞춤

    # McNemar's chi-squared
    if b + c == 0:
        return 0.0, 1.0  # 차이 없음

    # 연속성 보정 적용
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value


def bootstrap_confidence_interval(y_true, y_prob, metric_fn, n_bootstrap=1000, ci=0.95):
    """Bootstrap Confidence Interval"""
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            scores.append(score)
        except:
            continue

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    mean = np.mean(scores)

    return mean, lower, upper


def paired_bootstrap_test(y_true, y_prob_a, y_prob_b, metric_fn, n_bootstrap=1000):
    """Paired Bootstrap Test: 두 모델 메트릭 차이의 유의성"""
    n = len(y_true)
    diff_scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            score_a = metric_fn(y_true[idx], y_prob_a[idx])
            score_b = metric_fn(y_true[idx], y_prob_b[idx])
            diff_scores.append(score_a - score_b)
        except:
            continue

    diff_scores = np.array(diff_scores)

    # p-value: 0이 confidence interval에 포함되는지
    ci_lower = np.percentile(diff_scores, 2.5)
    ci_upper = np.percentile(diff_scores, 97.5)

    # 단측 p-value 근사
    p_value = min(
        np.mean(diff_scores <= 0),
        np.mean(diff_scores >= 0)
    ) * 2

    return np.mean(diff_scores), ci_lower, ci_upper, p_value


def calibration_bins(y_true, y_prob, n_bins=10):
    """캘리브레이션 빈 데이터"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins_data = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bins_data.append({
                'bin': i,
                'bin_start': bin_edges[i],
                'bin_end': bin_edges[i + 1],
                'mean_predicted': y_prob[mask].mean(),
                'mean_actual': y_true[mask].mean(),
                'count': int(mask.sum()),
                'calibration_error': abs(y_prob[mask].mean() - y_true[mask].mean())
            })

    return bins_data


def main():
    print("=" * 70)
    print("  Phase 2: Unified Evaluation Framework")
    print("=" * 70)

    # 데이터 로드
    data_dir = PROJECT_ROOT / "data" / "benchmark"
    df = pd.read_parquet(data_dir / "unified_predictions.parquet")

    y_true = df['y_true'].values
    v1_prob = df['v1_prob'].values
    v2_prob = df['v2_prob'].values
    v3_prob = df['v3_prob'].values

    v1_pred = df['v1_pred'].values
    v2_pred = df['v2_pred'].values
    v3_pred = df['v3_pred'].values

    # 1. 기본 평가
    print("\n" + "=" * 70)
    print("  1. Comprehensive Metrics")
    print("=" * 70)

    v1_metrics = evaluate_model(y_true, v1_prob, "V1 (Ridge)")
    v2_metrics = evaluate_model(y_true, v2_prob, "V2 (XGBoost)")
    v3_metrics = evaluate_model(y_true, v3_prob, "V3 (Stacking)")

    metrics_list = [v1_metrics, v2_metrics, v3_metrics]

    print(f"\n  {'Metric':<20} {'V1 (Ridge)':<15} {'V2 (XGBoost)':<15} {'V3 (Stacking)':<15}")
    print("  " + "-" * 65)

    print(f"  {'Accuracy':<20} {v1_metrics.accuracy:.4f}          {v2_metrics.accuracy:.4f}          {v3_metrics.accuracy:.4f}")
    print(f"  {'Precision':<20} {v1_metrics.precision:.4f}          {v2_metrics.precision:.4f}          {v3_metrics.precision:.4f}")
    print(f"  {'Recall':<20} {v1_metrics.recall:.4f}          {v2_metrics.recall:.4f}          {v3_metrics.recall:.4f}")
    print(f"  {'F1 Score':<20} {v1_metrics.f1:.4f}          {v2_metrics.f1:.4f}          {v3_metrics.f1:.4f}")
    print()
    print(f"  {'Brier Score':<20} {v1_metrics.brier_score:.4f}          {v2_metrics.brier_score:.4f}          {v3_metrics.brier_score:.4f}")
    print(f"  {'Log Loss':<20} {v1_metrics.log_loss_value:.4f}          {v2_metrics.log_loss_value:.4f}          {v3_metrics.log_loss_value:.4f}")
    print(f"  {'AUC-ROC':<20} {v1_metrics.auc_roc:.4f}          {v2_metrics.auc_roc:.4f}          {v3_metrics.auc_roc:.4f}")
    print()
    print(f"  {'ECE':<20} {v1_metrics.ece:.4f}          {v2_metrics.ece:.4f}          {v3_metrics.ece:.4f}")
    print(f"  {'MCE':<20} {v1_metrics.mce:.4f}          {v2_metrics.mce:.4f}          {v3_metrics.mce:.4f}")

    # 2. 신뢰도별 정확도
    print("\n" + "=" * 70)
    print("  2. Confidence-based Accuracy")
    print("=" * 70)

    print(f"\n  {'Confidence':<12} {'V1 Acc':<10} {'V1 Cov':<10} {'V2 Acc':<10} {'V2 Cov':<10} {'V3 Acc':<10} {'V3 Cov':<10}")
    print("  " + "-" * 70)

    for thresh, (v1a, v1c, v2a, v2c, v3a, v3c) in [
        ('55%+', (v1_metrics.conf_55_acc, v1_metrics.conf_55_coverage, v2_metrics.conf_55_acc, v2_metrics.conf_55_coverage, v3_metrics.conf_55_acc, v3_metrics.conf_55_coverage)),
        ('60%+', (v1_metrics.conf_60_acc, v1_metrics.conf_60_coverage, v2_metrics.conf_60_acc, v2_metrics.conf_60_coverage, v3_metrics.conf_60_acc, v3_metrics.conf_60_coverage)),
        ('65%+', (v1_metrics.conf_65_acc, v1_metrics.conf_65_coverage, v2_metrics.conf_65_acc, v2_metrics.conf_65_coverage, v3_metrics.conf_65_acc, v3_metrics.conf_65_coverage)),
        ('70%+', (v1_metrics.conf_70_acc, v1_metrics.conf_70_coverage, v2_metrics.conf_70_acc, v2_metrics.conf_70_coverage, v3_metrics.conf_70_acc, v3_metrics.conf_70_coverage)),
        ('75%+', (v1_metrics.conf_75_acc, v1_metrics.conf_75_coverage, v2_metrics.conf_75_acc, v2_metrics.conf_75_coverage, v3_metrics.conf_75_acc, v3_metrics.conf_75_coverage)),
    ]:
        v1a_str = f"{v1a:.2%}" if not np.isnan(v1a) else "N/A"
        v2a_str = f"{v2a:.2%}" if not np.isnan(v2a) else "N/A"
        v3a_str = f"{v3a:.2%}" if not np.isnan(v3a) else "N/A"
        print(f"  {thresh:<12} {v1a_str:<10} {v1c:.1%}      {v2a_str:<10} {v2c:.1%}      {v3a_str:<10} {v3c:.1%}")

    # 3. McNemar Test
    print("\n" + "=" * 70)
    print("  3. Statistical Significance (McNemar Test)")
    print("=" * 70)

    chi2_v1v2, p_v1v2 = mcnemar_test(y_true, v1_pred, v2_pred)
    chi2_v1v3, p_v1v3 = mcnemar_test(y_true, v1_pred, v3_pred)
    chi2_v2v3, p_v2v3 = mcnemar_test(y_true, v2_pred, v3_pred)

    print(f"\n  {'Comparison':<20} {'Chi-squared':<15} {'p-value':<15} {'Significant (α=0.05)'}")
    print("  " + "-" * 70)
    print(f"  {'V1 vs V2':<20} {chi2_v1v2:<15.4f} {p_v1v2:<15.4f} {'Yes' if p_v1v2 < 0.05 else 'No'}")
    print(f"  {'V1 vs V3':<20} {chi2_v1v3:<15.4f} {p_v1v3:<15.4f} {'Yes' if p_v1v3 < 0.05 else 'No'}")
    print(f"  {'V2 vs V3':<20} {chi2_v2v3:<15.4f} {p_v2v3:<15.4f} {'Yes' if p_v2v3 < 0.05 else 'No'}")

    # 4. Bootstrap Confidence Intervals
    print("\n" + "=" * 70)
    print("  4. Bootstrap Confidence Intervals (95%)")
    print("=" * 70)

    print("\n  Computing bootstrap CIs (n=1000)...")

    def acc_fn(y, p): return accuracy_score(y, (p >= 0.5).astype(int))
    def brier_fn(y, p): return brier_score_loss(y, np.clip(p, 1e-7, 1-1e-7))

    v1_acc_mean, v1_acc_lo, v1_acc_hi = bootstrap_confidence_interval(y_true, v1_prob, acc_fn)
    v2_acc_mean, v2_acc_lo, v2_acc_hi = bootstrap_confidence_interval(y_true, v2_prob, acc_fn)
    v3_acc_mean, v3_acc_lo, v3_acc_hi = bootstrap_confidence_interval(y_true, v3_prob, acc_fn)

    v1_brier_mean, v1_brier_lo, v1_brier_hi = bootstrap_confidence_interval(y_true, v1_prob, brier_fn)
    v2_brier_mean, v2_brier_lo, v2_brier_hi = bootstrap_confidence_interval(y_true, v2_prob, brier_fn)
    v3_brier_mean, v3_brier_lo, v3_brier_hi = bootstrap_confidence_interval(y_true, v3_prob, brier_fn)

    print(f"\n  Accuracy:")
    print(f"    V1 (Ridge):    {v1_acc_mean:.4f} [{v1_acc_lo:.4f}, {v1_acc_hi:.4f}]")
    print(f"    V2 (XGBoost):  {v2_acc_mean:.4f} [{v2_acc_lo:.4f}, {v2_acc_hi:.4f}]")
    print(f"    V3 (Stacking): {v3_acc_mean:.4f} [{v3_acc_lo:.4f}, {v3_acc_hi:.4f}]")

    print(f"\n  Brier Score:")
    print(f"    V1 (Ridge):    {v1_brier_mean:.4f} [{v1_brier_lo:.4f}, {v1_brier_hi:.4f}]")
    print(f"    V2 (XGBoost):  {v2_brier_mean:.4f} [{v2_brier_lo:.4f}, {v2_brier_hi:.4f}]")
    print(f"    V3 (Stacking): {v3_brier_mean:.4f} [{v3_brier_lo:.4f}, {v3_brier_hi:.4f}]")

    # 5. Paired Bootstrap Test
    print("\n" + "=" * 70)
    print("  5. Paired Bootstrap Test (Accuracy Difference)")
    print("=" * 70)

    diff_v1v3, lo_v1v3, hi_v1v3, p_v1v3_boot = paired_bootstrap_test(y_true, v1_prob, v3_prob, acc_fn)
    diff_v2v3, lo_v2v3, hi_v2v3, p_v2v3_boot = paired_bootstrap_test(y_true, v2_prob, v3_prob, acc_fn)

    print(f"\n  {'Comparison':<15} {'Mean Diff':<12} {'95% CI':<25} {'p-value':<12} {'Significant'}")
    print("  " + "-" * 75)
    print(f"  {'V1 - V3':<15} {diff_v1v3:+.4f}       [{lo_v1v3:+.4f}, {hi_v1v3:+.4f}]      {p_v1v3_boot:.4f}        {'Yes' if p_v1v3_boot < 0.05 else 'No'}")
    print(f"  {'V2 - V3':<15} {diff_v2v3:+.4f}       [{lo_v2v3:+.4f}, {hi_v2v3:+.4f}]      {p_v2v3_boot:.4f}        {'Yes' if p_v2v3_boot < 0.05 else 'No'}")

    # 6. Calibration Analysis
    print("\n" + "=" * 70)
    print("  6. Calibration Analysis (Reliability)")
    print("=" * 70)

    v1_bins = calibration_bins(y_true, v1_prob)
    v2_bins = calibration_bins(y_true, v2_prob)
    v3_bins = calibration_bins(y_true, v3_prob)

    print(f"\n  V1 (Ridge) Calibration:")
    print(f"    {'Bin':<10} {'Predicted':<12} {'Actual':<12} {'Count':<10} {'Error':<10}")
    for b in v1_bins:
        print(f"    {b['bin_start']:.1f}-{b['bin_end']:.1f}     {b['mean_predicted']:.3f}        {b['mean_actual']:.3f}        {b['count']:<10} {b['calibration_error']:.3f}")

    print(f"\n  V3 (Stacking) Calibration:")
    print(f"    {'Bin':<10} {'Predicted':<12} {'Actual':<12} {'Count':<10} {'Error':<10}")
    for b in v3_bins:
        print(f"    {b['bin_start']:.1f}-{b['bin_end']:.1f}     {b['mean_predicted']:.3f}        {b['mean_actual']:.3f}        {b['count']:<10} {b['calibration_error']:.3f}")

    # 7. 결과 저장
    print("\n" + "=" * 70)
    print("  Saving Results")
    print("=" * 70)

    results = {
        'metrics': {
            'v1': asdict(v1_metrics),
            'v2': asdict(v2_metrics),
            'v3': asdict(v3_metrics),
        },
        'mcnemar_tests': {
            'v1_vs_v2': {'chi2': chi2_v1v2, 'p_value': p_v1v2, 'significant': p_v1v2 < 0.05},
            'v1_vs_v3': {'chi2': chi2_v1v3, 'p_value': p_v1v3, 'significant': p_v1v3 < 0.05},
            'v2_vs_v3': {'chi2': chi2_v2v3, 'p_value': p_v2v3, 'significant': p_v2v3 < 0.05},
        },
        'bootstrap_ci': {
            'accuracy': {
                'v1': {'mean': v1_acc_mean, 'lower': v1_acc_lo, 'upper': v1_acc_hi},
                'v2': {'mean': v2_acc_mean, 'lower': v2_acc_lo, 'upper': v2_acc_hi},
                'v3': {'mean': v3_acc_mean, 'lower': v3_acc_lo, 'upper': v3_acc_hi},
            },
            'brier_score': {
                'v1': {'mean': v1_brier_mean, 'lower': v1_brier_lo, 'upper': v1_brier_hi},
                'v2': {'mean': v2_brier_mean, 'lower': v2_brier_lo, 'upper': v2_brier_hi},
                'v3': {'mean': v3_brier_mean, 'lower': v3_brier_lo, 'upper': v3_brier_hi},
            }
        },
        'paired_bootstrap': {
            'v1_vs_v3': {'diff': diff_v1v3, 'ci_lower': lo_v1v3, 'ci_upper': hi_v1v3, 'p_value': p_v1v3_boot},
            'v2_vs_v3': {'diff': diff_v2v3, 'ci_lower': lo_v2v3, 'ci_upper': hi_v2v3, 'p_value': p_v2v3_boot},
        },
        'calibration': {
            'v1': v1_bins,
            'v2': v2_bins,
            'v3': v3_bins,
        }
    }

    with open(data_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n  Results saved to: {data_dir / 'evaluation_results.json'}")

    print("\n" + "=" * 70)
    print("  Phase 2 Complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
