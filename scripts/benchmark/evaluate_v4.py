#!/usr/bin/env python3
"""
V4 모델 종합 평가.

V1, V2, V3, V4 모델을 동일한 테스트셋에서 비교 평가합니다.
통계적 유의성 검정 포함.

실행: python scripts/benchmark/evaluate_v4.py
"""

import sys
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bucketsvision_v3" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "bucketsvision_v4" / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from scipy.stats import chi2_contingency


def load_test_data():
    """테스트 데이터 로드"""
    v2_data_dir = PROJECT_ROOT / "bucketsvision_v2" / "data"

    val_games = pd.read_parquet(v2_data_dir / "val_games.parquet")
    val_boxscores = pd.read_parquet(v2_data_dir / "val_boxscores.parquet")
    train_boxscores = pd.read_parquet(v2_data_dir / "train_boxscores.parquet")
    team_stats = pd.read_parquet(v2_data_dir / "team_stats.parquet")

    all_boxscores = pd.concat([train_boxscores, val_boxscores], ignore_index=True)

    return val_games, all_boxscores, team_stats


def margin_to_probability(margin, k=12.0):
    """점수차를 승률 확률로 변환"""
    return 1 / (1 + np.exp(-margin / k))


def build_v1_features(games_df, team_stats_df):
    """V1 Ridge 모델용 피처 생성"""
    features_list = []

    for _, game in games_df.iterrows():
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        game_date = game['game_date']

        home_epm = team_stats_df[
            (team_stats_df['team_id'] == home_id) &
            (team_stats_df['game_dt'] < game_date)
        ].sort_values('game_dt', ascending=False)

        away_epm = team_stats_df[
            (team_stats_df['team_id'] == away_id) &
            (team_stats_df['game_dt'] < game_date)
        ].sort_values('game_dt', ascending=False)

        if len(home_epm) == 0 or len(away_epm) == 0:
            continue

        home_latest = home_epm.iloc[0]
        away_latest = away_epm.iloc[0]

        feat = {
            'game_id': game['game_id'],
            'team_epm_diff': home_latest.get('team_epm', 0) - away_latest.get('team_epm', 0),
            'team_oepm_diff': home_latest.get('team_oepm', 0) - away_latest.get('team_oepm', 0),
            'team_depm_diff': home_latest.get('team_depm', 0) - away_latest.get('team_depm', 0),
            'team_epm_go_diff': home_latest.get('team_epm_game_optimized', 0) - away_latest.get('team_epm_game_optimized', 0),
            'team_oepm_go_diff': home_latest.get('team_oepm_game_optimized', 0) - away_latest.get('team_oepm_game_optimized', 0),
            'team_depm_go_diff': home_latest.get('team_depm_game_optimized', 0) - away_latest.get('team_depm_game_optimized', 0),
            'sos_diff': home_latest.get('sos', 0) - away_latest.get('sos', 0),
            'sos_o_diff': home_latest.get('sos_o', 0) - away_latest.get('sos_o', 0),
            'sos_d_diff': home_latest.get('sos_d', 0) - away_latest.get('sos_d', 0),
            'team_epm_rk_diff': home_latest.get('team_epm_rk', 15) - away_latest.get('team_epm_rk', 15),
            'team_oepm_rk_diff': home_latest.get('team_oepm_rk', 15) - away_latest.get('team_oepm_rk', 15),
            'team_depm_rk_diff': home_latest.get('team_depm_rk', 15) - away_latest.get('team_depm_rk', 15),
            'team_epm_z_diff': home_latest.get('team_epm_z', 0) - away_latest.get('team_epm_z', 0),
            'team_oepm_z_diff': home_latest.get('team_oepm_z', 0) - away_latest.get('team_oepm_z', 0),
            'team_depm_z_diff': home_latest.get('team_depm_z', 0) - away_latest.get('team_depm_z', 0),
            'home_advantage': 3.0,
        }
        features_list.append(feat)

    return pd.DataFrame(features_list)


def build_v3_features(games_df, boxscores_df, team_stats_df):
    """V3 Stacking 모델용 피처 생성"""
    from features.feature_builder import V3FeatureBuilder
    builder = V3FeatureBuilder()
    return builder.build_features(games_df, boxscores_df, team_stats_df, verbose=False)


def build_v4_features(games_df, boxscores_df, team_stats_df):
    """V4 Logistic 모델용 피처 생성"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v4_feature_builder",
        str(PROJECT_ROOT / "bucketsvision_v4" / "src" / "features" / "feature_builder.py")
    )
    v4_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4_module)
    builder = v4_module.V4FeatureBuilder()
    return builder.build_features(games_df, boxscores_df, team_stats_df, verbose=False)


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar Test"""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = np.sum(correct_a & ~correct_b)  # A만 맞춤
    c = np.sum(~correct_a & correct_b)  # B만 맞춤

    if b + c == 0:
        return 0.0, 1.0

    chi2 = (b - c) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return chi2, p_value


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Bootstrap 신뢰구간"""
    stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stats.append(np.mean(sample))

    lower = np.percentile(stats, (1 - ci) / 2 * 100)
    upper = np.percentile(stats, (1 + ci) / 2 * 100)

    return np.mean(stats), lower, upper


def main():
    print("=" * 70)
    print("  V4 Model Comprehensive Evaluation")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1/6] Loading Test Data")
    val_games, all_boxscores, team_stats = load_test_data()
    y_true_all = val_games.set_index('game_id')['home_win']
    print(f"  Test games: {len(val_games)}")

    # 2. V1 예측
    print("\n[2/6] V1 Ridge Predictions")
    v1_model_path = PROJECT_ROOT / "data" / "models" / "final" / "ridge" / "model.pkl"
    with open(v1_model_path, "rb") as f:
        v1_model = pickle.load(f)
    with open(PROJECT_ROOT / "data" / "models" / "final" / "feature_names.json", "r") as f:
        v1_features = json.load(f)

    X_v1 = build_v1_features(val_games, team_stats)
    v1_game_ids = X_v1['game_id'].values
    X_v1_input = X_v1[v1_features].fillna(0)
    v1_margin = v1_model.predict(X_v1_input)
    v1_prob = margin_to_probability(v1_margin)
    v1_pred = (v1_margin > 0).astype(int)

    # 3. V3 예측
    print("\n[3/6] V3 Stacking Predictions")
    v3_model_path = PROJECT_ROOT / "bucketsvision_v3" / "models" / "v3_ensemble.pkl"
    with open(v3_model_path, "rb") as f:
        v3_model = pickle.load(f)

    X_v3 = build_v3_features(val_games, all_boxscores, team_stats)
    v3_game_ids = X_v3['game_id'].values
    feature_cols = [c for c in X_v3.columns if c != 'game_id']
    X_v3_input = X_v3[feature_cols].fillna(0).values
    v3_prob = v3_model.predict_proba(X_v3_input)
    v3_pred = (v3_prob >= 0.5).astype(int)

    # 4. V4 예측
    print("\n[4/6] V4 Logistic Predictions")
    v4_model_path = PROJECT_ROOT / "bucketsvision_v4" / "models" / "v4_model.pkl"
    v4_scaler_path = PROJECT_ROOT / "bucketsvision_v4" / "models" / "v4_scaler.pkl"
    with open(v4_model_path, "rb") as f:
        v4_model = pickle.load(f)
    with open(v4_scaler_path, "rb") as f:
        v4_scaler = pickle.load(f)
    with open(PROJECT_ROOT / "bucketsvision_v4" / "models" / "v4_feature_names.json", "r") as f:
        v4_features = json.load(f)

    X_v4 = build_v4_features(val_games, all_boxscores, team_stats)
    v4_game_ids = X_v4['game_id'].values
    X_v4_input = X_v4[v4_features].fillna(0).values
    X_v4_scaled = v4_scaler.transform(X_v4_input)
    v4_prob = v4_model.predict_proba(X_v4_scaled)[:, 1]
    v4_pred = (v4_prob >= 0.5).astype(int)

    # 공통 게임 ID 찾기
    common_ids = set(v1_game_ids) & set(v3_game_ids) & set(v4_game_ids)
    print(f"\n  Common test samples: {len(common_ids)}")

    # 데이터프레임 정렬
    v1_df = pd.DataFrame({'game_id': v1_game_ids, 'v1_prob': v1_prob, 'v1_pred': v1_pred})
    v3_df = pd.DataFrame({'game_id': v3_game_ids, 'v3_prob': v3_prob, 'v3_pred': v3_pred})
    v4_df = pd.DataFrame({'game_id': v4_game_ids, 'v4_prob': v4_prob, 'v4_pred': v4_pred})

    merged = v1_df.merge(v3_df, on='game_id').merge(v4_df, on='game_id')
    merged = merged[merged['game_id'].isin(common_ids)]
    merged['y_true'] = merged['game_id'].map(y_true_all)

    # 5. 성능 비교
    print("\n[5/6] Performance Comparison")
    print("\n" + "=" * 70)

    y = merged['y_true'].values
    v1_p = merged['v1_pred'].values
    v3_p = merged['v3_pred'].values
    v4_p = merged['v4_pred'].values
    v1_proba = merged['v1_prob'].values
    v3_proba = merged['v3_prob'].values
    v4_proba = merged['v4_prob'].values

    metrics = {}
    for name, pred, proba in [('V1', v1_p, v1_proba), ('V3', v3_p, v3_proba), ('V4', v4_p, v4_proba)]:
        metrics[name] = {
            'Accuracy': accuracy_score(y, pred),
            'Brier Score': brier_score_loss(y, proba),
            'Log Loss': log_loss(y, proba),
            'AUC-ROC': roc_auc_score(y, proba),
        }

    print(f"\n  {'Metric':<15} {'V1 (Ridge)':<15} {'V3 (Stacking)':<15} {'V4 (Logistic)':<15} {'Best':<8}")
    print("  " + "-" * 68)
    for metric in ['Accuracy', 'Brier Score', 'Log Loss', 'AUC-ROC']:
        v1_val = metrics['V1'][metric]
        v3_val = metrics['V3'][metric]
        v4_val = metrics['V4'][metric]

        if metric == 'Accuracy' or metric == 'AUC-ROC':
            best = 'V4' if v4_val >= max(v1_val, v3_val) else ('V3' if v3_val >= v1_val else 'V1')
        else:
            best = 'V4' if v4_val <= min(v1_val, v3_val) else ('V3' if v3_val <= v1_val else 'V1')

        print(f"  {metric:<15} {v1_val:<15.4f} {v3_val:<15.4f} {v4_val:<15.4f} {best:<8}")

    # 6. 통계적 유의성 검정
    print("\n[6/6] Statistical Significance Tests")
    print("\n  McNemar Test (p < 0.05 = significant):")
    print("  " + "-" * 50)

    for name1, name2, pred1, pred2 in [
        ('V1', 'V3', v1_p, v3_p),
        ('V1', 'V4', v1_p, v4_p),
        ('V3', 'V4', v3_p, v4_p)
    ]:
        chi2, p = mcnemar_test(y, pred1, pred2)
        sig = "Yes*" if p < 0.05 else "No"
        print(f"  {name1} vs {name2}: χ² = {chi2:.4f}, p = {p:.4f}, Significant: {sig}")

    # Bootstrap CI
    print("\n  Bootstrap 95% CI for Accuracy:")
    print("  " + "-" * 50)
    for name, pred in [('V1', v1_p), ('V3', v3_p), ('V4', v4_p)]:
        correct = (pred == y).astype(int)
        mean, lower, upper = bootstrap_ci(correct)
        print(f"  {name}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

    # 신뢰도별 성능
    print("\n  High Confidence (75%+) Performance:")
    print("  " + "-" * 50)
    for name, proba, pred in [('V1', v1_proba, v1_p), ('V3', v3_proba, v3_p), ('V4', v4_proba, v4_p)]:
        mask = (proba >= 0.75) | (proba <= 0.25)
        if mask.sum() > 0:
            acc = accuracy_score(y[mask], pred[mask])
            cov = mask.mean()
            print(f"  {name}: {acc:.2%} accuracy, {cov:.1%} coverage ({mask.sum()} games)")

    # 결과 저장
    output_dir = PROJECT_ROOT / "data" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'n_samples': len(merged),
        'metrics': {
            'V1': {k: float(v) for k, v in metrics['V1'].items()},
            'V3': {k: float(v) for k, v in metrics['V3'].items()},
            'V4': {k: float(v) for k, v in metrics['V4'].items()},
        }
    }

    with open(output_dir / "v4_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"\n  V4 Accuracy: {metrics['V4']['Accuracy']:.2%}")
    print(f"  V3 Accuracy: {metrics['V3']['Accuracy']:.2%}")
    print(f"  V1 Accuracy: {metrics['V1']['Accuracy']:.2%}")
    print(f"\n  V4 vs V1: +{(metrics['V4']['Accuracy'] - metrics['V1']['Accuracy'])*100:.2f}%p")
    print(f"  V4 vs V3: +{(metrics['V4']['Accuracy'] - metrics['V3']['Accuracy'])*100:.2f}%p")

    print("\n" + "=" * 70)
    print("  Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
