#!/usr/bin/env python3
"""
Phase 3: 피처 효과 vs 모델 효과 분리 실험.

성능 차이가 피처 때문인지 모델 때문인지 분리합니다.

실험 매트릭스:
| 실험 | 피처   | 모델     | 목적            |
|-----|--------|----------|----------------|
| A   | V1 16개 | Ridge    | V1 베이스라인    |
| B   | V1 16개 | XGBoost  | 모델 효과        |
| C   | V3 26개 | Ridge    | 피처 효과        |
| D   | V3 26개 | XGBoost  | 피처+모델 효과   |

실행: python scripts/benchmark/feature_vs_model_experiment.py
"""

import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bucketsvision_v3" / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import xgboost as xgb


def load_data():
    """데이터 로드"""
    v2_data_dir = PROJECT_ROOT / "bucketsvision_v2" / "data"

    train_games = pd.read_parquet(v2_data_dir / "train_games.parquet")
    val_games = pd.read_parquet(v2_data_dir / "val_games.parquet")
    train_boxscores = pd.read_parquet(v2_data_dir / "train_boxscores.parquet")
    val_boxscores = pd.read_parquet(v2_data_dir / "val_boxscores.parquet")
    team_stats = pd.read_parquet(v2_data_dir / "team_stats.parquet")

    all_boxscores = pd.concat([train_boxscores, val_boxscores], ignore_index=True)

    return train_games, val_games, all_boxscores, team_stats


def build_v1_features(games_df, team_stats_df):
    """V1 16개 EPM 피처"""
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
    """V3 26개 피처"""
    from features.feature_builder import V3FeatureBuilder

    builder = V3FeatureBuilder()
    X_df = builder.build_features(games_df, boxscores_df, team_stats_df, verbose=False)
    return X_df


def margin_to_probability(margin, k=12.0):
    """점수차 → 확률 변환"""
    return 1 / (1 + np.exp(-margin / k))


def train_ridge_classifier(X_train, y_train, X_val, y_val):
    """Ridge 회귀 기반 분류 (점수차 예측 → 확률 변환)"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Ridge로 점수차 예측 (y를 margin으로 변환)
    # 분류 문제이므로 직접 확률 예측이 필요
    # 여기서는 LogisticRegression 사용
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = model.predict(X_val_scaled)

    return y_prob, y_pred


def train_xgboost_classifier(X_train, y_train, X_val, y_val):
    """XGBoost 분류"""
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    return y_prob, y_pred


def evaluate(y_true, y_prob, y_pred):
    """평가 메트릭"""
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'brier_score': brier_score_loss(y_true, y_prob_clipped),
        'auc_roc': roc_auc_score(y_true, y_prob_clipped),
    }


def main():
    print("=" * 70)
    print("  Phase 3: Feature vs Model Effect Separation")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1/5] Loading Data")
    train_games, val_games, all_boxscores, team_stats = load_data()

    y_train = train_games['home_win'].values
    y_val = val_games['home_win'].values

    print(f"  Train: {len(train_games)}, Val: {len(val_games)}")

    # 2. V1 피처 생성
    print("\n[2/5] Building V1 Features (16 EPM features)")
    X_train_v1 = build_v1_features(train_games, team_stats)
    X_val_v1 = build_v1_features(val_games, team_stats)

    # game_id로 정렬하여 일치시키기
    train_game_ids_v1 = set(X_train_v1['game_id'].values)
    val_game_ids_v1 = set(X_val_v1['game_id'].values)

    # 일치하는 게임만 사용
    train_mask = train_games['game_id'].isin(train_game_ids_v1)
    val_mask = val_games['game_id'].isin(val_game_ids_v1)

    train_games_filtered = train_games[train_mask].reset_index(drop=True)
    val_games_filtered = val_games[val_mask].reset_index(drop=True)

    y_train = train_games_filtered['home_win'].values
    y_val = val_games_filtered['home_win'].values

    v1_feature_cols = [c for c in X_train_v1.columns if c != 'game_id']
    X_train_v1_arr = X_train_v1[v1_feature_cols].fillna(0).values
    X_val_v1_arr = X_val_v1[v1_feature_cols].fillna(0).values

    print(f"  Features: {len(v1_feature_cols)}")
    print(f"  Train samples (filtered): {len(X_train_v1_arr)}")
    print(f"  Val samples (filtered): {len(X_val_v1_arr)}")

    # 3. V3 피처 생성
    print("\n[3/5] Building V3 Features (26 features)")

    # Train용 박스스코어 (train 데이터만)
    train_boxscores = all_boxscores[all_boxscores['game_date'] < val_games_filtered['game_date'].min()]

    X_train_v3 = build_v3_features(train_games_filtered, train_boxscores, team_stats)
    X_val_v3 = build_v3_features(val_games_filtered, all_boxscores, team_stats)

    v3_feature_cols = [c for c in X_train_v3.columns if c != 'game_id']

    # V3 피처도 동일한 순서로 정렬
    X_train_v3_sorted = X_train_v3.sort_values('game_id').reset_index(drop=True)
    X_val_v3_sorted = X_val_v3.sort_values('game_id').reset_index(drop=True)

    X_train_v3_arr = X_train_v3_sorted[v3_feature_cols].fillna(0).values
    X_val_v3_arr = X_val_v3_sorted[v3_feature_cols].fillna(0).values

    print(f"  Features: {len(v3_feature_cols)}")
    print(f"  Train samples: {len(X_train_v3_arr)}")
    print(f"  Val samples: {len(X_val_v3_arr)}")

    # 4. 실험 실행
    print("\n[4/5] Running Experiments")
    print("\n  Experiment Matrix:")
    print("  " + "-" * 60)
    print(f"  {'Exp':<5} {'Features':<15} {'Model':<15} {'Purpose':<20}")
    print("  " + "-" * 60)
    print(f"  {'A':<5} {'V1 (16)':<15} {'Ridge/Log':<15} {'Baseline':<20}")
    print(f"  {'B':<5} {'V1 (16)':<15} {'XGBoost':<15} {'Model Effect':<20}")
    print(f"  {'C':<5} {'V3 (26)':<15} {'Ridge/Log':<15} {'Feature Effect':<20}")
    print(f"  {'D':<5} {'V3 (26)':<15} {'XGBoost':<15} {'Both Effects':<20}")

    results = {}

    # Exp A: V1 피처 + Ridge/Logistic
    print("\n  Running Experiment A (V1 + Ridge/Log)...")
    y_prob_a, y_pred_a = train_ridge_classifier(X_train_v1_arr, y_train, X_val_v1_arr, y_val)
    results['A'] = evaluate(y_val, y_prob_a, y_pred_a)
    results['A']['features'] = 'V1 (16)'
    results['A']['model'] = 'Ridge/Log'

    # Exp B: V1 피처 + XGBoost
    print("  Running Experiment B (V1 + XGBoost)...")
    y_prob_b, y_pred_b = train_xgboost_classifier(X_train_v1_arr, y_train, X_val_v1_arr, y_val)
    results['B'] = evaluate(y_val, y_prob_b, y_pred_b)
    results['B']['features'] = 'V1 (16)'
    results['B']['model'] = 'XGBoost'

    # Exp C: V3 피처 + Ridge/Logistic
    print("  Running Experiment C (V3 + Ridge/Log)...")
    y_prob_c, y_pred_c = train_ridge_classifier(X_train_v3_arr, y_train, X_val_v3_arr, y_val)
    results['C'] = evaluate(y_val, y_prob_c, y_pred_c)
    results['C']['features'] = 'V3 (26)'
    results['C']['model'] = 'Ridge/Log'

    # Exp D: V3 피처 + XGBoost
    print("  Running Experiment D (V3 + XGBoost)...")
    y_prob_d, y_pred_d = train_xgboost_classifier(X_train_v3_arr, y_train, X_val_v3_arr, y_val)
    results['D'] = evaluate(y_val, y_prob_d, y_pred_d)
    results['D']['features'] = 'V3 (26)'
    results['D']['model'] = 'XGBoost'

    # 5. 결과 분석
    print("\n[5/5] Results Analysis")
    print("\n" + "=" * 70)
    print("  Experiment Results")
    print("=" * 70)

    print(f"\n  {'Exp':<5} {'Features':<12} {'Model':<12} {'Accuracy':<12} {'Brier':<12} {'AUC-ROC':<12}")
    print("  " + "-" * 65)
    for exp, r in results.items():
        print(f"  {exp:<5} {r['features']:<12} {r['model']:<12} {r['accuracy']:.4f}       {r['brier_score']:.4f}       {r['auc_roc']:.4f}")

    # 효과 분석
    print("\n" + "=" * 70)
    print("  Effect Analysis")
    print("=" * 70)

    # 모델 효과 (같은 피처, 다른 모델)
    model_effect_v1 = results['B']['accuracy'] - results['A']['accuracy']
    model_effect_v3 = results['D']['accuracy'] - results['C']['accuracy']

    # 피처 효과 (같은 모델, 다른 피처)
    feature_effect_ridge = results['C']['accuracy'] - results['A']['accuracy']
    feature_effect_xgb = results['D']['accuracy'] - results['B']['accuracy']

    print(f"\n  Model Effect (XGBoost vs Ridge/Log):")
    print(f"    With V1 features: {model_effect_v1:+.4f} ({model_effect_v1*100:+.2f}%p)")
    print(f"    With V3 features: {model_effect_v3:+.4f} ({model_effect_v3*100:+.2f}%p)")
    print(f"    Average model effect: {(model_effect_v1 + model_effect_v3)/2:+.4f}")

    print(f"\n  Feature Effect (V3 vs V1):")
    print(f"    With Ridge/Log: {feature_effect_ridge:+.4f} ({feature_effect_ridge*100:+.2f}%p)")
    print(f"    With XGBoost:   {feature_effect_xgb:+.4f} ({feature_effect_xgb*100:+.2f}%p)")
    print(f"    Average feature effect: {(feature_effect_ridge + feature_effect_xgb)/2:+.4f}")

    # 상호작용 효과
    interaction = (results['D']['accuracy'] - results['C']['accuracy']) - (results['B']['accuracy'] - results['A']['accuracy'])

    print(f"\n  Interaction Effect (Feature × Model):")
    print(f"    {interaction:+.4f} ({interaction*100:+.2f}%p)")

    # 실제 모델과 비교
    print("\n" + "=" * 70)
    print("  Comparison with Actual Models")
    print("=" * 70)

    # 저장된 결과 로드
    with open(PROJECT_ROOT / "data" / "benchmark" / "evaluation_results.json", "r") as f:
        eval_results = json.load(f)

    v1_actual = eval_results['metrics']['v1']['accuracy']
    v2_actual = eval_results['metrics']['v2']['accuracy']
    v3_actual = eval_results['metrics']['v3']['accuracy']

    print(f"\n  {'Model':<25} {'Accuracy':<12} {'Brier':<12}")
    print("  " + "-" * 50)
    print(f"  {'V1 (Ridge, 16 feat)':<25} {v1_actual:.4f}       {eval_results['metrics']['v1']['brier_score']:.4f}")
    print(f"  {'V2 (XGBoost, 15 feat)':<25} {v2_actual:.4f}       {eval_results['metrics']['v2']['brier_score']:.4f}")
    print(f"  {'V3 (Stacking, 26 feat)':<25} {v3_actual:.4f}       {eval_results['metrics']['v3']['brier_score']:.4f}")
    print("  " + "-" * 50)
    print(f"  {'Exp A (Ridge, 16 feat)':<25} {results['A']['accuracy']:.4f}       {results['A']['brier_score']:.4f}")
    print(f"  {'Exp B (XGB, 16 feat)':<25} {results['B']['accuracy']:.4f}       {results['B']['brier_score']:.4f}")
    print(f"  {'Exp C (Ridge, 26 feat)':<25} {results['C']['accuracy']:.4f}       {results['C']['brier_score']:.4f}")
    print(f"  {'Exp D (XGB, 26 feat)':<25} {results['D']['accuracy']:.4f}       {results['D']['brier_score']:.4f}")

    # 결론
    print("\n" + "=" * 70)
    print("  Conclusions")
    print("=" * 70)

    if abs(model_effect_v1) > abs(feature_effect_ridge) and abs(model_effect_v3) > abs(feature_effect_xgb):
        print("\n  > Model choice has MORE impact than feature set")
    elif abs(model_effect_v1) < abs(feature_effect_ridge) and abs(model_effect_v3) < abs(feature_effect_xgb):
        print("\n  > Feature set has MORE impact than model choice")
    else:
        print("\n  > Effects are mixed - depends on the combination")

    if model_effect_v1 > 0 and model_effect_v3 > 0:
        print("  > XGBoost consistently outperforms Ridge/Logistic")
    elif model_effect_v1 < 0 and model_effect_v3 < 0:
        print("  > Ridge/Logistic consistently outperforms XGBoost")

    if feature_effect_ridge > 0 and feature_effect_xgb > 0:
        print("  > V3 features (26) consistently outperform V1 features (16)")
    elif feature_effect_ridge < 0 and feature_effect_xgb < 0:
        print("  > V1 features (16) consistently outperform V3 features (26)")

    # 결과 저장
    output_dir = PROJECT_ROOT / "data" / "benchmark"
    experiment_results = {
        'experiments': results,
        'effects': {
            'model_effect_v1_features': model_effect_v1,
            'model_effect_v3_features': model_effect_v3,
            'feature_effect_ridge': feature_effect_ridge,
            'feature_effect_xgboost': feature_effect_xgb,
            'interaction_effect': interaction,
        },
        'comparison': {
            'v1_actual': v1_actual,
            'v2_actual': v2_actual,
            'v3_actual': v3_actual,
        }
    }

    with open(output_dir / "feature_model_experiment.json", "w") as f:
        json.dump(experiment_results, f, indent=2, default=float)

    print(f"\n  Results saved to: {output_dir / 'feature_model_experiment.json'}")

    print("\n" + "=" * 70)
    print("  Phase 3 Complete!")
    print("=" * 70)

    return experiment_results


if __name__ == "__main__":
    main()
