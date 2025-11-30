#!/usr/bin/env python3
"""
V4.1 피처 가중치 최적화 실험.

Phase 1: 정규화 강도(C) 최적화
Phase 2: 정규화 방식 비교 (L1/L2/Elastic Net)
Phase 3: 가중치 민감도 분석
Phase 4: Bayesian Optimization (Optuna)

실행: python scripts/experiments/weight_optimization.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bucketsvision_v4" / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import json

from features.feature_builder import V4FeatureBuilder, get_v4_feature_names


def load_data():
    """데이터 로드"""
    v2_data_dir = PROJECT_ROOT / "bucketsvision_v2" / "data"

    train_games = pd.read_parquet(v2_data_dir / "train_games.parquet")
    train_boxscores = pd.read_parquet(v2_data_dir / "train_boxscores.parquet")
    val_games = pd.read_parquet(v2_data_dir / "val_games.parquet")
    val_boxscores = pd.read_parquet(v2_data_dir / "val_boxscores.parquet")
    team_stats = pd.read_parquet(v2_data_dir / "team_stats.parquet")

    all_boxscores = pd.concat([train_boxscores, val_boxscores], ignore_index=True)

    return train_games, val_games, train_boxscores, all_boxscores, team_stats


def build_features():
    """V4.1 피처 빌드"""
    print("  Loading data...")
    train_games, val_games, train_boxscores, all_boxscores, team_stats = load_data()

    print("  Building features...")
    builder = V4FeatureBuilder()
    X_train_df = builder.build_features(train_games, train_boxscores, team_stats, verbose=False)
    X_val_df = builder.build_features(val_games, all_boxscores, team_stats, verbose=False)

    # 레이블 매칭
    train_labels = train_games.set_index('game_id')['home_win']
    val_labels = val_games.set_index('game_id')['home_win']

    X_train_df = X_train_df.set_index('game_id')
    X_val_df = X_val_df.set_index('game_id')

    common_train = X_train_df.index.intersection(train_labels.index)
    common_val = X_val_df.index.intersection(val_labels.index)

    feature_names = get_v4_feature_names()

    X_train_raw = X_train_df.loc[common_train, feature_names]
    y_train_raw = train_labels.loc[common_train]
    X_val_raw = X_val_df.loc[common_val, feature_names]
    y_val_raw = val_labels.loc[common_val]

    # NaN 제거
    train_valid = ~X_train_raw.isna().any(axis=1)
    val_valid = ~X_val_raw.isna().any(axis=1)

    X_train = X_train_raw[train_valid].values
    y_train = y_train_raw[train_valid].values
    X_val = X_val_raw[val_valid].values
    y_val = y_val_raw[val_valid].values

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    return X_train, y_train, X_val, y_val, feature_names


def evaluate_model(model, X_train, y_train, X_val, y_val, scaler=None):
    """모델 평가"""
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    model.fit(X_train_scaled, y_train)

    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    return {
        'accuracy': accuracy_score(y_val, val_pred),
        'brier': brier_score_loss(y_val, val_proba),
        'auc': roc_auc_score(y_val, val_proba),
        'log_loss': log_loss(y_val, val_proba),
    }


# =============================================================================
# Phase 1: 정규화 강도(C) 최적화
# =============================================================================

def phase1_regularization_strength(X_train, y_train, X_val, y_val):
    """Phase 1: C 값 최적화"""
    print("\n" + "=" * 70)
    print("  Phase 1: 정규화 강도(C) 최적화")
    print("=" * 70)

    C_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    results = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42, solver='lbfgs')
        model.fit(X_train_scaled, y_train)

        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        acc = accuracy_score(y_val, val_pred)
        brier = brier_score_loss(y_val, val_proba)

        results.append({'C': C, 'accuracy': acc, 'brier': brier})
        print(f"  C={C:<8} Accuracy: {acc:.4f}, Brier: {brier:.4f}")

    # 최적 C 찾기
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n  Best C: {best['C']} (Accuracy: {best['accuracy']:.4f})")

    return best['C'], results


# =============================================================================
# Phase 2: 정규화 방식 비교
# =============================================================================

def phase2_regularization_type(X_train, y_train, X_val, y_val, best_C):
    """Phase 2: L1/L2/Elastic Net 비교"""
    print("\n" + "=" * 70)
    print("  Phase 2: 정규화 방식 비교")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    results = []

    # L2 (Ridge)
    model_l2 = LogisticRegression(C=best_C, penalty='l2', max_iter=1000, random_state=42, solver='lbfgs')
    model_l2.fit(X_train_scaled, y_train)
    proba_l2 = model_l2.predict_proba(X_val_scaled)[:, 1]
    acc_l2 = accuracy_score(y_val, (proba_l2 >= 0.5).astype(int))
    results.append({'type': 'L2 (Ridge)', 'accuracy': acc_l2, 'brier': brier_score_loss(y_val, proba_l2)})
    print(f"  L2 (Ridge):     Accuracy: {acc_l2:.4f}")

    # L1 (Lasso)
    model_l1 = LogisticRegression(C=best_C, penalty='l1', max_iter=1000, random_state=42, solver='saga')
    model_l1.fit(X_train_scaled, y_train)
    proba_l1 = model_l1.predict_proba(X_val_scaled)[:, 1]
    acc_l1 = accuracy_score(y_val, (proba_l1 >= 0.5).astype(int))
    results.append({'type': 'L1 (Lasso)', 'accuracy': acc_l1, 'brier': brier_score_loss(y_val, proba_l1)})
    print(f"  L1 (Lasso):     Accuracy: {acc_l1:.4f}")

    # L1에서 0이 된 피처 확인
    zero_features = np.sum(np.abs(model_l1.coef_[0]) < 0.001)
    print(f"    → L1이 제거한 피처 수: {zero_features}/11")

    # Elastic Net
    for l1_ratio in [0.2, 0.5, 0.8]:
        model_en = LogisticRegression(C=best_C, penalty='elasticnet', l1_ratio=l1_ratio,
                                       max_iter=1000, random_state=42, solver='saga')
        model_en.fit(X_train_scaled, y_train)
        proba_en = model_en.predict_proba(X_val_scaled)[:, 1]
        acc_en = accuracy_score(y_val, (proba_en >= 0.5).astype(int))
        results.append({'type': f'Elastic Net (α={l1_ratio})', 'accuracy': acc_en,
                        'brier': brier_score_loss(y_val, proba_en)})
        print(f"  Elastic Net (α={l1_ratio}): Accuracy: {acc_en:.4f}")

    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n  Best: {best['type']} (Accuracy: {best['accuracy']:.4f})")

    return best['type'], results


# =============================================================================
# Phase 3: 가중치 민감도 분석
# =============================================================================

def phase3_weight_sensitivity(X_train, y_train, X_val, y_val, feature_names, best_C):
    """Phase 3: 각 피처 가중치의 민감도 분석"""
    print("\n" + "=" * 70)
    print("  Phase 3: 가중치 민감도 분석")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 기본 모델 학습
    base_model = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    base_model.fit(X_train_scaled, y_train)
    base_proba = base_model.predict_proba(X_val_scaled)[:, 1]
    base_acc = accuracy_score(y_val, (base_proba >= 0.5).astype(int))

    print(f"\n  기준 정확도: {base_acc:.4f}")
    print(f"\n  피처별 민감도 (제거 시 정확도 변화):")
    print("  " + "-" * 50)

    sensitivity = []

    for i, feat in enumerate(feature_names):
        # 해당 피처 제거
        mask = [j for j in range(len(feature_names)) if j != i]
        X_train_reduced = X_train_scaled[:, mask]
        X_val_reduced = X_val_scaled[:, mask]

        model = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
        model.fit(X_train_reduced, y_train)
        proba = model.predict_proba(X_val_reduced)[:, 1]
        acc = accuracy_score(y_val, (proba >= 0.5).astype(int))

        diff = acc - base_acc
        sensitivity.append({'feature': feat, 'acc_without': acc, 'diff': diff})

    # 정렬 (제거 시 가장 큰 손실을 주는 피처 순)
    sensitivity.sort(key=lambda x: x['diff'])

    for s in sensitivity:
        direction = "↓" if s['diff'] < 0 else "↑"
        print(f"  {s['feature']:<25} {s['diff']*100:+.2f}%p {direction}")

    # 가장 중요한 피처와 불필요한 피처
    most_important = sensitivity[0]
    least_important = sensitivity[-1]

    print(f"\n  가장 중요: {most_important['feature']} (제거 시 {most_important['diff']*100:+.2f}%p)")
    print(f"  가장 불필요: {least_important['feature']} (제거 시 {least_important['diff']*100:+.2f}%p)")

    return sensitivity


# =============================================================================
# Phase 4: Bayesian Optimization
# =============================================================================

def phase4_bayesian_optimization(X_train, y_train, X_val, y_val, feature_names):
    """Phase 4: Optuna를 사용한 종합 최적화"""
    print("\n" + "=" * 70)
    print("  Phase 4: Bayesian Optimization (Optuna)")
    print("=" * 70)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed. Skipping Phase 4.")
        print("  Install with: pip install optuna")
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    def objective(trial):
        # 하이퍼파라미터
        C = trial.suggest_float('C', 0.01, 100.0, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])

        if penalty == 'elasticnet':
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
            solver = 'saga'
        elif penalty == 'l1':
            l1_ratio = None
            solver = 'saga'
        else:
            l1_ratio = None
            solver = 'lbfgs'

        # 피처별 스케일링 factor (선택적)
        use_feature_scaling = trial.suggest_categorical('use_feature_scaling', [True, False])

        if use_feature_scaling:
            feature_scales = []
            for feat in feature_names:
                scale = trial.suggest_float(f'scale_{feat}', 0.5, 2.0)
                feature_scales.append(scale)
            X_train_modified = X_train_scaled * np.array(feature_scales)
            X_val_modified = X_val_scaled * np.array(feature_scales)
        else:
            X_train_modified = X_train_scaled
            X_val_modified = X_val_scaled

        # 모델 학습
        if penalty == 'elasticnet':
            model = LogisticRegression(C=C, penalty=penalty, l1_ratio=l1_ratio,
                                       max_iter=1000, random_state=42, solver=solver)
        else:
            model = LogisticRegression(C=C, penalty=penalty,
                                       max_iter=1000, random_state=42, solver=solver)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_modified, y_train, cv=cv, scoring='accuracy')

        return scores.mean()

    study = optuna.create_study(direction='maximize')
    print("  Running optimization (100 trials)...")
    study.optimize(objective, n_trials=100, show_progress_bar=False)

    print(f"\n  Best trial:")
    print(f"    CV Accuracy: {study.best_value:.4f}")
    print(f"    Parameters:")
    for key, value in study.best_params.items():
        if not key.startswith('scale_'):
            print(f"      {key}: {value}")

    # 최적 파라미터로 최종 평가
    best_params = study.best_params
    C = best_params['C']
    penalty = best_params['penalty']

    if penalty == 'elasticnet':
        l1_ratio = best_params['l1_ratio']
        model = LogisticRegression(C=C, penalty=penalty, l1_ratio=l1_ratio,
                                   max_iter=1000, random_state=42, solver='saga')
    elif penalty == 'l1':
        model = LogisticRegression(C=C, penalty=penalty,
                                   max_iter=1000, random_state=42, solver='saga')
    else:
        model = LogisticRegression(C=C, penalty=penalty,
                                   max_iter=1000, random_state=42, solver='lbfgs')

    # 피처 스케일링 적용
    if best_params.get('use_feature_scaling', False):
        feature_scales = [best_params.get(f'scale_{feat}', 1.0) for feat in feature_names]
        X_train_final = X_train_scaled * np.array(feature_scales)
        X_val_final = X_val_scaled * np.array(feature_scales)
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
        feature_scales = None

    model.fit(X_train_final, y_train)
    val_proba = model.predict_proba(X_val_final)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    final_acc = accuracy_score(y_val, val_pred)

    print(f"\n  Final Validation Accuracy: {final_acc:.4f}")

    return {
        'best_params': best_params,
        'cv_accuracy': study.best_value,
        'val_accuracy': final_acc,
        'feature_scales': feature_scales,
        'model': model,
        'scaler': scaler
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  V4.1 피처 가중치 최적화")
    print("=" * 70)

    # 데이터 준비
    print("\n[0/4] 데이터 준비")
    X_train, y_train, X_val, y_val, feature_names = build_features()

    # 기준 성능 (V4.1 현재)
    print("\n  V4.1 기준 성능:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    base_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    base_model.fit(X_train_scaled, y_train)
    base_proba = base_model.predict_proba(X_val_scaled)[:, 1]
    base_acc = accuracy_score(y_val, (base_proba >= 0.5).astype(int))
    print(f"  Accuracy: {base_acc:.4f} (C=1.0, L2)")

    # Phase 1
    best_C, phase1_results = phase1_regularization_strength(X_train, y_train, X_val, y_val)

    # Phase 2
    best_type, phase2_results = phase2_regularization_type(X_train, y_train, X_val, y_val, best_C)

    # Phase 3
    sensitivity = phase3_weight_sensitivity(X_train, y_train, X_val, y_val, feature_names, best_C)

    # Phase 4
    optuna_result = phase4_bayesian_optimization(X_train, y_train, X_val, y_val, feature_names)

    # 최종 요약
    print("\n" + "=" * 70)
    print("  최종 요약")
    print("=" * 70)

    print(f"\n  V4.1 기준:     {base_acc:.4f}")
    print(f"  Phase 1 최적:  {max(r['accuracy'] for r in phase1_results):.4f} (C={best_C})")
    print(f"  Phase 2 최적:  {max(r['accuracy'] for r in phase2_results):.4f} ({best_type})")

    if optuna_result:
        print(f"  Phase 4 최적:  {optuna_result['val_accuracy']:.4f} (Optuna)")

        improvement = optuna_result['val_accuracy'] - base_acc
        print(f"\n  총 개선: {improvement*100:+.2f}%p")

        # 최적 파라미터 저장
        output = {
            'baseline_accuracy': float(base_acc),
            'optimized_accuracy': float(optuna_result['val_accuracy']),
            'best_C': float(optuna_result['best_params']['C']),
            'best_penalty': optuna_result['best_params']['penalty'],
            'best_params': {k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in optuna_result['best_params'].items()},
            'feature_sensitivity': [{
                'feature': s['feature'],
                'importance': float(-s['diff'])  # 제거 시 손실 = 중요도
            } for s in sensitivity]
        }

        output_path = PROJECT_ROOT / "data" / "benchmark" / "weight_optimization_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("  최적화 완료!")
    print("=" * 70)

    return optuna_result


if __name__ == "__main__":
    main()
