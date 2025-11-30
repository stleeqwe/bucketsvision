#!/usr/bin/env python3
"""
리바운드 피처 효과 실험.

공격 리바운드(ORB), 수비 리바운드(DRB) 피처가 예측력에 기여하는지 테스트.

실행: python scripts/experiments/rebound_feature_test.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


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


def get_team_history(boxscores_df, team_id, game_date):
    return boxscores_df[
        (boxscores_df['team_id'] == team_id) &
        (boxscores_df['game_date'] < game_date)
    ].sort_values('game_date', ascending=False)


def get_team_epm(team_stats_df, team_id, game_date):
    return team_stats_df[
        (team_stats_df['team_id'] == team_id) &
        (team_stats_df['game_dt'] < game_date)
    ].sort_values('game_dt', ascending=False)


def build_features_with_rebounds(games_df, boxscores_df, team_stats_df, window=10):
    """V4 피처 + 리바운드 피처"""
    features_list = []
    total = len(games_df)

    for idx, (_, game) in enumerate(games_df.iterrows()):
        if (idx + 1) % 500 == 0:
            print(f"    Progress: {idx + 1}/{total}")

        home_id = game['home_team_id']
        away_id = game['away_team_id']
        game_date = game['game_date']

        home_box = get_team_history(boxscores_df, home_id, game_date)
        away_box = get_team_history(boxscores_df, away_id, game_date)
        home_epm = get_team_epm(team_stats_df, home_id, game_date)
        away_epm = get_team_epm(team_stats_df, away_id, game_date)

        if len(home_epm) == 0 or len(away_epm) == 0:
            continue

        feat = {'game_id': game['game_id']}

        # === V4 기존 10개 피처 ===

        # 1. EPM (4개)
        feat['team_epm_diff'] = home_epm.iloc[0].get('team_epm', 0) - away_epm.iloc[0].get('team_epm', 0)
        feat['team_oepm_diff'] = home_epm.iloc[0].get('team_oepm', 0) - away_epm.iloc[0].get('team_oepm', 0)
        feat['team_depm_diff'] = home_epm.iloc[0].get('team_depm', 0) - away_epm.iloc[0].get('team_depm', 0)
        feat['sos_diff'] = home_epm.iloc[0].get('sos', 0) - away_epm.iloc[0].get('sos', 0)

        h_window = home_box.head(window)
        a_window = away_box.head(window)

        # 2. Four Factors (2개)
        def calc_efg(box):
            if len(box) == 0:
                return 0.50
            fg, fg3, fga = box['fg'].sum(), box['fg3'].sum(), box['fga'].sum()
            return (fg + 0.5 * fg3) / fga if fga > 0 else 0.50

        def calc_ft_rate(box):
            if len(box) == 0:
                return 0.20
            ft = box['ft'].sum() if 'ft' in box.columns else 0
            fga = box['fga'].sum()
            return ft / fga if fga > 0 else 0.20

        feat['efg_pct_diff'] = calc_efg(h_window) - calc_efg(a_window)
        feat['ft_rate_diff'] = calc_ft_rate(h_window) - calc_ft_rate(a_window)

        # 3. 모멘텀 (3개)
        h_last5 = home_box.head(5)
        a_last5 = away_box.head(5)
        h_win5 = (h_last5['result'] == 'W').mean() if len(h_last5) > 0 else 0.5
        a_win5 = (a_last5['result'] == 'W').mean() if len(a_last5) > 0 else 0.5
        feat['last5_win_pct_diff'] = h_win5 - a_win5

        def calc_streak(box):
            if len(box) == 0:
                return 0
            streak = 0
            first_result = box.iloc[0]['result']
            for _, row in box.iterrows():
                if row['result'] == first_result:
                    streak += 1 if first_result == 'W' else -1
                else:
                    break
            return min(max(streak, -10), 10)

        feat['streak_diff'] = calc_streak(home_box) - calc_streak(away_box)

        def calc_ewma_margin(box, span=5):
            if len(box) < 3:
                return 0.0
            return box.head(10)['margin'].ewm(span=span, adjust=False).mean().iloc[0]

        feat['margin_ewma_diff'] = calc_ewma_margin(home_box) - calc_ewma_margin(away_box)

        # 4. 컨텍스트 (1개)
        def calc_away_record(box):
            if len(box) == 0:
                return 0.45
            away_games = box[box['is_home'] == False]
            if len(away_games) == 0:
                return 0.45
            return (away_games['result'] == 'W').mean()

        feat['away_road_strength'] = calc_away_record(away_box) - 0.5

        # === 리바운드 피처 ===

        # 평균 리바운드
        h_orb = h_window['orb'].mean() if len(h_window) > 0 else 10
        a_orb = a_window['orb'].mean() if len(a_window) > 0 else 10
        h_drb = h_window['drb'].mean() if len(h_window) > 0 else 34
        a_drb = a_window['drb'].mean() if len(a_window) > 0 else 34
        h_trb = h_window['trb'].mean() if len(h_window) > 0 and 'trb' in h_window.columns else h_orb + h_drb
        a_trb = a_window['trb'].mean() if len(a_window) > 0 and 'trb' in a_window.columns else a_orb + a_drb

        feat['orb_diff'] = h_orb - a_orb  # 공격 리바운드 차이
        feat['drb_diff'] = h_drb - a_drb  # 수비 리바운드 차이
        feat['trb_diff'] = h_trb - a_trb  # 총 리바운드 차이

        # 리바운드 비율 (ORB%, DRB% 근사)
        # ORB% = ORB / (ORB + Opp_DRB) ≈ ORB / (ORB + DRB)
        def calc_orb_pct(box):
            if len(box) == 0:
                return 0.25
            orb = box['orb'].sum()
            drb = box['drb'].sum()
            total = orb + drb
            return orb / total if total > 0 else 0.25

        def calc_drb_pct(box):
            if len(box) == 0:
                return 0.75
            orb = box['orb'].sum()
            drb = box['drb'].sum()
            total = orb + drb
            return drb / total if total > 0 else 0.75

        feat['orb_pct_diff'] = calc_orb_pct(h_window) - calc_orb_pct(a_window)
        feat['drb_pct_diff'] = calc_drb_pct(h_window) - calc_drb_pct(a_window)

        # 2nd Chance Points 근사 (ORB 기반)
        # 공격리바운드가 많으면 세컨찬스 득점 기회 증가
        feat['second_chance_diff'] = feat['orb_diff']  # ORB와 동일하지만 의미론적으로 구분

        features_list.append(feat)

    return pd.DataFrame(features_list)


def evaluate_model(X_train, y_train, X_val, y_val):
    """모델 학습 및 평가"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    base_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    model.fit(X_train_scaled, y_train)

    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    # 피처 계수 (첫 번째 calibrator의 base estimator에서)
    try:
        coefs = model.calibrated_classifiers_[0].estimator.coef_[0]
    except:
        coefs = None

    return {
        'accuracy': accuracy_score(y_val, val_pred),
        'brier_score': brier_score_loss(y_val, val_proba),
        'auc_roc': roc_auc_score(y_val, val_proba),
        'coefficients': coefs
    }


def main():
    print("=" * 70)
    print("  리바운드 피처 효과 실험")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/3] 데이터 로드")
    train_games, val_games, train_boxscores, all_boxscores, team_stats = load_data()

    # 피처 빌드
    print("\n[2/3] 피처 빌드")
    print("  Building training features...")
    X_train_df = build_features_with_rebounds(train_games, train_boxscores, team_stats)
    print("  Building validation features...")
    X_val_df = build_features_with_rebounds(val_games, all_boxscores, team_stats)

    # 레이블 매칭
    train_labels = train_games.set_index('game_id')['home_win']
    val_labels = val_games.set_index('game_id')['home_win']

    X_train_df = X_train_df.set_index('game_id')
    X_val_df = X_val_df.set_index('game_id')

    common_train = X_train_df.index.intersection(train_labels.index)
    common_val = X_val_df.index.intersection(val_labels.index)

    y_train = train_labels.loc[common_train].values
    y_val = val_labels.loc[common_val].values

    X_train_full = X_train_df.loc[common_train]
    X_val_full = X_val_df.loc[common_val]

    # NaN 제거
    train_valid = ~X_train_full.isna().any(axis=1)
    val_valid = ~X_val_full.isna().any(axis=1)

    X_train_full = X_train_full[train_valid]
    X_val_full = X_val_full[val_valid]
    y_train = y_train[train_valid]
    y_val = y_val[val_valid]

    print(f"\n  Train: {len(y_train)}, Val: {len(y_val)}")

    # 피처 세트 정의
    v4_features = [
        'team_epm_diff', 'team_oepm_diff', 'team_depm_diff', 'sos_diff',
        'last5_win_pct_diff', 'efg_pct_diff', 'streak_diff',
        'margin_ewma_diff', 'ft_rate_diff', 'away_road_strength'
    ]

    rebound_features = ['orb_diff', 'drb_diff', 'trb_diff', 'orb_pct_diff', 'drb_pct_diff']

    # 테스트할 조합
    print("\n[3/3] 피처 조합 테스트")
    print("=" * 70)

    feature_sets = {
        '1. V4 Original (10)': v4_features,
        '2. V4 + ORB only': v4_features + ['orb_diff'],
        '3. V4 + DRB only': v4_features + ['drb_diff'],
        '4. V4 + TRB only': v4_features + ['trb_diff'],
        '5. V4 + ORB% only': v4_features + ['orb_pct_diff'],
        '6. V4 + ORB + DRB': v4_features + ['orb_diff', 'drb_diff'],
        '7. V4 + ORB% + DRB%': v4_features + ['orb_pct_diff', 'drb_pct_diff'],
        '8. V4 + All Rebounds': v4_features + rebound_features,
    }

    results = []

    for name, features in feature_sets.items():
        X_train = X_train_full[features].values
        X_val = X_val_full[features].values

        metrics = evaluate_model(X_train, y_train, X_val, y_val)
        metrics['name'] = name
        metrics['n_features'] = len(features)
        results.append(metrics)

        print(f"\n  {name}")
        print(f"    Accuracy: {metrics['accuracy']:.4f}, Brier: {metrics['brier_score']:.4f}")

    # 결과 요약
    print("\n" + "=" * 70)
    print("  결과 요약")
    print("=" * 70)

    baseline = results[0]['accuracy']

    print(f"\n  {'Feature Set':<30} {'Acc':<10} {'vs V4':<10} {'Brier':<10}")
    print("  " + "-" * 60)

    for r in results:
        diff = r['accuracy'] - baseline
        marker = "★" if r['accuracy'] == max(x['accuracy'] for x in results) else ""
        print(f"  {r['name']:<30} {r['accuracy']:<10.4f} {diff:+.4f}    {r['brier_score']:<10.4f} {marker}")

    # 리바운드 피처별 효과
    print("\n" + "=" * 70)
    print("  리바운드 피처별 효과 분석")
    print("=" * 70)

    print(f"\n  피처         V4 대비 효과")
    print("  " + "-" * 40)
    print(f"  ORB (공격)   {(results[1]['accuracy'] - baseline)*100:+.2f}%p")
    print(f"  DRB (수비)   {(results[2]['accuracy'] - baseline)*100:+.2f}%p")
    print(f"  TRB (총)     {(results[3]['accuracy'] - baseline)*100:+.2f}%p")
    print(f"  ORB%         {(results[4]['accuracy'] - baseline)*100:+.2f}%p")
    print(f"  ORB + DRB    {(results[5]['accuracy'] - baseline)*100:+.2f}%p")
    print(f"  ORB% + DRB%  {(results[6]['accuracy'] - baseline)*100:+.2f}%p")
    print(f"  All Rebounds {(results[7]['accuracy'] - baseline)*100:+.2f}%p")

    # 최적 조합의 계수 분석
    best_idx = max(range(len(results)), key=lambda i: results[i]['accuracy'])
    best = results[best_idx]

    if best['coefficients'] is not None and best_idx > 0:
        print(f"\n  Best Model ({best['name']}) 피처 계수:")
        features = list(feature_sets.values())[best_idx]
        coef_dict = dict(zip(features, best['coefficients']))

        # 리바운드 관련 피처만 표시
        print("  " + "-" * 40)
        for feat in features:
            if 'rb' in feat.lower() or 'rebound' in feat.lower():
                coef = coef_dict[feat]
                print(f"  {feat:<20} {coef:+.4f}")

    # 최종 판정
    print("\n" + "=" * 70)
    print("  최종 판정")
    print("=" * 70)

    best_acc = max(r['accuracy'] for r in results)
    best_result = [r for r in results if r['accuracy'] == best_acc][0]

    if best_result['name'] == '1. V4 Original (10)':
        print("\n  결론: 리바운드 피처 추가 효과 없음")
        print("  권장: V4 유지 (10개 피처)")
    else:
        improvement = best_acc - baseline
        if improvement > 0.005:
            print(f"\n  결론: {best_result['name']} 채택 권장")
            print(f"  개선: +{improvement*100:.2f}%p")
        else:
            print("\n  결론: 유의미한 개선 없음")
            print("  권장: V4 유지 (복잡성 대비 이득 미미)")

    print(f"\n  V4 Baseline: {baseline:.2%}")
    print(f"  Best Result: {best_acc:.2%}")

    return results


if __name__ == "__main__":
    main()
