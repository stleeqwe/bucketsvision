#!/usr/bin/env python3
"""
V4 + recent_opp_strength_diff 테스트.

V4 기존 10개 피처에 상대강도 피처 추가 시 성능 변화 확인.

실행: python scripts/experiments/v4_plus_opp_strength.py
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
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def load_data():
    """데이터 로드 및 상대팀 ID 추출"""
    v2_data_dir = PROJECT_ROOT / "bucketsvision_v2" / "data"

    train_games = pd.read_parquet(v2_data_dir / "train_games.parquet")
    train_boxscores = pd.read_parquet(v2_data_dir / "train_boxscores.parquet")
    val_games = pd.read_parquet(v2_data_dir / "val_games.parquet")
    val_boxscores = pd.read_parquet(v2_data_dir / "val_boxscores.parquet")
    team_stats = pd.read_parquet(v2_data_dir / "team_stats.parquet")

    all_boxscores = pd.concat([train_boxscores, val_boxscores], ignore_index=True)

    # 팀 약어 → ID 매핑
    team_mapping = all_boxscores[['team_abbr', 'team_id']].drop_duplicates()
    team_abbr_to_id = dict(zip(team_mapping['team_abbr'], team_mapping['team_id']))

    def extract_opp_team_id(row):
        matchup = row['matchup']
        team_abbr = row['team_abbr']
        if ' @ ' in matchup:
            parts = matchup.split(' @ ')
        elif ' vs. ' in matchup:
            parts = matchup.split(' vs. ')
        else:
            return None
        opp_abbr = parts[1] if parts[0] == team_abbr else parts[0]
        return team_abbr_to_id.get(opp_abbr)

    all_boxscores['opp_team_id'] = all_boxscores.apply(extract_opp_team_id, axis=1)
    train_boxscores['opp_team_id'] = train_boxscores.apply(extract_opp_team_id, axis=1)

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


def get_opponent_epm(team_stats_df, opp_team_id, game_date):
    if pd.isna(opp_team_id):
        return 0.0
    opp_epm = team_stats_df[
        (team_stats_df['team_id'] == opp_team_id) &
        (team_stats_df['game_dt'] < game_date)
    ].sort_values('game_dt', ascending=False)
    if len(opp_epm) == 0:
        return 0.0
    return opp_epm.iloc[0].get('team_epm', 0.0)


def calc_avg_opp_strength(box_df, team_stats_df, game_date):
    """최근 N경기 평균 상대 강도"""
    if len(box_df) == 0:
        return 0.0
    strengths = []
    for _, row in box_df.iterrows():
        opp_id = row.get('opp_team_id')
        if pd.notna(opp_id):
            strength = get_opponent_epm(team_stats_df, opp_id, game_date)
            strengths.append(strength)
    return np.mean(strengths) if strengths else 0.0


def build_v4_plus_features(games_df, boxscores_df, team_stats_df, window=5):
    """V4 피처 + recent_opp_strength_diff"""
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
        h_win5 = (h_window['result'] == 'W').mean() if len(h_window) > 0 else 0.5
        a_win5 = (a_window['result'] == 'W').mean() if len(a_window) > 0 else 0.5
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

        # === 새로운 피처: 최근 상대강도 ===
        h_opp_str = calc_avg_opp_strength(h_window, team_stats_df, game_date)
        a_opp_str = calc_avg_opp_strength(a_window, team_stats_df, game_date)
        feat['recent_opp_strength_diff'] = h_opp_str - a_opp_str

        features_list.append(feat)

    return pd.DataFrame(features_list)


def evaluate_model(X_train, y_train, X_val, y_val, model_name):
    """모델 학습 및 평가"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Calibrated Logistic Regression
    base_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    model.fit(X_train_scaled, y_train)

    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
        'brier_score': brier_score_loss(y_val, val_proba),
        'log_loss': log_loss(y_val, val_proba),
        'auc_roc': roc_auc_score(y_val, val_proba),
    }

    # 신뢰도별 성능
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = (val_proba >= threshold) | (val_proba <= (1 - threshold))
        if mask.sum() > 0:
            metrics[f'conf_{int(threshold*100)}_acc'] = accuracy_score(y_val[mask], val_pred[mask])
            metrics[f'conf_{int(threshold*100)}_cov'] = mask.mean()

    return model, scaler, metrics


def main():
    print("=" * 70)
    print("  V4 + recent_opp_strength_diff 테스트")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/4] 데이터 로드")
    train_games, val_games, train_boxscores, all_boxscores, team_stats = load_data()

    # 피처 빌드
    print("\n[2/4] 피처 빌드")
    print("  Building training features...")
    X_train_df = build_v4_plus_features(train_games, train_boxscores, team_stats)
    print("  Building validation features...")
    X_val_df = build_v4_plus_features(val_games, all_boxscores, team_stats)

    # 레이블 매칭
    train_labels = train_games.set_index('game_id')['home_win']
    val_labels = val_games.set_index('game_id')['home_win']

    X_train_df = X_train_df.set_index('game_id')
    X_val_df = X_val_df.set_index('game_id')

    common_train = X_train_df.index.intersection(train_labels.index)
    common_val = X_val_df.index.intersection(val_labels.index)

    y_train = train_labels.loc[common_train].values
    y_val = val_labels.loc[common_val].values

    # NaN 제거
    X_train_full = X_train_df.loc[common_train]
    X_val_full = X_val_df.loc[common_val]

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

    v4_plus_features = v4_features + ['recent_opp_strength_diff']

    # 테스트
    print("\n[3/4] 모델 비교")
    print("=" * 70)

    results = {}

    # V4 Original
    print("\n  V4 Original (10 features)")
    X_train_v4 = X_train_full[v4_features].values
    X_val_v4 = X_val_full[v4_features].values
    _, _, metrics_v4 = evaluate_model(X_train_v4, y_train, X_val_v4, y_val, "V4")
    results['V4 (10 features)'] = metrics_v4
    print(f"    Accuracy: {metrics_v4['accuracy']:.4f}")
    print(f"    Brier: {metrics_v4['brier_score']:.4f}")
    print(f"    AUC-ROC: {metrics_v4['auc_roc']:.4f}")

    # V4 + opp_strength
    print("\n  V4 + recent_opp_strength_diff (11 features)")
    X_train_v4p = X_train_full[v4_plus_features].values
    X_val_v4p = X_val_full[v4_plus_features].values
    model_v4p, scaler_v4p, metrics_v4p = evaluate_model(X_train_v4p, y_train, X_val_v4p, y_val, "V4+")
    results['V4+ (11 features)'] = metrics_v4p
    print(f"    Accuracy: {metrics_v4p['accuracy']:.4f}")
    print(f"    Brier: {metrics_v4p['brier_score']:.4f}")
    print(f"    AUC-ROC: {metrics_v4p['auc_roc']:.4f}")

    # 결과 비교
    print("\n[4/4] 결과 비교")
    print("=" * 70)

    print(f"\n  {'Metric':<20} {'V4 (10)':<15} {'V4+ (11)':<15} {'Diff':<12}")
    print("  " + "-" * 60)

    for metric in ['accuracy', 'brier_score', 'log_loss', 'auc_roc']:
        v4_val = metrics_v4[metric]
        v4p_val = metrics_v4p[metric]
        diff = v4p_val - v4_val

        if metric in ['brier_score', 'log_loss']:
            better = "↓" if diff < 0 else "↑"
        else:
            better = "↑" if diff > 0 else "↓"

        print(f"  {metric:<20} {v4_val:<15.4f} {v4p_val:<15.4f} {diff:+.4f} {better}")

    # 신뢰도별 비교
    print(f"\n  신뢰도별 정확도:")
    print("  " + "-" * 50)
    print(f"  {'Confidence':<12} {'V4':<12} {'V4+':<12} {'Diff':<12}")
    print("  " + "-" * 50)

    for conf in [55, 60, 65, 70, 75]:
        acc_key = f'conf_{conf}_acc'
        cov_key = f'conf_{conf}_cov'
        if acc_key in metrics_v4 and acc_key in metrics_v4p:
            v4_acc = metrics_v4[acc_key]
            v4p_acc = metrics_v4p[acc_key]
            v4p_cov = metrics_v4p[cov_key]
            diff = v4p_acc - v4_acc
            print(f"  {conf}%+         {v4_acc:<12.2%} {v4p_acc:<12.2%} {diff:+.2%}")

    # 최종 판정
    print("\n" + "=" * 70)
    print("  최종 판정")
    print("=" * 70)

    acc_diff = metrics_v4p['accuracy'] - metrics_v4['accuracy']
    brier_diff = metrics_v4p['brier_score'] - metrics_v4['brier_score']

    if acc_diff > 0.005:  # 0.5%p 이상 개선
        verdict = "V4+ 채택 권장"
        reason = f"정확도 {acc_diff*100:+.2f}%p 개선"
    elif acc_diff < -0.005:
        verdict = "V4 유지 권장"
        reason = f"정확도 {acc_diff*100:+.2f}%p 하락"
    else:
        if brier_diff < -0.002:
            verdict = "V4+ 채택 고려"
            reason = f"정확도 유사, Brier {brier_diff:+.4f} 개선"
        else:
            verdict = "V4 유지 권장"
            reason = "유의미한 개선 없음"

    print(f"\n  결론: {verdict}")
    print(f"  사유: {reason}")

    print(f"\n  V4:  {metrics_v4['accuracy']:.2%} accuracy")
    print(f"  V4+: {metrics_v4p['accuracy']:.2%} accuracy")
    print(f"  차이: {acc_diff*100:+.2f}%p")

    return results


if __name__ == "__main__":
    main()
