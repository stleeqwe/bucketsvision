"""
시즌 전체 예측 결과 CSV 내보내기
"""

import sys
from pathlib import Path
from datetime import date, timedelta, datetime

import pandas as pd
import pytz
from scipy.stats import norm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.predictor_v4 import V4PredictionService
from app.services.data_loader import DataLoader, TEAM_INFO

# V4.4 B2B 보정 상수
B2B_WEIGHT = 3.0


def apply_b2b_correction(base_prob: float, home_b2b: bool, away_b2b: bool) -> float:
    """B2B 보정 적용"""
    b2b_simple = (1 if away_b2b else 0) - (1 if home_b2b else 0)
    if b2b_simple == 0:
        return base_prob
    b2b_margin = b2b_simple * B2B_WEIGHT
    prob_shift = norm.cdf(b2b_margin / 12.0) - 0.5
    return min(max(base_prob + prob_shift, 0.01), 0.99)


def get_et_today() -> date:
    """미국 동부 시간 기준 오늘 날짜"""
    et = pytz.timezone('America/New_York')
    return datetime.now(et).date()


def main():
    print("=" * 60)
    print("BucketsVision 2025-26 시즌 예측 결과 CSV 내보내기")
    print("=" * 60)

    # 설정
    season_start = date(2025, 10, 22)
    et_today = get_et_today()

    print(f"\n기간: {season_start} ~ {et_today}")

    # 서비스 로드
    model_dir = project_root / "bucketsvision_v4" / "models"
    predictor = V4PredictionService(model_dir, version="4.3")

    data_dir = project_root / "data"
    loader = DataLoader(data_dir)

    # 팀 EPM 로드
    print("\n팀 EPM 데이터 로딩...")
    team_epm = loader.load_team_epm(et_today)
    if not team_epm:
        print("ERROR: 팀 EPM 데이터를 불러올 수 없습니다.")
        return

    # 결과 저장 리스트
    results = []

    # 날짜별 경기 처리
    current_date = season_start
    total_days = (et_today - season_start).days + 1

    print(f"\n{total_days}일간의 경기 데이터 수집 중...")

    day_count = 0
    while current_date <= et_today:
        day_count += 1

        # 진행률 표시
        if day_count % 7 == 0:
            print(f"  진행: {day_count}/{total_days}일 ({day_count/total_days*100:.0f}%)")

        games = loader.get_games(current_date)

        if games:
            for game in games:
                home_id = game["home_team_id"]
                away_id = game["away_team_id"]

                home_info = TEAM_INFO.get(home_id, {})
                away_info = TEAM_INFO.get(away_id, {})

                home_abbr = home_info.get("abbr", "UNK")
                away_abbr = away_info.get("abbr", "UNK")

                # 피처 생성 및 예측
                features = loader.build_v4_3_features(home_id, away_id, team_epm, current_date)
                base_prob = predictor.predict_proba(features)

                # B2B 보정
                home_b2b = game.get("home_b2b", False)
                away_b2b = game.get("away_b2b", False)
                home_win_prob = apply_b2b_correction(base_prob, home_b2b, away_b2b)

                # 확률 범위 제한 (0.01 ~ 0.99)
                home_win_prob = min(max(home_win_prob, 0.01), 0.99)

                # 마진 계산
                raw_margin = norm.ppf(home_win_prob) * 12.0
                if abs(home_win_prob - 0.5) > 0.25:
                    predicted_margin = raw_margin * 0.85
                else:
                    predicted_margin = raw_margin

                # 예측 승자
                predicted_winner = home_abbr if home_win_prob >= 0.5 else away_abbr

                # 경기 결과
                game_status = game.get("game_status", 1)
                home_score = game.get("home_score")
                away_score = game.get("away_score")

                # 실제 결과 (종료된 경기만)
                actual_winner = None
                actual_margin = None
                is_correct = None
                margin_error = None

                if game_status == 3 and home_score is not None and away_score is not None:
                    actual_winner = home_abbr if home_score > away_score else away_abbr
                    actual_margin = home_score - away_score
                    is_correct = predicted_winner == actual_winner
                    margin_error = abs(predicted_margin - actual_margin)

                # 결과 추가
                results.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "date_kst": (current_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "game_time": game.get("game_time", ""),
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "home_b2b": home_b2b,
                    "away_b2b": away_b2b,
                    "home_win_prob": round(home_win_prob * 100, 1),
                    "away_win_prob": round((1 - home_win_prob) * 100, 1),
                    "predicted_margin": round(predicted_margin, 1),
                    "predicted_winner": predicted_winner,
                    "game_status": "종료" if game_status == 3 else ("진행중" if game_status == 2 else "예정"),
                    "home_score": home_score,
                    "away_score": away_score,
                    "actual_winner": actual_winner,
                    "actual_margin": actual_margin,
                    "is_correct": is_correct,
                    "margin_error": round(margin_error, 1) if margin_error is not None else None,
                })

        current_date += timedelta(days=1)

    # DataFrame 생성
    df = pd.DataFrame(results)

    # 통계 계산
    finished_df = df[df["game_status"] == "종료"]
    total_games = len(df)
    finished_games = len(finished_df)
    correct_games = finished_df["is_correct"].sum() if finished_games > 0 else 0
    accuracy = correct_games / finished_games * 100 if finished_games > 0 else 0
    mae = finished_df["margin_error"].mean() if finished_games > 0 else 0

    print(f"\n{'=' * 60}")
    print("통계 요약")
    print(f"{'=' * 60}")
    print(f"전체 경기: {total_games}")
    print(f"종료 경기: {finished_games}")
    print(f"적중: {correct_games}")
    print(f"적중률: {accuracy:.1f}%")
    print(f"평균 오차 (MAE): {mae:.1f}pt")

    # CSV 저장
    output_path = project_root / "data" / "predictions" / "season_2025_26_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV 저장 완료: {output_path}")

    # 요약 통계도 저장
    summary = {
        "generated_at": datetime.now().isoformat(),
        "season": "2025-26",
        "date_range": f"{season_start} ~ {et_today}",
        "total_games": total_games,
        "finished_games": finished_games,
        "correct_predictions": int(correct_games),
        "accuracy_pct": round(accuracy, 2),
        "mae": round(mae, 2),
    }

    import json
    summary_path = output_path.parent / "season_2025_26_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 요약 저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
