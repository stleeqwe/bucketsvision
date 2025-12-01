"""
일일 예측 결과 스냅샷 저장 (수정 불가 검증용)

매일 경기 종료 후 실행하여 예측 결과를 해시 검증 가능한 형태로 저장.
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

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


def compute_hash(data: dict) -> str:
    """데이터의 SHA256 해시 계산"""
    # 정렬된 JSON으로 변환하여 일관된 해시 생성
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def create_daily_snapshot(target_date: Optional[date] = None) -> dict:
    """
    특정 날짜의 예측 결과 스냅샷 생성.

    Args:
        target_date: 대상 날짜 (None이면 어제 ET 기준)

    Returns:
        스냅샷 데이터 (해시 포함)
    """
    et_today = get_et_today()

    # 기본값: 어제 경기 (오늘 새벽에 종료된 경기)
    if target_date is None:
        target_date = et_today - timedelta(days=1)

    print(f"📅 스냅샷 대상 날짜: {target_date} (ET)")

    # 서비스 로드
    model_dir = project_root / "bucketsvision_v4" / "models"
    predictor = V4PredictionService(model_dir, version="4.3")

    data_dir = project_root / "data"
    loader = DataLoader(data_dir)

    # 팀 EPM 로드
    team_epm = loader.load_team_epm(et_today)
    if not team_epm:
        raise RuntimeError("팀 EPM 데이터를 불러올 수 없습니다.")

    # 경기 가져오기
    games = loader.get_games(target_date)

    if not games:
        print(f"⚠️ {target_date}에 경기가 없습니다.")
        return None

    # 예측 결과 수집
    predictions = []
    finished_count = 0
    correct_count = 0

    for game in games:
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})

        home_abbr = home_info.get("abbr", "UNK")
        away_abbr = away_info.get("abbr", "UNK")

        # 피처 생성 및 예측
        features = loader.build_v4_3_features(home_id, away_id, team_epm, target_date)
        base_prob = predictor.predict_proba(features)

        # B2B 보정
        home_b2b = game.get("home_b2b", False)
        away_b2b = game.get("away_b2b", False)
        home_win_prob = apply_b2b_correction(base_prob, home_b2b, away_b2b)
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

        # 실제 결과
        actual_winner = None
        is_correct = None

        if game_status == 3 and home_score is not None and away_score is not None:
            finished_count += 1
            actual_winner = home_abbr if home_score > away_score else away_abbr
            is_correct = predicted_winner == actual_winner
            if is_correct:
                correct_count += 1

        # 배당 정보 조회 (Pinnacle)
        odds_info = loader.get_game_odds(home_abbr, away_abbr)
        odds_record = None
        if odds_info:
            odds_record = {
                "bookmaker": odds_info.get("bookmaker", ""),
                "moneyline_home": odds_info.get("moneyline_home"),
                "moneyline_away": odds_info.get("moneyline_away"),
                "spread_home": odds_info.get("spread_home"),
                "spread_away": odds_info.get("spread_away"),
                "total_line": odds_info.get("total_line"),
            }

            # 머니라인 Edge 계산
            ml_home = odds_info.get("moneyline_home")
            ml_away = odds_info.get("moneyline_away")
            if ml_home and ml_away and ml_home > 1 and ml_away > 1:
                # Implied probability (vig 제거)
                implied_home = 1 / ml_home
                implied_away = 1 / ml_away
                total_implied = implied_home + implied_away
                fair_home = implied_home / total_implied
                fair_away = implied_away / total_implied

                # Edge 계산
                edge_home = home_win_prob - fair_home
                edge_away = (1 - home_win_prob) - fair_away

                odds_record["market_prob_home"] = round(fair_home * 100, 1)
                odds_record["market_prob_away"] = round(fair_away * 100, 1)
                odds_record["edge_home"] = round(edge_home * 100, 1)
                odds_record["edge_away"] = round(edge_away * 100, 1)

        pred_record = {
            "game_id": game.get("game_id", ""),
            "game_time": game.get("game_time", ""),
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,
            "home_win_prob": round(home_win_prob * 100, 1),
            "predicted_margin": round(predicted_margin, 1),
            "predicted_winner": predicted_winner,
            "game_status": game_status,
            "home_score": home_score,
            "away_score": away_score,
            "actual_winner": actual_winner,
            "is_correct": is_correct,
            "odds": odds_record,  # 배당 정보 추가
        }
        predictions.append(pred_record)

    # 요약 통계
    accuracy = round(correct_count / finished_count * 100, 1) if finished_count > 0 else None

    # 스냅샷 데이터 구성
    snapshot_data = {
        "meta": {
            "version": "1.0",
            "model": "V4.4 (Logistic + Player EPM + B2B)",
            "created_at": datetime.now(pytz.UTC).isoformat(),
            "game_date_et": target_date.isoformat(),
            "game_date_kst": (target_date + timedelta(days=1)).isoformat(),
        },
        "summary": {
            "total_games": len(predictions),
            "finished_games": finished_count,
            "correct_predictions": correct_count,
            "accuracy_pct": accuracy,
        },
        "predictions": predictions,
    }

    # 해시 계산 (predictions 부분만)
    predictions_hash = compute_hash({"predictions": predictions})
    snapshot_data["integrity"] = {
        "hash_algorithm": "SHA256",
        "predictions_hash": predictions_hash,
    }

    return snapshot_data


def save_snapshot(snapshot: dict, target_date: date) -> Path:
    """스냅샷을 파일로 저장"""
    # 저장 디렉토리
    snapshot_dir = project_root / "data" / "snapshots" / str(target_date.year)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 파일명: YYYY-MM-DD_snapshot.json
    filename = f"{target_date.isoformat()}_snapshot.json"
    filepath = snapshot_dir / filename

    # 이미 존재하면 경고
    if filepath.exists():
        print(f"⚠️ 스냅샷이 이미 존재합니다: {filepath}")
        print("   기존 파일을 덮어쓰지 않습니다.")
        return None

    # JSON 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"✅ 스냅샷 저장: {filepath}")
    return filepath


def verify_snapshot(filepath: Path) -> bool:
    """스냅샷 무결성 검증"""
    with open(filepath, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    # 저장된 해시
    stored_hash = snapshot.get("integrity", {}).get("predictions_hash")
    if not stored_hash:
        print("❌ 해시 정보가 없습니다.")
        return False

    # 현재 데이터로 해시 재계산
    current_hash = compute_hash({"predictions": snapshot["predictions"]})

    if stored_hash == current_hash:
        print(f"✅ 무결성 검증 통과: {filepath.name}")
        return True
    else:
        print(f"❌ 무결성 검증 실패: {filepath.name}")
        print(f"   저장된 해시: {stored_hash[:16]}...")
        print(f"   현재 해시:   {current_hash[:16]}...")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="일일 예측 결과 스냅샷 생성/검증")
    parser.add_argument("--date", type=str, help="대상 날짜 (YYYY-MM-DD, 기본: 어제)")
    parser.add_argument("--verify", type=str, help="검증할 스냅샷 파일 경로")
    parser.add_argument("--verify-all", action="store_true", help="모든 스냅샷 검증")
    args = parser.parse_args()

    # 검증 모드
    if args.verify:
        filepath = Path(args.verify)
        if not filepath.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
            return
        verify_snapshot(filepath)
        return

    # 전체 검증 모드
    if args.verify_all:
        snapshot_dir = project_root / "data" / "snapshots"
        if not snapshot_dir.exists():
            print("❌ 스냅샷 디렉토리가 없습니다.")
            return

        all_valid = True
        for filepath in sorted(snapshot_dir.rglob("*_snapshot.json")):
            if not verify_snapshot(filepath):
                all_valid = False

        if all_valid:
            print("\n✅ 모든 스냅샷이 검증을 통과했습니다.")
        else:
            print("\n❌ 일부 스냅샷이 검증에 실패했습니다.")
        return

    # 스냅샷 생성 모드
    print("=" * 60)
    print("BucketsVision 일일 스냅샷 생성")
    print("=" * 60)

    # 대상 날짜 파싱
    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        target_date = get_et_today() - timedelta(days=1)

    # 스냅샷 생성
    snapshot = create_daily_snapshot(target_date)

    if snapshot is None:
        return

    # 결과 출력
    summary = snapshot["summary"]
    print(f"\n{'=' * 60}")
    print("스냅샷 요약")
    print(f"{'=' * 60}")
    print(f"전체 경기: {summary['total_games']}")
    print(f"종료 경기: {summary['finished_games']}")
    print(f"적중: {summary['correct_predictions']}")
    print(f"적중률: {summary['accuracy_pct']}%")
    print(f"해시: {snapshot['integrity']['predictions_hash'][:32]}...")

    # 파일 저장
    filepath = save_snapshot(snapshot, target_date)

    if filepath:
        print("\n📌 스냅샷 검증 명령어:")
        print(f"   python scripts/daily_snapshot.py --verify {filepath}")


if __name__ == "__main__":
    main()
