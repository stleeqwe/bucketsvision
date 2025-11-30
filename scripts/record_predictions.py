#!/usr/bin/env python3
"""
예측 기록 스크립트.

오늘 경기에 대한 예측을 기록합니다.
경기 시작 1시간 전 기준으로 예측을 저장합니다.

사용법:
    # 오늘 경기 예측 기록
    python scripts/record_predictions.py

    # 특정 날짜 경기 예측 기록
    python scripts/record_predictions.py --date 2025-11-30

    # 과거 경기 일괄 기록 (백필)
    python scripts/record_predictions.py --backfill --start 2025-10-22 --end 2025-11-29

자동 실행 (cron):
    # 매일 오후 5시 (ET) - 대부분 경기 시작 2시간 전
    0 17 * * * cd /path/to/bucketsvision && python scripts/record_predictions.py
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, date, timedelta
import pytz

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.data_loader import DataLoader
from app.services.predictor_v4 import V4PredictionService
from app.services.prediction_tracker import PredictionTracker
from config.constants import TEAM_INFO
from src.utils.helpers import get_season_from_date
from scipy.stats import norm


def record_predictions_for_date(
    target_date: date,
    tracker: PredictionTracker,
    loader: DataLoader,
    predictor: V4PredictionService,
    force: bool = False
) -> int:
    """
    특정 날짜의 경기 예측 기록.

    Args:
        target_date: 대상 날짜
        tracker: 예측 추적기
        loader: 데이터 로더
        predictor: 예측 서비스
        force: 이미 기록된 경기도 덮어쓰기

    Returns:
        기록된 경기 수
    """
    print(f"\n[{target_date}] 예측 기록 시작")

    # 경기 로드
    games = loader.get_games(target_date)
    if not games:
        print(f"  경기 없음")
        return 0

    print(f"  경기 수: {len(games)}")

    # 팀 EPM 로드
    team_epm = loader.load_team_epm(target_date)

    # 시즌 확인
    season = get_season_from_date(target_date)

    # 기존 기록 로드
    existing_records = tracker.load_season_records(season)

    # 예측 및 기록
    recorded = 0
    model_info = predictor.get_model_info()
    model_version = model_info.get('model_version', '4.3.0')

    for game in games:
        game_id = game['game_id']

        # 이미 기록된 경기 건너뛰기 (force 옵션 없으면)
        if game_id in existing_records and not force:
            print(f"  [{game_id}] 이미 기록됨, 건너뛰기")
            continue

        home_id = game['home_team_id']
        away_id = game['away_team_id']

        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})

        home_abbr = home_info.get('abbr', 'UNK')
        away_abbr = away_info.get('abbr', 'UNK')

        # V4.3 피처 생성
        features = loader.build_v4_3_features(home_id, away_id, team_epm, target_date)

        # 예측
        home_win_prob = predictor.predict_proba(features)
        predicted_margin = norm.ppf(home_win_prob) * 12.0  # 확률 -> 마진 근사

        # 기록
        tracker.record_prediction(
            game_id=game_id,
            game_date=target_date.strftime('%Y-%m-%d'),
            game_time_et=game.get('game_time', ''),
            home_team=home_abbr,
            away_team=away_abbr,
            home_team_id=home_id,
            away_team_id=away_id,
            home_win_prob=home_win_prob,
            predicted_margin=predicted_margin,
            model_version=model_version,
            features=features,
            season=season
        )

        predicted_winner = home_abbr if home_win_prob >= 0.5 else away_abbr
        print(f"  [{game_id}] {away_abbr} @ {home_abbr} -> {predicted_winner} ({home_win_prob:.1%})")
        recorded += 1

    print(f"  기록 완료: {recorded}경기")
    return recorded


def backfill_predictions(
    start_date: date,
    end_date: date,
    tracker: PredictionTracker,
    loader: DataLoader,
    predictor: V4PredictionService
) -> int:
    """과거 날짜 예측 일괄 기록 (백필)"""
    print(f"\n{'='*60}")
    print(f"예측 백필: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    total_recorded = 0
    current_date = start_date

    while current_date <= end_date:
        recorded = record_predictions_for_date(
            current_date, tracker, loader, predictor
        )
        total_recorded += recorded
        current_date += timedelta(days=1)

        # 캐시 초기화 (메모리 관리)
        loader.clear_cache()

    print(f"\n{'='*60}")
    print(f"백필 완료: 총 {total_recorded}경기 기록")
    print(f"{'='*60}")

    return total_recorded


def main():
    parser = argparse.ArgumentParser(description='경기 예측 기록')
    parser.add_argument('--date', type=str, help='대상 날짜 (YYYY-MM-DD)')
    parser.add_argument('--backfill', action='store_true', help='과거 일괄 기록')
    parser.add_argument('--start', type=str, help='백필 시작 날짜')
    parser.add_argument('--end', type=str, help='백필 종료 날짜')
    parser.add_argument('--force', action='store_true', help='기존 기록 덮어쓰기')

    args = parser.parse_args()

    # 서비스 초기화
    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "bucketsvision_v4" / "models"

    loader = DataLoader(data_dir)
    predictor = V4PredictionService(model_dir, version="4.3")
    tracker = PredictionTracker(data_dir)

    # ET 기준 오늘 날짜
    et_tz = pytz.timezone('America/New_York')
    et_today = datetime.now(et_tz).date()

    if args.backfill:
        # 백필 모드
        if not args.start or not args.end:
            print("오류: --backfill 모드에서는 --start와 --end가 필요합니다")
            return

        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

        backfill_predictions(start_date, end_date, tracker, loader, predictor)

    else:
        # 단일 날짜 모드
        if args.date:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        else:
            target_date = et_today

        record_predictions_for_date(
            target_date, tracker, loader, predictor, force=args.force
        )

    # 현재 통계 출력
    season = get_season_from_date(et_today)
    stats = tracker.get_stats(season)
    print(f"\n[시즌 {season} 통계]")
    print(f"  총 예측: {stats['total_predictions']}경기")
    print(f"  완료: {stats['completed_games']}경기")
    print(f"  적중: {stats['correct_predictions']}경기 ({stats['accuracy']:.1%})")
    print(f"  대기: {stats['pending_games']}경기")


if __name__ == "__main__":
    main()
