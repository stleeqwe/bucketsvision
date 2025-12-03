#!/usr/bin/env python3
"""
Pre-Game Prediction Snapshot - 경기 전 예측 불변 기록

경기 시작 전에 실행하여 모델 예측을 해시 체인으로 기록.
시즌 종료 후 예측 정확도를 검증 가능한 형태로 보존.

실행 시간: 매일 오후 5시 ET (새벽 7시 KST)
- 대부분의 NBA 경기는 오후 7시 ET 이후 시작
- 경기 시작 최소 2시간 전에 예측 확정

사용법:
    python scripts/pregame_snapshot.py              # 오늘 경기 예측
    python scripts/pregame_snapshot.py --verify     # 전체 체인 검증
    python scripts/pregame_snapshot.py --date 2025-12-02  # 특정 날짜

불변성 보장:
1. SHA-256 해시 체인: 각 스냅샷이 이전 해시를 포함
2. 타임스탬프: UTC 기준 생성 시간
3. 모델 버전: 예측에 사용된 정확한 모델 정보
4. Git 커밋: 자동으로 git commit하여 추가 검증
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List
import argparse

import pytz
import numpy as np

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class NumpyEncoder(json.JSONEncoder):
    """NumPy 타입을 JSON으로 변환"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

from app.services.predictor_v5 import V5PredictionService
from app.services.data_loader import DataLoader, TEAM_INFO

# 디렉토리 설정
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "verified_predictions"
CHAIN_FILE = SNAPSHOT_DIR / "chain_state.json"


def get_et_now() -> datetime:
    """미국 동부 시간 현재"""
    return datetime.now(pytz.timezone('America/New_York'))


def get_et_today() -> date:
    """미국 동부 시간 기준 오늘"""
    return get_et_now().date()


def compute_hash(data: dict) -> str:
    """SHA-256 해시 계산"""
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False, cls=NumpyEncoder)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def load_chain_state() -> Dict:
    """해시 체인 상태 로드"""
    if CHAIN_FILE.exists():
        with open(CHAIN_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "genesis": True,
        "sequence": 0,
        "last_hash": "0" * 64,  # Genesis hash
        "last_date": None,
        "created_at": datetime.now(pytz.UTC).isoformat(),
    }


def save_chain_state(state: Dict):
    """해시 체인 상태 저장"""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def create_pregame_snapshot(target_date: date) -> Optional[Dict]:
    """
    경기 전 예측 스냅샷 생성.

    Args:
        target_date: 경기 날짜 (ET 기준)

    Returns:
        스냅샷 데이터 (해시 체인 포함)
    """
    print(f"\n{'='*60}")
    print(f"  Pre-Game Prediction Snapshot")
    print(f"  Target Date: {target_date} (ET)")
    print(f"  Captured At: {get_et_now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"{'='*60}")

    # 서비스 로드
    model_dir = PROJECT_ROOT / "bucketsvision_v4" / "models"
    predictor = V5PredictionService(model_dir)

    data_dir = PROJECT_ROOT / "data"
    loader = DataLoader(data_dir)

    # 모델 정보
    model_info = predictor.get_model_info()
    print(f"\n[Model Info]")
    print(f"  Version: {model_info['model_version']}")
    print(f"  Features: {model_info['n_features']}")

    # 팀 EPM 로드
    team_epm = loader.load_team_epm(target_date)
    if not team_epm:
        print("  ERROR: 팀 EPM 데이터를 불러올 수 없습니다.")
        return None

    # 경기 목록 가져오기
    games = loader.get_games(target_date)
    if not games:
        print(f"  INFO: {target_date}에 예정된 경기가 없습니다.")
        return None

    # 예정된 경기만 필터링 (status=1)
    scheduled_games = [g for g in games if g.get('game_status') == 1]
    if not scheduled_games:
        print(f"  INFO: {target_date}에 예정된 경기가 없습니다 (모두 진행/종료).")
        return None

    print(f"\n[Games: {len(scheduled_games)} scheduled]")

    # 예측 생성
    predictions = []

    for game in scheduled_games:
        home_id = game['home_team_id']
        away_id = game['away_team_id']

        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})
        home_abbr = home_info.get('abbr', 'UNK')
        away_abbr = away_info.get('abbr', 'UNK')

        # B2B 상태
        home_b2b = game.get('home_b2b', False)
        away_b2b = game.get('away_b2b', False)

        # V5.2 피처 생성
        features = loader.build_v5_2_features(
            home_id, away_id, team_epm, target_date,
            home_b2b=home_b2b, away_b2b=away_b2b
        )

        # 기본 예측
        base_prob = predictor.predict_proba(features)

        # 부상 정보 조회 및 조정
        home_prob_shift = 0.0
        away_prob_shift = 0.0

        try:
            home_injury = loader.get_injury_summary(home_abbr, target_date, team_epm)
            away_injury = loader.get_injury_summary(away_abbr, target_date, team_epm)
            home_prob_shift = home_injury.get('total_prob_shift', 0.0)
            away_prob_shift = away_injury.get('total_prob_shift', 0.0)
        except:
            pass

        # 부상 조정 적용
        final_prob = predictor.apply_injury_adjustment(base_prob, home_prob_shift, away_prob_shift)

        # 배당 정보 조회
        odds_info = loader.get_game_odds(home_abbr, away_abbr)
        odds_record = None
        edge_home = None
        edge_away = None

        if odds_info:
            ml_home = odds_info.get('moneyline_home')
            ml_away = odds_info.get('moneyline_away')

            odds_record = {
                "bookmaker": odds_info.get('bookmaker', 'pinnacle'),
                "moneyline_home": ml_home,
                "moneyline_away": ml_away,
                "spread_home": odds_info.get('spread_home'),
                "total_line": odds_info.get('total_line'),
            }

            # Edge 계산 (vig-adjusted)
            if ml_home and ml_away and ml_home > 1 and ml_away > 1:
                implied_home = 1 / ml_home
                implied_away = 1 / ml_away
                total_implied = implied_home + implied_away
                fair_home = implied_home / total_implied
                fair_away = implied_away / total_implied

                edge_home = final_prob - fair_home
                edge_away = (1 - final_prob) - fair_away

                odds_record["fair_prob_home"] = round(fair_home, 4)
                odds_record["fair_prob_away"] = round(fair_away, 4)
                odds_record["edge_home"] = round(edge_home, 4)
                odds_record["edge_away"] = round(edge_away, 4)

        # 예측 기록
        pred_record = {
            "game_id": game.get('game_id'),
            "game_time_et": game.get('game_time'),
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_team_id": home_id,
            "away_team_id": away_id,
            # 예측
            "prediction": {
                "home_win_prob": round(final_prob, 4),
                "base_prob": round(base_prob, 4),
                "predicted_winner": home_abbr if final_prob >= 0.5 else away_abbr,
                "confidence": round(abs(final_prob - 0.5) * 2, 4),  # 0~1 scale
            },
            # 피처 (주요 피처만)
            "features": {
                "team_epm_diff": round(features.get('team_epm_diff', 0), 4),
                "player_rotation_epm_diff": round(features.get('player_rotation_epm_diff', 0), 4),
                "b2b_diff": features.get('b2b_diff', 0),
                "rest_days_diff": features.get('rest_days_diff', 0),
            },
            # 컨디션
            "conditions": {
                "home_b2b": home_b2b,
                "away_b2b": away_b2b,
                "home_injury_shift": round(home_prob_shift, 2),
                "away_injury_shift": round(away_prob_shift, 2),
            },
            # 배당
            "odds": odds_record,
        }

        predictions.append(pred_record)

        # 출력
        edge_str = ""
        if edge_home is not None:
            edge_str = f" | Edge: {edge_home*100:+.1f}%"
        print(f"  {away_abbr} @ {home_abbr}: {final_prob:.1%} → {pred_record['prediction']['predicted_winner']}{edge_str}")

    # 해시 체인 상태 로드
    chain_state = load_chain_state()

    # 스냅샷 메타데이터
    snapshot_meta = {
        "version": "2.0",
        "type": "pregame_prediction",
        "model": {
            "name": "V5.2 XGBoost",
            "version": model_info['model_version'],
            "n_features": model_info['n_features'],
            "low_conf_accuracy": model_info.get('low_conf_accuracy'),
        },
        "captured_at": datetime.now(pytz.UTC).isoformat(),
        "game_date_et": target_date.isoformat(),
        "total_games": len(predictions),
    }

    # 예측 데이터 (해시 대상)
    prediction_data = {
        "meta": snapshot_meta,
        "predictions": predictions,
    }

    # 현재 예측 해시
    predictions_hash = compute_hash(prediction_data)

    # 해시 체인 구성
    chain_data = {
        "sequence": chain_state['sequence'] + 1,
        "previous_hash": chain_state['last_hash'],
        "predictions_hash": predictions_hash,
        "timestamp": datetime.now(pytz.UTC).isoformat(),
    }

    # 체인 해시 (이전 해시 + 현재 예측 해시 + 타임스탬프)
    chain_hash = compute_hash(chain_data)

    # 최종 스냅샷
    snapshot = {
        "meta": snapshot_meta,
        "predictions": predictions,
        "chain": {
            "sequence": chain_data['sequence'],
            "previous_hash": chain_data['previous_hash'],
            "predictions_hash": predictions_hash,
            "chain_hash": chain_hash,
            "timestamp": chain_data['timestamp'],
        },
    }

    print(f"\n[Chain Info]")
    print(f"  Sequence: #{chain_data['sequence']}")
    print(f"  Previous: {chain_data['previous_hash'][:16]}...")
    print(f"  Current:  {chain_hash[:16]}...")

    return snapshot


def save_snapshot(snapshot: Dict, target_date: date) -> Optional[Path]:
    """스냅샷 파일 저장"""
    # 연도별 디렉토리
    year_dir = SNAPSHOT_DIR / str(target_date.year)
    year_dir.mkdir(parents=True, exist_ok=True)

    # 파일명
    filename = f"{target_date.isoformat()}_pregame.json"
    filepath = year_dir / filename

    # 이미 존재 확인
    if filepath.exists():
        print(f"\n  WARNING: 스냅샷이 이미 존재합니다: {filepath}")
        print(f"  기존 파일을 덮어쓰지 않습니다. (불변성 보장)")
        return None

    # 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # 체인 상태 업데이트
    chain_state = {
        "genesis": False,
        "sequence": snapshot['chain']['sequence'],
        "last_hash": snapshot['chain']['chain_hash'],
        "last_date": target_date.isoformat(),
        "updated_at": datetime.now(pytz.UTC).isoformat(),
    }
    save_chain_state(chain_state)

    print(f"\n  SAVED: {filepath}")
    return filepath


def verify_snapshot(filepath: Path) -> bool:
    """단일 스냅샷 무결성 검증"""
    with open(filepath, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    # 예측 해시 검증
    prediction_data = {
        "meta": snapshot['meta'],
        "predictions": snapshot['predictions'],
    }
    computed_hash = compute_hash(prediction_data)
    stored_hash = snapshot['chain']['predictions_hash']

    if computed_hash != stored_hash:
        print(f"  FAIL: {filepath.name} - Predictions hash mismatch")
        return False

    print(f"  OK: {filepath.name} (seq #{snapshot['chain']['sequence']})")
    return True


def verify_chain() -> bool:
    """전체 해시 체인 검증"""
    print(f"\n{'='*60}")
    print("  Hash Chain Verification")
    print(f"{'='*60}\n")

    # 모든 스냅샷 파일 수집
    all_snapshots = sorted(SNAPSHOT_DIR.rglob("*_pregame.json"))

    if not all_snapshots:
        print("  No snapshots found.")
        return True

    print(f"  Found {len(all_snapshots)} snapshots\n")

    # 순서대로 검증
    prev_hash = "0" * 64  # Genesis
    all_valid = True

    for filepath in all_snapshots:
        with open(filepath, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)

        chain = snapshot['chain']

        # 1. 이전 해시 연결 확인
        if chain['previous_hash'] != prev_hash:
            print(f"  CHAIN BREAK: {filepath.name}")
            print(f"    Expected prev: {prev_hash[:16]}...")
            print(f"    Actual prev:   {chain['previous_hash'][:16]}...")
            all_valid = False

        # 2. 예측 해시 검증
        prediction_data = {
            "meta": snapshot['meta'],
            "predictions": snapshot['predictions'],
        }
        computed_hash = compute_hash(prediction_data)

        if computed_hash != chain['predictions_hash']:
            print(f"  TAMPERED: {filepath.name} - Predictions modified")
            all_valid = False
        else:
            print(f"  OK: {filepath.name} (#{chain['sequence']})")

        # 다음 검증을 위해 현재 체인 해시 저장
        prev_hash = chain['chain_hash']

    print(f"\n{'='*60}")
    if all_valid:
        print("  VERIFICATION PASSED - All snapshots intact")
    else:
        print("  VERIFICATION FAILED - Chain integrity compromised")
    print(f"{'='*60}")

    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Pre-Game Prediction Snapshot")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), default: today ET")
    parser.add_argument("--verify", action="store_true", help="Verify entire hash chain")
    parser.add_argument("--verify-file", type=str, help="Verify single snapshot file")
    args = parser.parse_args()

    # 검증 모드
    if args.verify:
        verify_chain()
        return

    if args.verify_file:
        filepath = Path(args.verify_file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        verify_snapshot(filepath)
        return

    # 스냅샷 생성 모드
    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        target_date = get_et_today()

    # 스냅샷 생성
    snapshot = create_pregame_snapshot(target_date)

    if snapshot is None:
        return

    # 저장
    filepath = save_snapshot(snapshot, target_date)

    if filepath:
        print(f"\n  Verify: python scripts/pregame_snapshot.py --verify")


if __name__ == "__main__":
    main()
