#!/usr/bin/env python3
"""
Pinnacle 배당 자동 캡처 스크립트.

경기 시작 전 여러 시점에 배당을 저장하여 CLV 분석에 활용.
- 3시간 전: 베팅 결정용
- 1시간 전: 분석용
- 직전 (Closing): CLV 비교용
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import pytz

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.data_loader import DataLoader, TEAM_INFO


def get_et_now() -> datetime:
    """미국 동부 시간 현재"""
    et = pytz.timezone('America/New_York')
    return datetime.now(et)


def get_today_games(loader: DataLoader) -> List[Dict]:
    """오늘 경기 목록 조회"""
    et_now = get_et_now()
    today = et_now.date()
    return loader.get_games(today)


def capture_current_odds(loader: DataLoader) -> Dict:
    """
    현재 모든 경기의 배당 캡처.

    Returns:
        {
            "captured_at": "2025-12-01T10:00:00-05:00",
            "games": [
                {
                    "home_team": "LAL",
                    "away_team": "BOS",
                    "game_time": "7:30 pm ET",
                    "odds": {...}
                }
            ]
        }
    """
    et_now = get_et_now()
    games = get_today_games(loader)

    if not games:
        return {
            "captured_at": et_now.isoformat(),
            "games": []
        }

    captured_games = []

    for game in games:
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})

        home_abbr = home_info.get("abbr", "UNK")
        away_abbr = away_info.get("abbr", "UNK")

        # 배당 조회
        odds_info = loader.get_game_odds(home_abbr, away_abbr)

        game_record = {
            "game_id": game.get("game_id", ""),
            "home_team": home_abbr,
            "away_team": away_abbr,
            "game_time": game.get("game_time", ""),
            "game_status": game.get("game_status", 1),
        }

        if odds_info:
            game_record["odds"] = {
                "bookmaker": odds_info.get("bookmaker", ""),
                "moneyline_home": odds_info.get("moneyline_home"),
                "moneyline_away": odds_info.get("moneyline_away"),
                "spread_home": odds_info.get("spread_home"),
                "spread_away": odds_info.get("spread_away"),
                "total_line": odds_info.get("total_line"),
            }
        else:
            game_record["odds"] = None

        captured_games.append(game_record)

    return {
        "captured_at": et_now.isoformat(),
        "game_date": et_now.date().isoformat(),
        "games": captured_games
    }


def save_odds_snapshot(data: Dict, label: str = "") -> Path:
    """
    배당 스냅샷 저장.

    Args:
        data: 캡처된 데이터
        label: 라벨 (예: "3h_before", "1h_before", "closing")
    """
    game_date = data.get("game_date", date.today().isoformat())

    # 저장 디렉토리
    odds_dir = project_root / "data" / "odds_history" / game_date
    odds_dir.mkdir(parents=True, exist_ok=True)

    # 파일명
    timestamp = datetime.now().strftime("%H%M")
    if label:
        filename = f"{game_date}_{label}_{timestamp}.json"
    else:
        filename = f"{game_date}_{timestamp}.json"

    filepath = odds_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 배당 저장: {filepath}")
    return filepath


def get_games_starting_soon(loader: DataLoader, hours: float = 3.0) -> List[Dict]:
    """지정 시간 내 시작하는 경기 조회"""
    et_now = get_et_now()
    games = get_today_games(loader)

    starting_soon = []

    for game in games:
        game_time_str = game.get("game_time", "")
        # 간단한 시간 파싱 (예: "7:30 pm ET")
        try:
            # 시간 문자열 파싱
            time_part = game_time_str.replace(" ET", "").strip()
            if "pm" in time_part.lower():
                hour_min = time_part.lower().replace("pm", "").strip()
                hour, minute = map(int, hour_min.split(":"))
                if hour != 12:
                    hour += 12
            else:
                hour_min = time_part.lower().replace("am", "").strip()
                hour, minute = map(int, hour_min.split(":"))
                if hour == 12:
                    hour = 0

            game_datetime = et_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            time_until_game = (game_datetime - et_now).total_seconds() / 3600

            if 0 < time_until_game <= hours:
                game["hours_until"] = round(time_until_game, 1)
                starting_soon.append(game)

        except Exception:
            continue

    return starting_soon


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pinnacle 배당 캡처")
    parser.add_argument("--label", type=str, default="", help="라벨 (예: 3h_before)")
    parser.add_argument("--check", action="store_true", help="경기 시작 임박 여부 확인")
    args = parser.parse_args()

    print("=" * 60)
    print("🏀 Pinnacle 배당 캡처")
    print("=" * 60)

    # 데이터 로더
    data_dir = project_root / "data"
    loader = DataLoader(data_dir)

    et_now = get_et_now()
    print(f"현재 시간 (ET): {et_now.strftime('%Y-%m-%d %H:%M')}")

    if args.check:
        # 경기 시작 임박 여부만 확인
        games_3h = get_games_starting_soon(loader, 3.0)
        games_1h = get_games_starting_soon(loader, 1.0)

        print(f"\n3시간 내 시작: {len(games_3h)}경기")
        print(f"1시간 내 시작: {len(games_1h)}경기")

        for game in games_3h:
            home_id = game["home_team_id"]
            away_id = game["away_team_id"]
            home_abbr = TEAM_INFO.get(home_id, {}).get("abbr", "UNK")
            away_abbr = TEAM_INFO.get(away_id, {}).get("abbr", "UNK")
            print(f"  - {away_abbr} @ {home_abbr} ({game.get('hours_until', '?')}h)")

        return

    # 배당 캡처
    print("\n배당 조회 중...")
    data = capture_current_odds(loader)

    games_with_odds = sum(1 for g in data["games"] if g.get("odds"))
    print(f"총 경기: {len(data['games'])}")
    print(f"배당 있음: {games_with_odds}")

    if data["games"]:
        # 저장
        label = args.label or datetime.now().strftime("%H%M")
        save_odds_snapshot(data, label)

        # 요약 출력
        print("\n📊 배당 요약:")
        for game in data["games"]:
            odds = game.get("odds")
            if odds:
                ml_home = odds.get("moneyline_home", "-")
                ml_away = odds.get("moneyline_away", "-")
                spread = odds.get("spread_home", "-")
                print(f"  {game['away_team']} @ {game['home_team']}: "
                      f"ML {ml_home}/{ml_away}, Spread {spread}")
            else:
                print(f"  {game['away_team']} @ {game['home_team']}: 배당 없음")
    else:
        print("오늘 경기가 없습니다.")


if __name__ == "__main__":
    main()
