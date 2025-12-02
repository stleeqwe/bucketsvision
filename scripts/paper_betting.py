#!/usr/bin/env python3
"""
Paper Betting Tracker - Edge 기반 가상 베팅 수익률 추적

매일 실행하여:
1. Edge >= 5%인 경기에 대해 가상 베팅 기록
2. 종료된 경기 결과 업데이트
3. 수익률 리포트 자동 생성

사용법:
    python scripts/paper_betting.py              # 오늘 날짜 기준
    python scripts/paper_betting.py 2025-12-01   # 특정 날짜
    python scripts/paper_betting.py --update-all # 모든 pending 베팅 결과 업데이트
"""

import sys
import os
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# 설정
EDGE_THRESHOLD = 0.05  # 5%
UNIT_SIZE = 100  # $100
DATA_DIR = PROJECT_ROOT / "data" / "paper_betting"
BETS_FILE = DATA_DIR / "bets.json"
REPORT_FILE = DATA_DIR / "BETTING_REPORT.md"
ODDS_HISTORY_DIR = PROJECT_ROOT / "data" / "odds_history"


def load_bets() -> Dict:
    """베팅 기록 로드"""
    if BETS_FILE.exists():
        with open(BETS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "edge_threshold": EDGE_THRESHOLD,
            "unit_size": UNIT_SIZE,
        },
        "bets": [],
        "summary": {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "total_profit": 0.0,
            "roi": 0.0,
        }
    }


def save_bets(data: Dict):
    """베팅 기록 저장"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BETS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"베팅 기록 저장: {BETS_FILE}")


def load_closing_odds(target_date: date) -> Dict[str, Dict]:
    """
    해당 날짜의 closing 배당 로드 (가장 마지막 캡처된 파일 사용).

    Returns:
        {game_id: {moneyline_home, moneyline_away, ...}} 형태의 딕셔너리
    """
    date_str = target_date.isoformat()
    odds_dir = ODDS_HISTORY_DIR / date_str

    if not odds_dir.exists():
        logger.warning(f"배당 히스토리 없음: {odds_dir}")
        return {}

    # 가장 최근 파일 찾기 (파일명 기준 정렬)
    json_files = sorted(odds_dir.glob("*.json"), reverse=True)

    if not json_files:
        logger.warning(f"배당 파일 없음: {odds_dir}")
        return {}

    # 가장 최근 파일 로드
    latest_file = json_files[0]
    logger.info(f"Closing 배당 로드: {latest_file.name}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # game_id를 키로 하는 딕셔너리로 변환
    odds_by_game = {}
    for game in data.get("games", []):
        game_id = game.get("game_id")
        odds = game.get("odds")
        if game_id and odds:
            odds_by_game[game_id] = {
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "moneyline_home": odds.get("moneyline_home"),
                "moneyline_away": odds.get("moneyline_away"),
                "spread_home": odds.get("spread_home"),
                "bookmaker": odds.get("bookmaker", "pinnacle"),
            }

    logger.info(f"Closing 배당 로드: {len(odds_by_game)}경기")
    return odds_by_game


def calculate_edge(model_prob: float, ml_home: float, ml_away: float) -> Dict:
    """
    Edge 및 EV 계산 (vig-adjusted fair probability 사용)

    앱과 동일한 로직: 북메이커 vig를 제거한 fair probability 기준 edge 계산
    """
    if ml_home <= 1 or ml_away <= 1:
        return None

    implied_home = 1 / ml_home
    implied_away = 1 / ml_away
    total_implied = implied_home + implied_away  # 보통 1.02~1.05 (vig 포함)

    # Vig 제거된 fair probability
    fair_home = implied_home / total_implied
    fair_away = implied_away / total_implied

    edge_home = model_prob - fair_home
    edge_away = (1 - model_prob) - fair_away

    ev_home = model_prob * (ml_home - 1) - (1 - model_prob)
    ev_away = (1 - model_prob) * (ml_away - 1) - model_prob

    return {
        "fair_home": fair_home,
        "fair_away": fair_away,
        "edge_home": edge_home,
        "edge_away": edge_away,
        "ev_home": ev_home,
        "ev_away": ev_away,
    }


def get_data_loader():
    """DataLoader 인스턴스 생성"""
    from app.services.data_loader import DataLoader
    data_dir = PROJECT_ROOT / "data"
    return DataLoader(data_dir)


def get_prediction_service():
    """V4.3 예측 서비스 로드"""
    from app.services.predictor_v4 import V4PredictionService
    model_dir = PROJECT_ROOT / "bucketsvision_v4" / "models"
    return V4PredictionService(model_dir, version="4.3")


def get_predictions_for_date(target_date: date, use_closing_odds: bool = True) -> List[Dict]:
    """
    특정 날짜의 예측 및 배당 정보 가져오기.

    Args:
        target_date: 대상 날짜
        use_closing_odds: True면 저장된 closing 배당 사용, False면 실시간 배당 사용
    """
    from app.services.data_loader import TEAM_INFO

    loader = get_data_loader()
    predictor = get_prediction_service()

    # 데이터 로드
    games = loader.get_games(target_date)
    if not games:
        logger.warning(f"{target_date}: 경기 없음")
        return []

    logger.info(f"Found {len(games)} games for {target_date}")

    # Closing 배당 로드 (경기 직전 캡처된 배당)
    closing_odds = {}
    if use_closing_odds:
        closing_odds = load_closing_odds(target_date)
        if not closing_odds:
            logger.warning(f"{target_date}: Closing 배당 없음 - 베팅 기록 스킵")
            return []

    # EPM 데이터 로드
    team_epm = loader.load_team_epm(target_date)

    predictions = []

    for game in games:
        game_id = game.get('game_id')
        home_id = game.get('home_team_id')
        away_id = game.get('away_team_id')
        game_status = game.get('game_status', 1)

        # 팀 ID -> 약어 변환
        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})
        home_abbr = home_info.get("abbr", "UNK")
        away_abbr = away_info.get("abbr", "UNK")

        # 예측 수행
        try:
            # V4.3 피처 생성
            features = loader.build_v4_3_features(home_id, away_id, team_epm, target_date)

            if not features:
                logger.debug(f"{home_abbr} vs {away_abbr}: 피처 생성 실패")
                continue

            # V4.3 예측
            home_prob = predictor.predict_proba(features)

            # 배당 정보 가져오기
            if use_closing_odds:
                # Closing 배당 사용 (저장된 경기 직전 배당)
                odds_info = closing_odds.get(game_id)
                if not odds_info:
                    logger.debug(f"{home_abbr} vs {away_abbr}: Closing 배당 없음 (game_id={game_id})")
                    continue
                ml_home = odds_info.get('moneyline_home')
                ml_away = odds_info.get('moneyline_away')
            else:
                # 실시간 배당 사용
                odds_info = loader.get_game_odds(home_abbr, away_abbr)
                if not odds_info:
                    logger.debug(f"{home_abbr} vs {away_abbr}: 배당 정보 없음")
                    continue
                ml_home = odds_info.get('moneyline_home')
                ml_away = odds_info.get('moneyline_away')

            if ml_home is None or ml_away is None:
                logger.debug(f"{home_abbr} vs {away_abbr}: ML 배당 없음")
                continue

            # Edge 계산 (vig-adjusted fair probability 기준)
            edge_data = calculate_edge(home_prob, ml_home, ml_away)

            if edge_data is None:
                logger.debug(f"{home_abbr} vs {away_abbr}: Edge 계산 실패 (ml_home={ml_home}, ml_away={ml_away})")
                continue

            predictions.append({
                "game_id": game_id,
                "date": target_date.isoformat(),
                "home_team": home_abbr,
                "away_team": away_abbr,
                "game_status": game_status,
                "home_score": game.get('home_score'),
                "away_score": game.get('away_score'),
                "model_home_prob": home_prob,
                "ml_home": ml_home,
                "ml_away": ml_away,
                "home_edge": edge_data['edge_home'],
                "home_ev": edge_data['ev_home'],
                "away_edge": edge_data['edge_away'],
                "away_ev": edge_data['ev_away'],
                "odds_source": "closing" if use_closing_odds else "live",
            })

            logger.info(f"{home_abbr} vs {away_abbr}: prob={home_prob:.1%}, edge_home={edge_data['edge_home']*100:+.1f}%, edge_away={edge_data['edge_away']*100:+.1f}% [{'closing' if use_closing_odds else 'live'}]")

        except Exception as e:
            logger.error(f"예측 오류 {game_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    return predictions


def record_bets(target_date: date, use_closing_odds: bool = True) -> int:
    """
    Edge >= threshold인 경기에 베팅 기록.

    Args:
        target_date: 대상 날짜
        use_closing_odds: True면 저장된 closing 배당 사용
    """
    data = load_bets()

    # 이미 기록된 game_id 확인
    existing_ids = {b['game_id'] for b in data['bets']}

    predictions = get_predictions_for_date(target_date, use_closing_odds=use_closing_odds)
    new_bets = 0

    for pred in predictions:
        game_id = pred['game_id']

        # 이미 기록됨
        if game_id in existing_ids:
            continue

        # Edge 확인
        home_edge = pred['home_edge']
        away_edge = pred['away_edge']

        bet_side = None
        bet_odds = None
        bet_edge = None
        bet_ev = None
        bet_team = None

        if home_edge >= EDGE_THRESHOLD:
            bet_side = 'home'
            bet_odds = pred['ml_home']
            bet_edge = home_edge
            bet_ev = pred['home_ev']
            bet_team = pred['home_team']
        elif away_edge >= EDGE_THRESHOLD:
            bet_side = 'away'
            bet_odds = pred['ml_away']
            bet_edge = away_edge
            bet_ev = pred['away_ev']
            bet_team = pred['away_team']

        if bet_side is None:
            continue

        # 베팅 기록
        bet_record = {
            "game_id": game_id,
            "date": pred['date'],
            "home_team": pred['home_team'],
            "away_team": pred['away_team'],
            "bet_side": bet_side,
            "bet_team": bet_team,
            "bet_odds": bet_odds,
            "bet_edge": bet_edge,
            "bet_ev": bet_ev,
            "model_home_prob": pred['model_home_prob'],
            "unit_size": UNIT_SIZE,
            "potential_profit": UNIT_SIZE * (bet_odds - 1),
            "status": "pending",
            "result": None,
            "profit": None,
            "recorded_at": datetime.now().isoformat(),
        }

        data['bets'].append(bet_record)
        data['summary']['pending'] += 1
        data['summary']['total_bets'] += 1
        new_bets += 1

        logger.info(f"베팅 기록: {bet_team} @{bet_odds:.2f} (Edge {bet_edge*100:.1f}%)")

    if new_bets > 0:
        save_bets(data)

    return new_bets


def update_results() -> int:
    """Pending 베팅 결과 업데이트"""
    data = load_bets()
    updated = 0

    loader = get_data_loader()

    for bet in data['bets']:
        if bet['status'] != 'pending':
            continue

        bet_date = date.fromisoformat(bet['date'])
        game_id = bet['game_id']

        # 경기 결과 확인
        try:
            games = loader.get_games(bet_date)

            if not games:
                continue

            # game_id로 경기 찾기
            game = None
            for g in games:
                if g.get('game_id') == game_id:
                    game = g
                    break

            if game is None:
                continue

            game_status = game.get('game_status', 1)

            # 아직 종료 안됨
            if game_status != 3:
                continue

            home_score = game.get('home_score')
            away_score = game.get('away_score')

            if home_score is None or away_score is None:
                continue

            # 결과 계산
            home_won = home_score > away_score
            bet_won = (bet['bet_side'] == 'home' and home_won) or \
                      (bet['bet_side'] == 'away' and not home_won)

            if bet_won:
                profit = UNIT_SIZE * (bet['bet_odds'] - 1)
                bet['result'] = 'win'
                data['summary']['wins'] += 1
            else:
                profit = -UNIT_SIZE
                bet['result'] = 'loss'
                data['summary']['losses'] += 1

            bet['status'] = 'settled'
            bet['profit'] = profit
            bet['home_score'] = home_score
            bet['away_score'] = away_score
            bet['settled_at'] = datetime.now().isoformat()

            data['summary']['pending'] -= 1
            data['summary']['total_profit'] += profit

            updated += 1
            result_emoji = "✅" if bet_won else "❌"
            logger.info(f"{result_emoji} {bet['bet_team']}: {home_score}-{away_score} → ${profit:+.0f}")

        except Exception as e:
            logger.error(f"결과 업데이트 오류 {game_id}: {e}")
            continue

    # ROI 계산
    settled_count = data['summary']['wins'] + data['summary']['losses']
    if settled_count > 0:
        total_wagered = settled_count * UNIT_SIZE
        data['summary']['roi'] = data['summary']['total_profit'] / total_wagered * 100

    if updated > 0:
        save_bets(data)

    return updated


def generate_report():
    """마크다운 리포트 생성"""
    data = load_bets()
    summary = data['summary']
    bets = data['bets']

    # 최근 베팅 (최신순)
    sorted_bets = sorted(bets, key=lambda x: x['date'], reverse=True)

    # 날짜별 그룹핑
    from collections import defaultdict
    daily_stats = defaultdict(lambda: {"bets": [], "profit": 0, "wins": 0, "losses": 0})

    for bet in bets:
        d = bet['date']
        daily_stats[d]['bets'].append(bet)
        if bet['status'] == 'settled':
            daily_stats[d]['profit'] += bet['profit'] or 0
            if bet['result'] == 'win':
                daily_stats[d]['wins'] += 1
            else:
                daily_stats[d]['losses'] += 1

    # 리포트 생성
    lines = [
        "# 📊 Paper Betting Report",
        "",
        f"*마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "---",
        "",
        "## 📈 Overall Summary",
        "",
        "| 지표 | 값 |",
        "|------|-----|",
        f"| 총 베팅 | {summary['total_bets']} |",
        f"| 승리 | {summary['wins']} |",
        f"| 패배 | {summary['losses']} |",
        f"| 대기중 | {summary['pending']} |",
        f"| 승률 | {summary['wins']/(summary['wins']+summary['losses'])*100:.1f}% |" if (summary['wins']+summary['losses']) > 0 else "| 승률 | - |",
        f"| 총 수익 | **${summary['total_profit']:+,.0f}** |",
        f"| ROI | **{summary['roi']:+.1f}%** |",
        "",
        f"*Edge 기준: ≥{EDGE_THRESHOLD*100:.0f}% | Unit: ${UNIT_SIZE}*",
        "",
        "---",
        "",
        "## 📅 Daily Results",
        "",
    ]

    # 날짜별 결과
    for d in sorted(daily_stats.keys(), reverse=True)[:30]:  # 최근 30일
        stats = daily_stats[d]
        total = stats['wins'] + stats['losses']
        pending = len([b for b in stats['bets'] if b['status'] == 'pending'])

        if total > 0:
            win_rate = stats['wins'] / total * 100
            profit_str = f"${stats['profit']:+,.0f}"
            profit_emoji = "🟢" if stats['profit'] > 0 else ("🔴" if stats['profit'] < 0 else "⚪")
        else:
            win_rate = 0
            profit_str = "-"
            profit_emoji = "⏳"

        pending_str = f" (+{pending} pending)" if pending > 0 else ""
        lines.append(f"### {d}")
        lines.append(f"- 결과: {stats['wins']}W-{stats['losses']}L{pending_str}")
        lines.append(f"- 수익: {profit_emoji} {profit_str}")
        lines.append("")

        # 개별 베팅 상세
        for bet in stats['bets']:
            if bet['status'] == 'settled':
                result_emoji = "✅" if bet['result'] == 'win' else "❌"
                score = f"{bet.get('home_score', '?')}-{bet.get('away_score', '?')}"
                profit = f"${bet['profit']:+,.0f}"
            else:
                result_emoji = "⏳"
                score = "-"
                profit = f"(potential: ${bet['potential_profit']:+,.0f})"

            lines.append(f"  - {result_emoji} **{bet['bet_team']}** @{bet['bet_odds']:.2f} | Edge {bet['bet_edge']*100:.1f}% | {bet['home_team']} vs {bet['away_team']} [{score}] → {profit}")

        lines.append("")

    # 파일 저장
    report_content = "\n".join(lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"리포트 생성: {REPORT_FILE}")
    return report_content


def main():
    """
    메인 실행.

    새벽 3시 자동 실행 시:
    1. 어제(ET 기준) 경기의 closing 배당으로 베팅 기록
    2. 모든 pending 베팅 결과 업데이트
    3. 리포트 생성
    """
    import argparse

    parser = argparse.ArgumentParser(description="Paper Betting Tracker")
    parser.add_argument("date", nargs="?", help="날짜 (YYYY-MM-DD), 기본값: 어제(ET)")
    parser.add_argument("--update-all", action="store_true", help="모든 pending 베팅 결과 업데이트만")
    parser.add_argument("--report-only", action="store_true", help="리포트만 생성")
    parser.add_argument("--live-odds", action="store_true", help="실시간 배당 사용 (테스트용)")
    args = parser.parse_args()

    if args.report_only:
        generate_report()
        return

    # 결과 업데이트 (항상 먼저 실행)
    logger.info("=== 베팅 결과 업데이트 ===")
    updated = update_results()
    logger.info(f"업데이트된 베팅: {updated}건")

    if not args.update_all:
        # 베팅 기록 (closing 배당 사용)
        if args.date:
            target_date = date.fromisoformat(args.date)
        else:
            # 미국 동부 시간 기준 어제 (새벽 3시 실행 시 어제 경기 처리)
            from zoneinfo import ZoneInfo
            et_now = datetime.now(ZoneInfo("America/New_York"))
            # 새벽 6시 이전이면 어제 날짜 사용
            if et_now.hour < 6:
                target_date = et_now.date() - timedelta(days=1)
            else:
                target_date = et_now.date()

        logger.info(f"=== {target_date} 베팅 기록 (closing 배당) ===")

        # Closing 배당 사용 여부
        use_closing = not args.live_odds

        if use_closing:
            logger.info("Closing 배당 사용 (경기 직전 캡처된 배당)")
        else:
            logger.info("실시간 배당 사용 (테스트 모드)")

        new_bets = record_bets(target_date, use_closing_odds=use_closing)
        logger.info(f"새로운 베팅: {new_bets}건")

    # 리포트 생성
    logger.info("=== 리포트 생성 ===")
    generate_report()

    # 요약 출력
    data = load_bets()
    summary = data['summary']
    print("\n" + "="*50)
    print(f"📊 Paper Betting Summary")
    print("="*50)
    print(f"총 베팅: {summary['total_bets']} | 승: {summary['wins']} | 패: {summary['losses']} | 대기: {summary['pending']}")
    print(f"총 수익: ${summary['total_profit']:+,.0f} | ROI: {summary['roi']:+.1f}%")
    print("="*50)


if __name__ == "__main__":
    main()
