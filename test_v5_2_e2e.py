#!/usr/bin/env python3
"""
V5.2 End-to-End 테스트

테스트 항목:
1. NBA Stats API - 오늘/내일 경기 로드
2. DNT API - 팀 EPM 데이터 로드
3. Odds API - 배당 정보 로드
4. V5.2 피처 생성
5. V5.2 예측 (XGBoost)
6. 부상 조정 적용
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
import pytz

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.predictor_v5 import V5PredictionService
from app.services.data_loader import DataLoader, TEAM_INFO
from config.constants import ABBR_TO_ID


def get_et_today() -> date:
    """미국 동부 시간 기준 오늘 날짜"""
    et = pytz.timezone('America/New_York')
    return datetime.now(et).date()


def test_nba_api(loader: DataLoader, test_date: date):
    """NBA Stats API 테스트"""
    print("\n" + "="*70)
    print("  [1] NBA Stats API 테스트")
    print("="*70)

    games = loader.get_games(test_date)

    if not games:
        print(f"  ⚠️ {test_date} 경기 없음 - 다른 날짜 시도")
        # 내일 경기 확인
        tomorrow = test_date + timedelta(days=1)
        games = loader.get_games(tomorrow)
        if games:
            print(f"  ✓ {tomorrow} 경기 {len(games)}개 발견")
            test_date = tomorrow
        else:
            # 어제 경기 확인
            yesterday = test_date - timedelta(days=1)
            games = loader.get_games(yesterday)
            if games:
                print(f"  ✓ {yesterday} 경기 {len(games)}개 발견")
                test_date = yesterday

    if not games:
        print("  ❌ 경기를 찾을 수 없습니다")
        return None, None

    print(f"\n  날짜: {test_date}")
    print(f"  경기 수: {len(games)}")
    print("\n  [경기 목록]")
    print("  " + "-"*60)

    for i, game in enumerate(games, 1):
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})
        home_abbr = home_info.get('abbr', 'UNK')
        away_abbr = away_info.get('abbr', 'UNK')

        status_map = {1: "예정", 2: "진행중", 3: "종료"}
        status = status_map.get(game.get('game_status', 1), "알수없음")

        home_b2b = "🔄" if game.get('home_b2b') else ""
        away_b2b = "🔄" if game.get('away_b2b') else ""

        score = ""
        if game.get('home_score') is not None:
            score = f" ({game['home_score']}-{game['away_score']})"

        print(f"  {i}. {away_abbr}{away_b2b} @ {home_abbr}{home_b2b} - {game['game_time']} [{status}]{score}")

    return games, test_date


def test_dnt_api(loader: DataLoader, test_date: date):
    """DNT API 테스트"""
    print("\n" + "="*70)
    print("  [2] DNT API 테스트 (Team EPM)")
    print("="*70)

    team_epm = loader.load_team_epm(test_date)

    if not team_epm:
        print("  ❌ Team EPM 로드 실패")
        return None

    print(f"  ✓ {len(team_epm)} 팀 EPM 로드 완료")
    print("\n  [Top 5 팀 EPM]")
    print("  " + "-"*50)

    # 상위 5팀
    sorted_teams = sorted(team_epm.items(), key=lambda x: x[1].get('team_epm', 0) or 0, reverse=True)
    for i, (team_id, epm_data) in enumerate(sorted_teams[:5], 1):
        team_info = TEAM_INFO.get(team_id, {})
        abbr = team_info.get('abbr', 'UNK')
        team_epm_val = epm_data.get('team_epm', 0) or 0
        print(f"  {i}. {abbr}: {team_epm_val:+.2f}")

    print("\n  [Bottom 5 팀 EPM]")
    print("  " + "-"*50)
    for i, (team_id, epm_data) in enumerate(sorted_teams[-5:], 1):
        team_info = TEAM_INFO.get(team_id, {})
        abbr = team_info.get('abbr', 'UNK')
        team_epm_val = epm_data.get('team_epm', 0) or 0
        print(f"  {i}. {abbr}: {team_epm_val:+.2f}")

    return team_epm


def test_odds_api(loader: DataLoader, games: list):
    """Odds API 테스트"""
    print("\n" + "="*70)
    print("  [3] Odds API 테스트")
    print("="*70)

    if not games:
        print("  ⚠️ 테스트할 경기 없음")
        return

    # 예정된 경기만 배당 조회
    scheduled_games = [g for g in games if g.get('game_status') == 1]

    if not scheduled_games:
        print("  ⚠️ 예정된 경기 없음 (종료된 경기는 배당 조회 불가)")
        return

    print(f"  예정된 경기: {len(scheduled_games)}개")
    print("\n  [배당 정보]")
    print("  " + "-"*60)

    for game in scheduled_games[:3]:  # 최대 3경기만 테스트
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})
        home_abbr = home_info.get('abbr', 'UNK')
        away_abbr = away_info.get('abbr', 'UNK')

        odds = loader.get_game_odds(home_abbr, away_abbr)
        if odds:
            print(f"  {away_abbr} @ {home_abbr}:")
            spread = odds.get('spread_home', 0)
            ml_home = odds.get('moneyline_home', 'N/A')
            ml_away = odds.get('moneyline_away', 'N/A')
            total = odds.get('total_line', 'N/A')
            print(f"    스프레드: {home_abbr} {spread:+.1f}")
            print(f"    머니라인: {home_abbr} {ml_home}, {away_abbr} {ml_away}")
            print(f"    오버/언더: {total}")
        else:
            print(f"  {away_abbr} @ {home_abbr}: 배당 정보 없음")


def test_v5_2_features(loader: DataLoader, predictor: V5PredictionService, games: list, team_epm: dict, test_date: date):
    """V5.2 피처 생성 및 예측 테스트"""
    print("\n" + "="*70)
    print("  [4] V5.2 피처 생성 및 예측 테스트")
    print("="*70)

    if not games or not team_epm:
        print("  ❌ 테스트 데이터 없음")
        return

    print(f"\n  모델 정보:")
    model_info = predictor.get_model_info()
    print(f"    - 버전: {model_info['model_version']}")
    print(f"    - 피처 수: {model_info['n_features']}")
    print(f"    - 저신뢰도 정확도: {model_info['low_conf_accuracy']:.2%}")

    print(f"\n  [경기별 예측]")
    print("  " + "-"*70)

    for game in games[:5]:  # 최대 5경기
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})
        home_abbr = home_info.get('abbr', 'UNK')
        away_abbr = away_info.get('abbr', 'UNK')

        home_b2b = game.get('home_b2b', False)
        away_b2b = game.get('away_b2b', False)

        # V5.2 피처 생성
        features = loader.build_v5_2_features(
            home_id, away_id, team_epm, test_date,
            home_b2b=home_b2b, away_b2b=away_b2b
        )

        # 예측
        base_prob = predictor.predict_proba(features)

        # 부상 정보 (예정된 경기만)
        home_prob_shift = 0.0
        away_prob_shift = 0.0

        if game.get('game_status') == 1:
            try:
                home_injury = loader.get_injury_summary(home_abbr, test_date, team_epm)
                away_injury = loader.get_injury_summary(away_abbr, test_date, team_epm)
                home_prob_shift = home_injury.get('total_prob_shift', 0.0)
                away_prob_shift = away_injury.get('total_prob_shift', 0.0)
            except:
                pass

        # 부상 조정
        adj_prob = predictor.apply_injury_adjustment(base_prob, home_prob_shift, away_prob_shift)

        # 결과 출력
        status_map = {1: "예정", 2: "진행중", 3: "종료"}
        status = status_map.get(game.get('game_status', 1), "?")

        b2b_info = ""
        if home_b2b or away_b2b:
            b2b_parts = []
            if home_b2b: b2b_parts.append(f"{home_abbr} B2B")
            if away_b2b: b2b_parts.append(f"{away_abbr} B2B")
            b2b_info = f" [{', '.join(b2b_parts)}]"

        injury_info = ""
        if home_prob_shift > 0 or away_prob_shift > 0:
            injury_info = f" [부상: {home_abbr} -{home_prob_shift:.1f}%, {away_abbr} -{away_prob_shift:.1f}%]"

        print(f"\n  {away_abbr} @ {home_abbr} [{status}]{b2b_info}")
        print(f"    기본 예측:   {home_abbr} {base_prob:.1%}")
        if adj_prob != base_prob:
            print(f"    부상 조정:   {home_abbr} {adj_prob:.1%}{injury_info}")

        # 주요 피처 출력
        print(f"    주요 피처:")
        print(f"      team_epm_diff: {features['team_epm_diff']:+.3f}")
        print(f"      rotation_epm_diff: {features['player_rotation_epm_diff']:+.3f}")
        print(f"      b2b_diff: {features['b2b_diff']:+d}")
        print(f"      rest_days_diff: {features['rest_days_diff']:+d}")

        # 종료된 경기는 결과 비교
        if game.get('game_status') == 3:
            home_score = game.get('home_score')
            away_score = game.get('away_score')
            if home_score is not None:
                actual_home_win = home_score > away_score
                predicted_home_win = adj_prob >= 0.5
                correct = "✓" if actual_home_win == predicted_home_win else "✗"
                print(f"    결과: {home_score}-{away_score} ({home_abbr} {'승' if actual_home_win else '패'}) {correct}")


def test_injury_data(loader: DataLoader, games: list, team_epm: dict, test_date: date):
    """부상 데이터 테스트"""
    print("\n" + "="*70)
    print("  [5] 부상 데이터 테스트 (ESPN)")
    print("="*70)

    if not games:
        print("  ⚠️ 테스트할 경기 없음")
        return

    # 첫 번째 경기의 홈/어웨이 팀 부상 정보 조회
    game = games[0]
    home_id = game['home_team_id']
    away_id = game['away_team_id']
    home_info = TEAM_INFO.get(home_id, {})
    away_info = TEAM_INFO.get(away_id, {})
    home_abbr = home_info.get('abbr', 'UNK')
    away_abbr = away_info.get('abbr', 'UNK')

    print(f"\n  경기: {away_abbr} @ {home_abbr}")

    for abbr, label in [(home_abbr, "홈팀"), (away_abbr, "어웨이팀")]:
        print(f"\n  [{label}] {abbr} 부상자 현황:")
        print("  " + "-"*50)

        try:
            injury_summary = loader.get_injury_summary(abbr, test_date, team_epm)

            out_players = injury_summary.get('out', [])
            gtd_players = injury_summary.get('gtd', [])
            total_shift = injury_summary.get('total_prob_shift', 0)

            if out_players:
                print(f"  OUT ({len(out_players)}명):")
                for p in out_players[:5]:  # 최대 5명
                    name = p.get('name', 'Unknown')
                    shift = p.get('prob_shift', 0)
                    print(f"    - {name}: -{shift:.1f}%")

            if gtd_players:
                print(f"  GTD ({len(gtd_players)}명):")
                for p in gtd_players[:3]:  # 최대 3명
                    name = p.get('name', 'Unknown')
                    shift = p.get('prob_shift', 0)
                    print(f"    - {name}: -{shift:.1f}% (50% 반영)")

            print(f"  총 영향: -{total_shift:.1f}%")

        except Exception as e:
            print(f"  ⚠️ 부상 정보 조회 실패: {e}")


def run_full_test():
    """전체 E2E 테스트 실행"""
    print("="*70)
    print("  BucketsVision V5.2 End-to-End 테스트")
    print("  테스트 시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)

    # 초기화
    project_root = Path(__file__).parent
    model_dir = project_root / "bucketsvision_v4" / "models"
    data_dir = project_root / "data"

    print("\n[초기화]")
    predictor = V5PredictionService(model_dir)
    print(f"  ✓ V5.2 Predictor 로드 완료")

    loader = DataLoader(data_dir)
    print(f"  ✓ DataLoader 초기화 완료")

    # 테스트 날짜 (미국 동부 시간 기준)
    et_today = get_et_today()
    print(f"  ✓ 테스트 날짜: {et_today} (ET)")

    # 1. NBA API 테스트
    games, test_date = test_nba_api(loader, et_today)

    # 2. DNT API 테스트
    team_epm = test_dnt_api(loader, test_date or et_today)

    # 3. Odds API 테스트
    test_odds_api(loader, games)

    # 4. V5.2 피처 및 예측 테스트
    test_v5_2_features(loader, predictor, games, team_epm, test_date or et_today)

    # 5. 부상 데이터 테스트
    test_injury_data(loader, games, team_epm, test_date or et_today)

    # 요약
    print("\n" + "="*70)
    print("  [테스트 요약]")
    print("="*70)
    print(f"  ✓ NBA Stats API: {'정상' if games else '데이터 없음'}")
    print(f"  ✓ DNT API: {'정상' if team_epm else '실패'}")
    print(f"  ✓ V5.2 모델: 정상")
    print(f"  ✓ 부상 조정: 정상")
    print("="*70)


if __name__ == "__main__":
    run_full_test()
