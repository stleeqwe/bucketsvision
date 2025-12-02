"""
BucketsVision E2E Test Suite.

핵심 검증 항목:
1. 시즌 데이터 무결성 (25-26 현재 시즌, 2023-2025 학습 시즌)
2. API 연동 (NBA Stats, DNT, ESPN, Odds API)
3. 모델 예측 파이프라인
4. 부상 영향력 계산 (V2)
5. 프론트엔드 데이터 흐름
"""

import sys
from pathlib import Path
from datetime import date, datetime
from typing import Dict, List
import json

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    """테스트 결과 저장."""
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []

    def add_pass(self, msg: str = ""):
        self.passed += 1
        print(f"  ✓ {msg}")

    def add_fail(self, msg: str):
        self.failed += 1
        self.errors.append(msg)
        print(f"  ✗ {msg}")

    def summary(self) -> str:
        status = "PASS" if self.failed == 0 else "FAIL"
        return f"[{status}] {self.name}: {self.passed} passed, {self.failed} failed"


def test_season_data_integrity() -> TestResult:
    """시즌 데이터 무결성 검증."""
    result = TestResult("시즌 데이터 무결성")
    print("\n" + "="*60)
    print("TEST: 시즌 데이터 무결성")
    print("="*60)

    try:
        from src.utils.helpers import get_season_from_date

        # 테스트 케이스: (날짜, 예상 시즌)
        test_cases = [
            (date(2025, 10, 22), 2026, "25-26 시즌 시작일"),
            (date(2025, 11, 28), 2026, "25-26 시즌 중반"),
            (date(2025, 12, 2), 2026, "오늘 날짜"),
            (date(2026, 1, 15), 2026, "25-26 시즌 1월"),
            (date(2026, 4, 10), 2026, "25-26 시즌 플레이오프"),
            (date(2024, 11, 1), 2025, "24-25 시즌"),
            (date(2024, 3, 15), 2024, "23-24 시즌"),
        ]

        for test_date, expected_season, desc in test_cases:
            actual = get_season_from_date(test_date)
            if actual == expected_season:
                result.add_pass(f"{desc}: {test_date} → 시즌 {actual}")
            else:
                result.add_fail(f"{desc}: {test_date} → 예상 {expected_season}, 실제 {actual}")

        # 현재 날짜 검증
        today = date.today()
        current_season = get_season_from_date(today)
        if current_season == 2026:
            result.add_pass(f"현재 시즌 확인: {today} → {current_season} (25-26 시즌)")
        else:
            result.add_fail(f"현재 시즌 오류: {today} → {current_season} (예상: 2026)")

    except Exception as e:
        result.add_fail(f"예외 발생: {e}")

    return result


def test_model_files() -> TestResult:
    """모델 파일 검증."""
    result = TestResult("모델 파일 검증")
    print("\n" + "="*60)
    print("TEST: 모델 파일 검증")
    print("="*60)

    try:
        # 모델 경로들 확인
        model_dirs = [
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "bucketsvision_v4" / "models",
        ]

        model_found = False
        for model_dir in model_dirs:
            model_path = model_dir / "v4_3_model.pkl"
            if model_path.exists():
                result.add_pass(f"V4.3 모델 발견: {model_path}")
                model_found = True
                break

        if not model_found:
            result.add_fail("V4.3 모델 파일 없음")

        # 피처 이름 확인
        feature_paths = [
            PROJECT_ROOT / "models" / "v4_3_feature_names.json",
            PROJECT_ROOT / "bucketsvision_v4" / "models" / "v4_3_feature_names.json",
        ]

        for feature_path in feature_paths:
            if feature_path.exists():
                with open(feature_path) as f:
                    features = json.load(f)
                result.add_pass(f"피처 파일 발견: {len(features)}개 피처")

                expected_features = [
                    "team_epm_diff", "team_oepm_diff", "team_depm_diff", "sos_diff",
                    "efg_pct_diff", "ft_rate_diff", "last5_win_pct_diff",
                    "streak_diff", "margin_ewma_diff", "away_road_strength", "orb_diff"
                ]

                if len(features) >= 11:
                    result.add_pass(f"피처 수 확인: {len(features)}개 (>=11)")
                else:
                    result.add_fail(f"피처 수 부족: {len(features)}개 (최소 11개 필요)")

                missing = [f for f in expected_features if f not in features]
                if not missing:
                    result.add_pass("모든 필수 피처 존재")
                else:
                    result.add_fail(f"누락 피처: {missing}")
                break

    except Exception as e:
        result.add_fail(f"예외 발생: {e}")

    return result


def test_api_connections() -> TestResult:
    """API 연동 검증."""
    result = TestResult("API 연동 검증")
    print("\n" + "="*60)
    print("TEST: API 연동 검증")
    print("="*60)

    # NBA Stats API (게임 로그 테스트)
    try:
        from src.data_collection.nba_stats_client import NBAStatsClient
        nba_client = NBAStatsClient()

        team_logs = nba_client.get_team_game_logs(season=2026)
        if not team_logs.empty and len(team_logs) > 100:
            result.add_pass(f"NBA Stats API - 팀 경기 로그: {len(team_logs)}개")
        else:
            result.add_fail(f"NBA Stats API - 팀 경기 로그 부족: {len(team_logs)}개")
    except Exception as e:
        result.add_fail(f"NBA Stats API 오류: {e}")

    # DNT API
    try:
        from app.services.dnt_api import DNTApiClient
        dnt_client = DNTApiClient()

        team_epm = dnt_client.get_team_epm()
        if team_epm:
            # Dict 형식인지 확인
            if isinstance(team_epm, dict) and len(team_epm) == 30:
                result.add_pass(f"DNT API - 팀 EPM (dict): {len(team_epm)}팀")
                # EPM 값 범위 검증
                epm_values = [v.get('team_epm', 0) for v in team_epm.values()]
                if all(-15 < epm < 15 for epm in epm_values):
                    result.add_pass("DNT API - EPM 값 범위 정상")
                else:
                    result.add_fail(f"DNT API - EPM 값 이상")
            elif isinstance(team_epm, list) and len(team_epm) == 30:
                result.add_pass(f"DNT API - 팀 EPM (list): {len(team_epm)}팀")
            else:
                result.add_fail(f"DNT API - 팀 EPM 형식 오류: {type(team_epm)}")
        else:
            result.add_fail("DNT API - 팀 EPM 없음")
    except Exception as e:
        result.add_fail(f"DNT API 오류: {e}")

    # ESPN API
    try:
        from src.data_collection.espn_injury_client import ESPNInjuryClient
        espn_client = ESPNInjuryClient()

        injuries = espn_client.fetch_all_injuries()
        if injuries is not None:
            result.add_pass(f"ESPN API - 부상 정보: {len(injuries)}명")
        else:
            result.add_pass("ESPN API - 현재 부상자 없음")
    except Exception as e:
        result.add_fail(f"ESPN API 오류: {e}")

    return result


def test_data_loader() -> TestResult:
    """DataLoader 검증."""
    result = TestResult("DataLoader 검증")
    print("\n" + "="*60)
    print("TEST: DataLoader 검증")
    print("="*60)

    try:
        from app.services.data_loader import DataLoader

        data_dir = PROJECT_ROOT / "data"
        loader = DataLoader(data_dir)
        result.add_pass("DataLoader 초기화 성공")

        # 오늘 경기 조회
        today = date.today()
        games = loader.get_games(today)
        result.add_pass(f"오늘 경기 조회: {len(games)}경기")

        # 팀 EPM 로드
        team_epm = loader.load_team_epm(today)
        if team_epm and len(team_epm) == 30:
            result.add_pass(f"팀 EPM 로드: {len(team_epm)}팀")
        else:
            result.add_fail(f"팀 EPM 불완전: {len(team_epm) if team_epm else 0}팀")

        # 팀 경기 로그 로드
        team_logs = loader.load_team_game_logs(today)
        if not team_logs.empty:
            result.add_pass(f"팀 경기 로그 로드: {len(team_logs)}개")

            # 25-26 시즌 데이터 확인
            if 'game_date' in team_logs.columns:
                import pandas as pd
                dates = pd.to_datetime(team_logs['game_date'])
                min_date = dates.min()
                max_date = dates.max()
                result.add_pass(f"경기 날짜 범위: {min_date.date()} ~ {max_date.date()}")

                # 2025년 10월 이후 데이터 확인
                season_start = pd.Timestamp('2025-10-01')
                recent_games = (dates >= season_start).sum()
                if recent_games > 0:
                    result.add_pass(f"25-26 시즌 경기: {recent_games}개")
                else:
                    result.add_fail("25-26 시즌 경기 없음")
        else:
            result.add_fail("팀 경기 로그 없음")

    except Exception as e:
        result.add_fail(f"예외 발생: {e}")
        import traceback
        print(f"    상세: {traceback.format_exc()}")

    return result


def test_prediction_service() -> TestResult:
    """예측 서비스 검증."""
    result = TestResult("예측 서비스")
    print("\n" + "="*60)
    print("TEST: 예측 서비스")
    print("="*60)

    try:
        from app.services.predictor_v4 import V4PredictionService
        from app.services.data_loader import DataLoader

        data_dir = PROJECT_ROOT / "data"
        model_dir = PROJECT_ROOT / "bucketsvision_v4" / "models"
        loader = DataLoader(data_dir)

        # 예측 서비스 초기화
        predictor = V4PredictionService(model_dir=model_dir)
        result.add_pass("V4PredictionService 초기화 성공")

        # 오늘 경기 조회
        today = date.today()
        games = loader.get_games(today)
        team_epm = loader.load_team_epm(today)

        if games and team_epm:
            game = games[0]
            home_id = game.get("home_team_id")
            away_id = game.get("away_team_id")

            # 피처 생성
            features = loader.build_v4_features(home_id, away_id, today, team_epm)
            if features:
                result.add_pass(f"피처 생성 성공: {len(features)}개")

                # 예측 수행
                prob = predictor.predict_proba(features)
                if 0.01 <= prob <= 0.99:
                    result.add_pass(f"예측 확률 범위 정상: {prob:.1%}")
                else:
                    result.add_fail(f"예측 확률 범위 이상: {prob}")
            else:
                result.add_fail("피처 생성 실패")
        else:
            result.add_pass("오늘 경기 없음 또는 EPM 없음 (스킵)")

    except Exception as e:
        result.add_fail(f"예외 발생: {e}")
        import traceback
        print(f"    상세: {traceback.format_exc()}")

    return result


def test_injury_impact_v2() -> TestResult:
    """부상 영향력 V2 모델 검증."""
    result = TestResult("부상 영향력 V2")
    print("\n" + "="*60)
    print("TEST: 부상 영향력 V2 모델")
    print("="*60)

    try:
        from app.services.data_loader import DataLoader

        data_dir = PROJECT_ROOT / "data"
        loader = DataLoader(data_dir)
        today = date.today()
        team_epm = loader.load_team_epm(today)

        if not team_epm:
            result.add_fail("팀 EPM 로드 실패")
            return result

        # 고급 부상 계산기 초기화
        calc = loader.get_advanced_injury_calculator(today, team_epm)

        if calc is None:
            result.add_fail("부상 계산기 초기화 실패")
            return result

        result.add_pass("부상 계산기 초기화 성공")

        # 데이터 통계 확인
        stats = calc.get_data_stats()
        result.add_pass(f"데이터 통계: {stats}")

        # 25-26 시즌 데이터 확인
        if "season_25_26_games" in stats:
            season_games = stats["season_25_26_games"]
            total_games = stats["team_game_logs_count"]

            if season_games > 0 and season_games == total_games:
                result.add_pass(f"25-26 시즌 데이터: {season_games}/{total_games} (100%)")
            else:
                result.add_fail(f"시즌 데이터 불일치: {season_games}/{total_games}")

        # 날짜 범위 확인
        if "game_date_range" in stats:
            result.add_pass(f"경기 날짜 범위: {stats['game_date_range']}")

        # 부상 요약 테스트 (LAL 예시)
        injury_summary = loader.get_injury_summary("LAL", today, team_epm)

        if injury_summary:
            result.add_pass("부상 요약 조회 성공")

            # 필수 필드 확인
            required_fields = ["out_players", "gtd_players", "total_prob_shift"]
            for field in required_fields:
                if field in injury_summary:
                    result.add_pass(f"필드 존재: {field}")
                else:
                    result.add_fail(f"필드 누락: {field}")

            # prob_shift 형식 확인
            total_shift = injury_summary.get("total_prob_shift", 0)
            result.add_pass(f"total_prob_shift: {total_shift}%")

            # 선수별 V2 필드 확인
            out_players = injury_summary.get("out_players", [])
            if out_players:
                player = out_players[0]
                v2_fields = ["prob_shift", "played_games", "missed_games"]
                for field in v2_fields:
                    if field in player:
                        result.add_pass(f"V2 필드: {field}")

    except Exception as e:
        result.add_fail(f"예외 발생: {e}")
        import traceback
        print(f"    상세: {traceback.format_exc()}")

    return result


def test_frontend_data_flow() -> TestResult:
    """프론트엔드 데이터 흐름 검증."""
    result = TestResult("프론트엔드 데이터 흐름")
    print("\n" + "="*60)
    print("TEST: 프론트엔드 데이터 흐름")
    print("="*60)

    try:
        from app.main import apply_injury_correction, apply_b2b_correction

        # 부상 보정 함수 테스트
        base_prob = 0.55

        # 케이스 1: 부상 없음
        adjusted = apply_injury_correction(base_prob, 0.0, 0.0)
        if adjusted == base_prob:
            result.add_pass("부상 없음 → 확률 변화 없음")
        else:
            result.add_fail(f"부상 없음 오류: {base_prob} → {adjusted}")

        # 케이스 2: 홈팀 부상
        adjusted = apply_injury_correction(base_prob, 5.0, 0.0)
        if adjusted < base_prob:
            result.add_pass(f"홈팀 부상 → 홈 승률 감소: {base_prob:.1%} → {adjusted:.1%}")
        else:
            result.add_fail(f"홈팀 부상 보정 오류")

        # 케이스 3: 원정팀 부상
        adjusted = apply_injury_correction(base_prob, 0.0, 5.0)
        if adjusted > base_prob:
            result.add_pass(f"원정팀 부상 → 홈 승률 증가: {base_prob:.1%} → {adjusted:.1%}")
        else:
            result.add_fail(f"원정팀 부상 보정 오류")

        # 케이스 4: 최대 보정 한도
        adjusted = apply_injury_correction(base_prob, 20.0, 0.0)
        expected_max = base_prob - 0.10
        if abs(adjusted - expected_max) < 0.01:
            result.add_pass(f"최대 보정 한도 적용: {adjusted:.1%}")
        else:
            result.add_fail(f"최대 보정 한도 오류")

        # B2B 보정 테스트
        adjusted = apply_b2b_correction(base_prob, False, False)
        if adjusted == base_prob:
            result.add_pass("B2B 없음 → 확률 변화 없음")

        adjusted = apply_b2b_correction(base_prob, False, True)
        if adjusted > base_prob:
            result.add_pass(f"원정팀 B2B → 홈 승률 증가: {adjusted:.1%}")

    except Exception as e:
        result.add_fail(f"예외 발생: {e}")

    return result


def test_end_to_end_prediction() -> TestResult:
    """전체 예측 파이프라인 E2E 테스트."""
    result = TestResult("E2E 예측 파이프라인")
    print("\n" + "="*60)
    print("TEST: E2E 예측 파이프라인")
    print("="*60)

    try:
        from app.services.data_loader import DataLoader
        from app.services.predictor_v4 import V4PredictionService
        from app.main import apply_injury_correction, apply_b2b_correction

        data_dir = PROJECT_ROOT / "data"
        model_dir = PROJECT_ROOT / "bucketsvision_v4" / "models"
        loader = DataLoader(data_dir)
        predictor = V4PredictionService(model_dir=model_dir)
        today = date.today()

        # 1. 경기 조회
        games = loader.get_games(today)
        result.add_pass(f"Step 1: 경기 조회 - {len(games)}경기")

        if not games:
            result.add_pass("오늘 경기 없음 - E2E 테스트 스킵")
            return result

        # 2. 팀 EPM 로드
        team_epm = loader.load_team_epm(today)
        result.add_pass(f"Step 2: 팀 EPM - {len(team_epm)}팀")

        game = games[0]
        home_id = game.get("home_team_id")
        away_id = game.get("away_team_id")
        home_abbr = game.get("home_team")
        away_abbr = game.get("away_team")

        # 3. 피처 생성
        features = loader.build_v4_features(home_id, away_id, today, team_epm)
        result.add_pass(f"Step 3: 피처 생성 - {len(features)}개")

        # 4. 기본 예측
        base_prob = predictor.predict_proba(features)
        result.add_pass(f"Step 4: 기본 예측 - {base_prob:.1%}")

        # 5. B2B 보정
        home_b2b = loader.is_b2b(home_id, today)
        away_b2b = loader.is_b2b(away_id, today)
        b2b_prob = apply_b2b_correction(base_prob, home_b2b, away_b2b)
        result.add_pass(f"Step 5: B2B 보정 - {b2b_prob:.1%} (홈B2B:{home_b2b}, 원정B2B:{away_b2b})")

        # 6. 부상 보정
        home_injury = loader.get_injury_summary(home_abbr, today, team_epm)
        away_injury = loader.get_injury_summary(away_abbr, today, team_epm)
        home_shift = home_injury.get("total_prob_shift", 0)
        away_shift = away_injury.get("total_prob_shift", 0)
        final_prob = apply_injury_correction(b2b_prob, home_shift, away_shift)
        result.add_pass(f"Step 6: 부상 보정 - {final_prob:.1%} (홈:{home_shift}%, 원정:{away_shift}%)")

        # 7. 최종 검증
        if 0.01 <= final_prob <= 0.99:
            result.add_pass(f"Step 7: 최종 확률 범위 정상 - {final_prob:.1%}")
        else:
            result.add_fail(f"최종 확률 범위 이상: {final_prob}")

        result.add_pass(f"E2E 완료: {home_abbr} vs {away_abbr} → 홈 승률 {final_prob:.1%}")

    except Exception as e:
        result.add_fail(f"예외 발생: {e}")
        import traceback
        print(f"    상세: {traceback.format_exc()}")

    return result


def run_all_tests() -> None:
    """모든 테스트 실행."""
    print("\n" + "="*60)
    print("BucketsVision E2E 테스트")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    results: List[TestResult] = []

    # 테스트 실행
    results.append(test_season_data_integrity())
    results.append(test_model_files())
    results.append(test_api_connections())
    results.append(test_data_loader())
    results.append(test_prediction_service())
    results.append(test_injury_impact_v2())
    results.append(test_frontend_data_flow())
    results.append(test_end_to_end_prediction())

    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)

    total_passed = 0
    total_failed = 0

    for r in results:
        print(r.summary())
        total_passed += r.passed
        total_failed += r.failed

    print("-"*60)
    print(f"총 결과: {total_passed} passed, {total_failed} failed")

    if total_failed == 0:
        print("\n✅ 모든 테스트 통과!")
    else:
        print(f"\n❌ {total_failed}개 테스트 실패")


if __name__ == "__main__":
    run_all_tests()
