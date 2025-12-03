#!/usr/bin/env python3
"""
BucketsVision 통합 테스트 프레임워크.

테스트 범위:
1. API 연동 테스트 (DNT, NBA Stats, ESPN, Odds)
2. 서비스 통합 테스트 (DataLoader, Predictor, InjuryCalculator)
3. E2E 파이프라인 테스트

실행:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -m smoke
    pytest tests/test_integration.py -v -m api
    pytest tests/test_integration.py -v -m e2e
"""

import sys
import json
import math
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pytest
import pytz
import numpy as np
import pandas as pd

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_season_from_date


# =============================================================================
# Level 1: 단위 테스트
# =============================================================================

@pytest.mark.unit
class TestDateSeasonLogic:
    """날짜 및 시즌 로직 검증"""

    def test_current_season_is_2026(self, et_today):
        """현재 시즌이 2026인지 확인 (2025-26 시즌)"""
        season = get_season_from_date(et_today)
        assert season == 2026, f"Expected season 2026, got {season}"

    def test_season_calculation_october_onwards(self):
        """10월 이후 날짜 → 다음 해 시즌"""
        test_cases = [
            (date(2025, 10, 22), 2026),  # 시즌 시작
            (date(2025, 11, 28), 2026),  # 11월
            (date(2025, 12, 25), 2026),  # 크리스마스
            (date(2026, 1, 15), 2026),   # 1월
            (date(2026, 4, 10), 2026),   # 레귤러시즌 끝
            (date(2026, 6, 15), 2026),   # 플레이오프
        ]
        for test_date, expected_season in test_cases:
            result = get_season_from_date(test_date)
            assert result == expected_season, \
                f"Date {test_date}: expected {expected_season}, got {result}"

    def test_season_calculation_before_october(self):
        """10월 이전 날짜 → 같은 해 시즌"""
        test_cases = [
            (date(2026, 9, 30), 2026),  # 프리시즌
            (date(2025, 9, 15), 2025),  # 이전 시즌
        ]
        for test_date, expected_season in test_cases:
            result = get_season_from_date(test_date)
            assert result == expected_season, \
                f"Date {test_date}: expected {expected_season}, got {result}"


@pytest.mark.unit
class TestTeamInfoMapping:
    """팀 정보 매핑 검증"""

    def test_all_30_teams_exist(self, team_info):
        """30개 팀 데이터 완비"""
        assert len(team_info) == 30, f"Expected 30 teams, got {len(team_info)}"

    def test_team_info_required_fields(self, team_info):
        """팀 정보 필수 필드"""
        for team_id, info in team_info.items():
            assert 'abbr' in info, f"Missing abbr for team {team_id}"
            assert 'name' in info, f"Missing name for team {team_id}"
            assert 'color' in info, f"Missing color for team {team_id}"

    def test_team_colors_format(self, team_info):
        """팀 컬러 형식 (#XXXXXX)"""
        for team_id, info in team_info.items():
            color = info['color']
            assert color.startswith('#'), f"Invalid color format: {color}"
            assert len(color) == 7, f"Invalid color length: {color}"

    def test_abbr_to_id_mapping(self, abbr_to_id, team_info):
        """팀 약어 → ID 매핑"""
        assert len(abbr_to_id) == 30
        for abbr, team_id in abbr_to_id.items():
            assert team_id in team_info, f"Team ID {team_id} not in TEAM_INFO"
            assert team_info[team_id]['abbr'] == abbr


# =============================================================================
# Level 2: API 통합 테스트
# =============================================================================

@pytest.mark.api
@pytest.mark.smoke
class TestDNTAPIIntegration:
    """DNT API 연동 검증"""

    def test_team_epm_returns_30_teams(self, loader, et_today):
        """팀 EPM: 30개 팀 데이터 반환"""
        team_epm = loader.load_team_epm(et_today)
        assert len(team_epm) == 30, f"Expected 30 teams, got {len(team_epm)}"

    def test_team_epm_required_fields(self, team_epm):
        """팀 EPM 필수 필드"""
        required_fields = ['team_epm', 'team_oepm', 'team_depm', 'sos']
        for team_id, data in team_epm.items():
            for field in required_fields:
                assert field in data, f"Missing field '{field}' for team {team_id}"

    def test_team_epm_value_range(self, team_epm):
        """EPM 값 범위: -15 ~ +15"""
        for team_id, data in team_epm.items():
            epm = data.get('team_epm', 0)
            assert -15 <= epm <= 15, f"Team {team_id} EPM {epm} out of range"

    def test_player_epm_returns_sufficient_data(self, loader, current_season):
        """선수 EPM: 최소 300명 이상"""
        player_epm = loader.load_player_epm(current_season)
        assert not player_epm.empty, "Player EPM data is empty"
        assert len(player_epm) >= 300, f"Expected 300+ players, got {len(player_epm)}"

    def test_player_epm_required_columns(self, loader, current_season):
        """선수 EPM 필수 컬럼"""
        player_epm = loader.load_player_epm(current_season)
        required_cols = ['player_id', 'player_name', 'team_id', 'team_alias', 'tot', 'mpg']
        for col in required_cols:
            assert col in player_epm.columns, f"Missing column '{col}'"


@pytest.mark.api
class TestNBAStatsAPIIntegration:
    """NBA Stats API 연동 검증"""

    def test_scoreboard_returns_games(self, loader, et_today):
        """경기 스케줄 조회"""
        for offset in [0, 1, -1, 2, -2]:
            test_date = et_today + timedelta(days=offset)
            games = loader.get_games(test_date)
            if games:
                break
        # 최소 한 날짜에서는 경기가 있어야 함
        assert games is not None

    def test_game_data_structure(self, sample_game):
        """경기 데이터 구조"""
        if sample_game is None:
            pytest.skip("No games found for testing")

        game = sample_game["game"]
        required_keys = ['game_id', 'home_team_id', 'away_team_id', 'game_time', 'game_status']
        for key in required_keys:
            assert key in game, f"Missing key '{key}' in game data"

    def test_team_game_logs_for_current_season(self, loader, et_today):
        """현재 시즌 팀 경기 로그"""
        logs = loader.load_team_game_logs(et_today)
        assert not logs.empty, "Team game logs is empty"

        # 2025-10-01 이후 데이터 확인
        if 'game_date' in logs.columns:
            dates = pd.to_datetime(logs['game_date'])
            season_start = pd.Timestamp('2025-10-01')
            recent_games = (dates >= season_start).sum()
            assert recent_games > 0, "No 25-26 season games found"

    def test_team_game_logs_required_columns(self, loader, et_today):
        """팀 게임 로그 필수 컬럼"""
        logs = loader.load_team_game_logs(et_today)
        required_cols = ['team_id', 'game_id', 'game_date', 'result', 'pts']
        for col in required_cols:
            assert col in logs.columns, f"Missing column '{col}'"


@pytest.mark.api
class TestESPNAPIIntegration:
    """ESPN 부상 API 연동 검증"""

    def test_out_players_retrieval(self, loader):
        """Out 상태 선수 조회"""
        test_teams = ['LAL', 'BOS', 'GSW']
        for team in test_teams:
            injuries = loader.get_injuries(team)
            assert isinstance(injuries, list), f"Expected list for {team}"

    def test_gtd_players_retrieval(self, loader):
        """GTD 상태 선수 조회"""
        test_teams = ['LAL', 'BOS', 'GSW']
        for team in test_teams:
            gtd = loader.get_gtd_players(team)
            assert isinstance(gtd, list), f"Expected list for {team}"


@pytest.mark.api
class TestOddsAPIIntegration:
    """배당 API 연동 검증"""

    def test_odds_retrieval(self, loader):
        """배당 정보 조회"""
        # 캐시 초기화
        loader.clear_odds_cache()
        # 임의의 팀 조합으로 테스트 (없어도 None 반환)
        odds = loader.get_game_odds("LAL", "BOS")
        # 배당이 있거나 None (경기가 없으면)
        assert odds is None or isinstance(odds, dict)


# =============================================================================
# Level 3: 서비스 통합 테스트
# =============================================================================

@pytest.mark.smoke
class TestDataLoaderIntegration:
    """DataLoader 서비스 통합 검증"""

    def test_initialization(self, loader):
        """DataLoader 초기화"""
        assert loader is not None
        assert loader.nba_client is not None
        assert loader.dnt_client is not None
        assert loader.espn_client is not None

    def test_team_epm_caching(self, loader, et_today):
        """팀 EPM 캐싱 동작"""
        # 첫 번째 호출
        epm1 = loader.load_team_epm(et_today)
        # 두 번째 호출 (캐시 사용)
        epm2 = loader.load_team_epm(et_today)
        # 동일 결과
        assert epm1 == epm2

    def test_cache_invalidation(self, loader):
        """캐시 초기화 동작"""
        loader.clear_cache()
        assert loader._team_epm_date_cache == {}
        assert loader._team_game_logs_cache is None


@pytest.mark.smoke
class TestV54FeatureBuild:
    """V5.4 피처 빌드 검증"""

    def test_feature_count_is_5(self, loader, team_epm, et_today, abbr_to_id):
        """5개 피처 반환"""
        home_id = abbr_to_id['LAL']
        away_id = abbr_to_id['BOS']
        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)
        assert len(features) == 5, f"Expected 5 features, got {len(features)}"

    def test_feature_names_match_model(self, loader, team_epm, et_today, abbr_to_id, v54_feature_names):
        """피처명이 모델과 일치"""
        home_id = abbr_to_id['LAL']
        away_id = abbr_to_id['BOS']
        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)

        for feature_name in v54_feature_names:
            assert feature_name in features, f"Missing feature: {feature_name}"

    def test_no_nan_values(self, loader, team_epm, et_today, abbr_to_id):
        """NaN 값 없음"""
        home_id = abbr_to_id['LAL']
        away_id = abbr_to_id['BOS']
        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)

        for name, value in features.items():
            assert not math.isnan(value), f"NaN value for {name}"

    def test_team_epm_diff_calculation(self, loader, team_epm, et_today, abbr_to_id):
        """team_epm_diff = home_epm - away_epm"""
        home_id = abbr_to_id['BOS']  # 강팀
        away_id = abbr_to_id['DET']  # 약팀

        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)

        home_epm = team_epm.get(home_id, {}).get('team_epm', 0)
        away_epm = team_epm.get(away_id, {}).get('team_epm', 0)
        expected_diff = home_epm - away_epm

        assert abs(features['team_epm_diff'] - expected_diff) < 0.001

    def test_ft_rate_diff_range(self, loader, team_epm, et_today, abbr_to_id):
        """FT Rate 차이 범위"""
        home_id = abbr_to_id['MIA']
        away_id = abbr_to_id['NYK']
        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)
        assert -0.3 <= features['ft_rate_diff'] <= 0.3


@pytest.mark.smoke
class TestV54ModelPrediction:
    """V5.4 모델 예측 검증"""

    def test_model_and_scaler_loaded(self, predictor):
        """모델 및 스케일러 로드"""
        assert predictor.model is not None, "Model not loaded"
        assert predictor.scaler is not None, "Scaler not loaded"
        assert predictor.feature_names is not None, "Feature names not loaded"

    def test_model_metadata(self, predictor):
        """모델 메타데이터 검증"""
        info = predictor.get_model_info()
        assert info['n_features'] == 5
        assert info['model_version'] == '5.4.0'
        assert 0.70 <= info['overall_accuracy'] <= 0.85

    def test_balanced_game_around_50_percent(self, predictor, mock_balanced_features):
        """균형 경기(모든 피처 0) → ~50%"""
        prob = predictor.predict_proba(mock_balanced_features)
        assert 0.40 <= prob <= 0.60, f"Balanced game should be ~50%, got {prob:.1%}"

    def test_strong_home_above_70_percent(self, predictor, mock_strong_home_features):
        """강팀 홈 경기 → 70% 이상"""
        prob = predictor.predict_proba(mock_strong_home_features)
        assert prob >= 0.70, f"Strong home should be >70%, got {prob:.1%}"

    def test_weak_home_below_30_percent(self, predictor, mock_weak_home_features):
        """약팀 홈 경기 → 30% 이하"""
        prob = predictor.predict_proba(mock_weak_home_features)
        assert prob <= 0.30, f"Weak home should be <30%, got {prob:.1%}"

    def test_prediction_monotonicity(self, predictor, mock_balanced_features):
        """피처 증가 → 확률 증가 (단조성)"""
        base_prob = predictor.predict_proba(mock_balanced_features)

        higher_epm = mock_balanced_features.copy()
        higher_epm['team_epm_diff'] = 5.0
        higher_prob = predictor.predict_proba(higher_epm)

        assert higher_prob > base_prob, \
            f"Higher team_epm_diff should increase prob: {base_prob:.3f} -> {higher_prob:.3f}"

    def test_probability_range_within_bounds(self, predictor, mock_strong_home_features, mock_weak_home_features):
        """확률 범위: 1% ~ 99%"""
        high_prob = predictor.predict_proba(mock_strong_home_features)
        low_prob = predictor.predict_proba(mock_weak_home_features)

        assert 0.01 <= low_prob <= 0.99, f"Low prob {low_prob:.3f} out of range"
        assert 0.01 <= high_prob <= 0.99, f"High prob {high_prob:.3f} out of range"


class TestInjuryImpactCalculation:
    """부상 영향 계산 검증 (v1.0.0)"""

    def test_calculator_initialization(self, loader, et_today, team_epm):
        """AdvancedInjuryImpactCalculator 초기화"""
        calc = loader.get_advanced_injury_calculator(et_today, team_epm)
        # 데이터 부족 시 None 가능
        if calc is None:
            pytest.skip("Advanced injury calculator not available (data insufficient)")

    def test_version_is_1_0_0(self, project_paths):
        """버전 1.0.0 확인"""
        meta_path = project_paths["model_dir"] / "injury_impact_v1_metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)
        assert metadata['version'] == '1.0.0'

    def test_injury_summary_structure(self, loader, et_today, team_epm):
        """부상 요약 데이터 구조"""
        summary = loader.get_injury_summary('LAL', et_today, team_epm)

        assert 'out_players' in summary
        assert 'gtd_players' in summary
        assert 'total_prob_shift' in summary

        assert isinstance(summary['out_players'], list)
        assert isinstance(summary['gtd_players'], list)
        assert isinstance(summary['total_prob_shift'], (int, float))


class TestInjuryAdjustmentApplication:
    """부상 조정 적용 검증"""

    def test_no_adjustment_when_no_injuries(self, predictor):
        """부상 없을 때 확률 변화 없음"""
        base_prob = 0.55
        adjusted = predictor.apply_injury_adjustment(base_prob, 0.0, 0.0)
        assert adjusted == base_prob

    def test_home_injury_decreases_home_prob(self, predictor):
        """홈팀 부상 → 홈 승률 감소"""
        base_prob = 0.55
        adjusted = predictor.apply_injury_adjustment(base_prob, 5.0, 0.0)
        assert adjusted < base_prob

    def test_away_injury_increases_home_prob(self, predictor):
        """원정팀 부상 → 홈 승률 증가"""
        base_prob = 0.55
        adjusted = predictor.apply_injury_adjustment(base_prob, 0.0, 5.0)
        assert adjusted > base_prob

    def test_no_limit_on_adjustment(self, predictor):
        """큰 부상 영향도 한도 없이 적용"""
        base_prob = 0.5
        adjusted = predictor.apply_injury_adjustment(base_prob, 15.0, 0.0)
        expected = 0.35  # 50% - 15%
        assert abs(adjusted - expected) < 0.01

    def test_probability_bounds_maintained(self, predictor):
        """확률 경계 유지: 1% ~ 99%"""
        base_prob = 0.10
        adjusted = predictor.apply_injury_adjustment(base_prob, 20.0, 0.0)
        assert adjusted >= 0.01, f"Probability {adjusted} below 1%"
        assert adjusted <= 0.99, f"Probability {adjusted} above 99%"


# =============================================================================
# Level 4: E2E 통합 테스트
# =============================================================================

@pytest.mark.e2e
@pytest.mark.smoke
class TestFullPredictionPipeline:
    """전체 예측 파이프라인 E2E 검증"""

    def test_e2e_single_game_prediction(self, loader, predictor, et_today, team_epm, team_info):
        """단일 경기 예측 전체 플로우"""
        # 경기 찾기
        games = None
        test_date = et_today

        for offset in [0, 1, -1, 2, -2]:
            test_date = et_today + timedelta(days=offset)
            games = loader.get_games(test_date)
            if games:
                break

        if not games:
            pytest.skip("No games found for testing")

        game = games[0]
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        home_info = team_info.get(home_id, {})
        away_info = team_info.get(away_id, {})
        home_abbr = home_info.get('abbr', 'UNK')
        away_abbr = away_info.get('abbr', 'UNK')

        # 1. 피처 생성
        features = loader.build_v5_4_features(home_id, away_id, team_epm, test_date)
        assert len(features) == 5

        # 2. 기본 예측
        base_prob = predictor.predict_proba(features)
        assert 0.01 <= base_prob <= 0.99

        # 3. 부상 정보 (예정된 경기만)
        home_shift = 0.0
        away_shift = 0.0

        if game.get('game_status') == 1:
            home_summary = loader.get_injury_summary(home_abbr, test_date, team_epm)
            away_summary = loader.get_injury_summary(away_abbr, test_date, team_epm)
            home_shift = home_summary.get('total_prob_shift', 0.0)
            away_shift = away_summary.get('total_prob_shift', 0.0)

        # 4. 부상 조정
        final_prob = predictor.apply_injury_adjustment(base_prob, home_shift, away_shift)
        assert 0.01 <= final_prob <= 0.99

    def test_e2e_multiple_games_prediction(self, loader, predictor, et_today, team_epm):
        """다중 경기 예측 일관성"""
        games = None
        test_date = et_today

        for offset in range(-3, 4):
            test_date = et_today + timedelta(days=offset)
            games = loader.get_games(test_date)
            if games and len(games) >= 3:
                break

        if not games or len(games) < 3:
            pytest.skip("Not enough games for multi-game test")

        predictions = []

        for game in games[:5]:
            home_id = game['home_team_id']
            away_id = game['away_team_id']

            features = loader.build_v5_4_features(home_id, away_id, team_epm, test_date)
            prob = predictor.predict_proba(features)

            predictions.append({
                'game_id': game['game_id'],
                'prob': prob,
            })

        # 모든 예측이 유효 범위
        for pred in predictions:
            assert 0.01 <= pred['prob'] <= 0.99

    def test_e2e_finished_game_accuracy(self, loader, predictor, et_today, team_epm):
        """종료된 경기 적중률 검증"""
        finished_games = []

        for offset in range(-7, 0):
            test_date = et_today + timedelta(days=offset)
            games = loader.get_games(test_date)

            for game in games or []:
                if game.get('game_status') == 3 and game.get('home_score') is not None:
                    finished_games.append((test_date, game))

        if len(finished_games) < 5:
            pytest.skip("Not enough finished games for accuracy test")

        correct = 0
        total = 0

        for test_date, game in finished_games[:20]:
            home_id = game['home_team_id']
            away_id = game['away_team_id']

            features = loader.build_v5_4_features(home_id, away_id, team_epm, test_date)
            prob = predictor.predict_proba(features)

            predicted_home_win = prob >= 0.5
            actual_home_win = game['home_score'] > game['away_score']

            if predicted_home_win == actual_home_win:
                correct += 1
            total += 1

        accuracy = correct / total
        # 최소 40% 이상 (랜덤보다 나아야 함)
        assert accuracy >= 0.40, f"Accuracy {accuracy:.1%} below 40%"


@pytest.mark.e2e
class TestScenarioBasedE2E:
    """시나리오 기반 E2E 테스트"""

    def test_scenario_strong_vs_weak_team(self, predictor):
        """시나리오: 강팀 vs 약팀"""
        # OKC (강팀) @ WAS (약팀) 수준의 피처
        features = {
            'team_epm_diff': -10.0,  # 약팀 홈
            'sos_diff': -0.5,
            'bench_strength_diff': -3.0,
            'top5_epm_diff': -5.0,
            'ft_rate_diff': -0.01,
        }
        prob = predictor.predict_proba(features)
        # 약팀 홈이므로 낮은 확률
        assert prob < 0.30, f"Weak home should be <30%, got {prob:.1%}"

    def test_scenario_balanced_matchup(self, predictor, mock_balanced_features):
        """시나리오: 균형 매치업 (홈코트 이점 포함)"""
        prob = predictor.predict_proba(mock_balanced_features)
        # 모델에 홈코트 이점이 내재되어 있어 ~56% 수준
        assert 0.40 <= prob <= 0.60, f"Balanced game should be ~50-55%, got {prob:.1%}"


# =============================================================================
# Level 5: PredictionPipeline 테스트 (리팩토링 Phase 3)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.smoke
class TestPredictionPipeline:
    """PredictionPipeline Facade 테스트"""

    @pytest.fixture
    def pipeline(self, project_paths):
        """PredictionPipeline 인스턴스"""
        from app.services.prediction_pipeline import PredictionPipeline
        return PredictionPipeline(
            data_dir=project_paths["data_dir"],
            model_dir=project_paths["model_dir"]
        )

    def test_pipeline_initialization(self, pipeline):
        """파이프라인 초기화"""
        assert pipeline is not None
        assert pipeline.data_dir is not None
        assert pipeline.model_dir is not None

    def test_pipeline_predict_games(self, pipeline, et_today):
        """경기 예측"""
        # 경기 찾기
        predictions = None
        test_date = et_today

        for offset in range(-3, 4):
            test_date = et_today + timedelta(days=offset)
            predictions = pipeline.predict_games(test_date)
            if predictions:
                break

        if not predictions:
            pytest.skip("No games found for pipeline test")

        # 예측 결과 검증
        pred = predictions[0]
        assert pred.game_id is not None
        assert 0.01 <= pred.adjusted_prob <= 0.99
        assert pred.home_abbr != "UNK"
        assert pred.away_abbr != "UNK"

    def test_pipeline_model_info(self, pipeline):
        """모델 정보 조회"""
        info = pipeline.get_model_info()
        assert info is not None
        assert "n_features" in info
        assert info["n_features"] == 5


# =============================================================================
# 독립 실행 스크립트
# =============================================================================

def run_quick_test():
    """빠른 검증 테스트 (pytest 없이)"""
    print("="*70)
    print("  BucketsVision 빠른 검증 테스트")
    print("  시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)

    from app.services.predictor_v5 import V5PredictionService
    from app.services.data_loader import DataLoader
    from config.constants import TEAM_INFO, ABBR_TO_ID

    model_dir = project_root / "bucketsvision_v4" / "models"
    data_dir = project_root / "data"

    predictor = V5PredictionService(model_dir)
    loader = DataLoader(data_dir)

    et = pytz.timezone('America/New_York')
    et_today = datetime.now(et).date()

    print(f"\n  테스트 날짜: {et_today} (ET)")
    print(f"  현재 시즌: {get_season_from_date(et_today)}")

    results = {'passed': 0, 'failed': 0, 'details': []}

    def run_test(name, test_func):
        try:
            test_func()
            results['passed'] += 1
            print(f"  ✓ {name}")
        except AssertionError as e:
            results['failed'] += 1
            results['details'].append((name, str(e)))
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            results['failed'] += 1
            results['details'].append((name, str(e)))
            print(f"  ⚠ {name}: {e}")

    # 기본 테스트
    print("\n[1] 시즌 로직")
    print("-"*50)
    run_test("현재 시즌 = 2026", lambda: assert_(get_season_from_date(et_today) == 2026))

    # 모델 테스트
    print("\n[2] 모델 로드")
    print("-"*50)
    run_test("모델 로드", lambda: assert_(predictor.model is not None))
    run_test("스케일러 로드", lambda: assert_(predictor.scaler is not None))

    # API 테스트
    print("\n[3] API 데이터")
    print("-"*50)
    team_epm = loader.load_team_epm(et_today)
    run_test("팀 EPM 30팀", lambda: assert_(len(team_epm) == 30))

    # 예측 테스트
    print("\n[4] 예측 로직")
    print("-"*50)
    balanced = {'team_epm_diff': 0, 'sos_diff': 0, 'bench_strength_diff': 0, 'top5_epm_diff': 0, 'ft_rate_diff': 0}
    prob = predictor.predict_proba(balanced)
    run_test("균형 경기 ~50%", lambda: assert_(0.4 <= prob <= 0.6))

    # 결과
    print("\n" + "="*70)
    print(f"  결과: ✓ {results['passed']} / ✗ {results['failed']}")
    print("="*70)

    return results['failed'] == 0


def assert_(condition):
    if not condition:
        raise AssertionError("Assertion failed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--pytest':
        pytest.main([__file__, '-v'])
    else:
        success = run_quick_test()
        sys.exit(0 if success else 1)
