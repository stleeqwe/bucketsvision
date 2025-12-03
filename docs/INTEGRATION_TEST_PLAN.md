# BucketsVision 통합 테스트 계획서

**버전**: 1.2.0
**작성일**: 2025-12-03
**대상 시스템**: BucketsVision V5.4 NBA 예측 시스템

---

## 1. 개요

### 1.1 목적
본 문서는 BucketsVision NBA 예측 시스템의 통합 테스트 수행을 위한 체계적인 계획을 제시합니다.
이 테스트 프레임워크를 통해 시스템의 핵심 기능, API 연동, 예측 파이프라인, 데이터 무결성을
지속적으로 검증할 수 있습니다.

### 1.2 범위
| 영역 | 테스트 대상 |
|------|------------|
| API 연동 | DNT API, NBA Stats API, ESPN API, Odds API |
| 데이터 처리 | DataLoader, 피처 빌드, 캐시 관리 |
| 캐시 & TTL | 자동 갱신, 수동 갱신, TTL 정책, NBA PDF 자동 감지 |
| 예측 파이프라인 | V5.4 모델, 부상 영향 계산, 확률 보정 |
| 비즈니스 로직 | 시즌 계산, 팀 정보 매핑, B2B 감지 |
| E2E 플로우 | 전체 예측 파이프라인 통합 |

### 1.3 프로젝트 특성 (테스트 시 고려사항)

#### 핵심 원칙
1. **실시간 API 데이터 사용**: 모든 EPM/스탯 데이터는 API 실시간 호출 필수
2. **현재 시즌**: 2025-26 시즌 (season=2026)
3. **모델 아키텍처**: V5.4 Logistic Regression (5개 피처)
4. **부상 영향**: 후행 지표로 예측 후 적용 (v1.0.0)

#### 시즌 계산 규칙
```python
# 10월 이후는 다음 해 시즌
def get_season_from_date(game_date):
    if game_date.month >= 10:
        return game_date.year + 1
    return game_date.year
```

---

## 2. 테스트 레벨 및 유형

### 2.1 테스트 레벨 구조

```
┌────────────────────────────────────────────────────────────────┐
│                    Level 4: E2E 통합 테스트                     │
│   전체 예측 파이프라인 (API → 피처 → 예측 → 부상조정 → 출력)      │
├────────────────────────────────────────────────────────────────┤
│                    Level 3: 서비스 통합 테스트                   │
│   DataLoader ↔ Predictor ↔ InjuryCalculator 연동               │
├────────────────────────────────────────────────────────────────┤
│                    Level 2: API 통합 테스트                      │
│   DNT API, NBA Stats API, ESPN API, Odds API 연동              │
├────────────────────────────────────────────────────────────────┤
│                    Level 1: 단위 테스트                          │
│   개별 함수, 계산 로직, 유틸리티                                  │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 테스트 유형

| 유형 | 설명 | 실행 빈도 |
|------|------|----------|
| **Smoke Test** | 핵심 기능 빠른 검증 | 매 배포 |
| **Regression Test** | 기존 기능 정상 동작 확인 | 매 커밋 |
| **API Health Check** | 외부 API 연결 상태 확인 | 매일 |
| **Accuracy Validation** | 모델 정확도 검증 | 매주 |
| **Performance Test** | 응답 시간 및 리소스 사용 | 월별 |

---

## 3. 테스트 시나리오

### 3.1 Level 1: 단위 테스트

#### 3.1.1 시즌/날짜 로직 테스트
```python
class TestDateSeasonLogic:
    """날짜 및 시즌 로직 검증"""

    def test_current_season_is_2026(self):
        """현재 시즌이 2026인지 확인 (2025-26 시즌)"""

    def test_season_calculation_october_onwards(self):
        """10월 이후 날짜 → 다음 해 시즌"""
        # 2025-10-22 → 2026
        # 2025-11-28 → 2026
        # 2026-01-15 → 2026

    def test_season_calculation_before_october(self):
        """10월 이전 날짜 → 같은 해 시즌"""
        # 2026-09-30 → 2026

    def test_nba_api_season_string(self):
        """NBA API 시즌 문자열 변환"""
        # 2026 → "2025-26"
```

#### 3.1.2 Four Factors 계산 테스트
```python
class TestFourFactorsCalculation:
    """Four Factors 통계 계산 검증"""

    def test_efg_calculation(self):
        """eFG% = (FG + 0.5 * 3P) / FGA"""

    def test_ft_rate_calculation(self):
        """FT Rate = FT / FGA"""

    def test_orb_pct_calculation(self):
        """ORB% = ORB / (ORB + DRB)"""

    def test_zero_division_handling(self):
        """0으로 나눌 때 기본값 반환"""
```

#### 3.1.3 팀 정보 매핑 테스트
```python
class TestTeamInfoMapping:
    """팀 정보 매핑 검증"""

    def test_all_30_teams_exist(self):
        """30개 팀 데이터 완비"""

    def test_abbr_to_id_mapping(self):
        """팀 약어 → ID 매핑 정확성"""

    def test_team_colors_format(self):
        """팀 컬러 형식 (#XXXXXX)"""
```

### 3.2 Level 2: API 통합 테스트

#### 3.2.1 DNT API 테스트
```python
class TestDNTAPIIntegration:
    """DNT API 연동 검증"""

    def test_team_epm_returns_30_teams(self):
        """팀 EPM: 30개 팀 데이터 반환"""
        # 필수 필드: team_epm, team_oepm, team_depm, sos

    def test_team_epm_value_range(self):
        """EPM 값 범위: -15 ~ +15"""

    def test_player_epm_returns_sufficient_data(self):
        """선수 EPM: 최소 300명 이상"""
        # 필수 필드: player_id, player_name, team_id, tot, mpg

    def test_season_epm_endpoint(self):
        """시즌 전체 선수 EPM 조회"""

    def test_rate_limiting_compliance(self):
        """Rate Limit 준수 (0.7초 간격)"""
```

#### 3.2.2 NBA Stats API 테스트
```python
class TestNBAStatsAPIIntegration:
    """NBA Stats API 연동 검증"""

    def test_scoreboard_returns_games(self):
        """경기 스케줄 조회"""
        # 필수 필드: game_id, home_team_id, away_team_id, game_status

    def test_team_game_logs_for_current_season(self):
        """현재 시즌 팀 경기 로그"""
        # 2025-10-01 이후 데이터 존재 확인

    def test_player_game_logs_structure(self):
        """선수 경기 로그 구조"""
        # 필수 필드: PLAYER_ID, GAME_ID, MIN, PTS

    def test_finished_game_scores(self):
        """종료된 경기 점수 반환"""
        # home_score, away_score 존재

    def test_live_game_detection(self):
        """라이브 경기 감지"""
        # game_status == 2
```

#### 3.2.3 ESPN API 테스트
```python
class TestESPNAPIIntegration:
    """ESPN 부상 API 연동 검증"""

    CACHE_TTL_MINUTES = 30  # ESPN 캐시 TTL

    def test_out_players_retrieval(self):
        """Out 상태 선수 조회"""

    def test_gtd_players_retrieval(self):
        """GTD 상태 선수 조회"""

    def test_injury_data_structure(self):
        """부상 데이터 구조"""
        # player_name, status, detail, team_abbr, fantasy_status 필드

    def test_all_teams_supported(self):
        """모든 팀 부상자 조회 가능"""

    def test_team_abbr_normalization(self):
        """팀 약어 정규화"""
        # GS → GSW, NY → NYK 등

    def test_cache_validity_check(self):
        """캐시 유효성 검사 메서드"""
        # _is_cache_valid() 동작

    def test_cache_timestamp_tracking(self):
        """캐시 타임스탬프 추적"""
        # _cache_timestamp 업데이트

    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True 시 캐시 무시"""
```

#### 3.2.4 Odds API 테스트
```python
class TestOddsAPIIntegration:
    """배당 API 연동 검증"""

    def test_odds_retrieval(self):
        """배당 정보 조회"""

    def test_odds_structure(self):
        """배당 데이터 구조"""
        # spread, moneyline, total_line 필드

    def test_pinnacle_bookmaker(self):
        """Pinnacle 배당 우선"""
```

### 3.3 Level 3: 서비스 통합 테스트

#### 3.3.1 DataLoader 통합 테스트
```python
class TestDataLoaderIntegration:
    """DataLoader 서비스 통합 검증"""

    def test_initialization(self):
        """DataLoader 초기화"""

    def test_load_team_epm_caching(self):
        """팀 EPM 캐싱 동작"""
        # 날짜별 캐시 키 검증

    def test_load_player_epm_caching(self):
        """선수 EPM 시즌별 캐싱"""

    def test_get_games_returns_complete_data(self):
        """경기 데이터 완전성"""
        # game_id, teams, scores, b2b 포함

    def test_team_game_logs_column_normalization(self):
        """컬럼명 정규화"""
        # TEAM_ID → team_id 등

    def test_cache_invalidation(self):
        """캐시 초기화 동작"""
```

#### 3.3.2 V5.4 피처 빌드 테스트
```python
class TestV54FeatureBuild:
    """V5.4 피처 빌드 검증"""

    V54_FEATURES = [
        'team_epm_diff',
        'sos_diff',
        'bench_strength_diff',
        'top5_epm_diff',
        'ft_rate_diff',
    ]

    def test_feature_count_is_5(self):
        """5개 피처 반환"""

    def test_feature_names_match_model(self):
        """피처명이 모델과 일치"""

    def test_team_epm_diff_calculation(self):
        """team_epm_diff = home_epm - away_epm"""

    def test_bench_strength_calculation(self):
        """벤치 선수(6-10위 MPG) 평균 EPM"""

    def test_top5_epm_calculation(self):
        """상위 5명(MPG 기준) 평균 EPM"""

    def test_ft_rate_diff_range(self):
        """FT Rate 차이 범위: -0.3 ~ +0.3"""

    def test_no_nan_values(self):
        """NaN 값 없음"""
```

#### 3.3.3 V5.4 모델 예측 테스트
```python
class TestV54ModelPrediction:
    """V5.4 모델 예측 검증"""

    def test_model_and_scaler_loaded(self):
        """모델 및 스케일러 로드"""

    def test_model_metadata(self):
        """모델 메타데이터 검증"""
        # version: 5.4.0
        # n_features: 5
        # overall_accuracy: ~0.78

    def test_balanced_game_around_50_percent(self):
        """균형 경기(모든 피처 0) → ~50%"""

    def test_strong_home_above_70_percent(self):
        """강팀 홈 경기 → 70% 이상"""

    def test_weak_home_below_30_percent(self):
        """약팀 홈 경기 → 30% 이하"""

    def test_prediction_monotonicity(self):
        """피처 증가 → 확률 증가 (단조성)"""

    def test_probability_range_within_bounds(self):
        """확률 범위: 1% ~ 99%"""
```

#### 3.3.4 부상 영향 계산 테스트
```python
class TestInjuryImpactCalculation:
    """부상 영향 계산 검증 (v1.1.0)"""

    # GTD 상태별 결장 확률 가중치
    GTD_WEIGHTS = {
        "Out": 1.0,
        "Doubtful": 0.75,
        "Questionable": 0.50,
        "Probable": 0.25,
    }

    def test_calculator_initialization(self):
        """AdvancedInjuryImpactCalculator 초기화"""

    def test_version_is_1_1_0(self):
        """버전 1.1.0 확인"""

    def test_player_finding_exact_match(self):
        """선수 이름 정확 매칭"""

    def test_player_finding_fuzzy_match(self):
        """선수 이름 퍼지 매칭 (유사도 0.8+)"""

    def test_traded_player_handling(self):
        """이적 선수 현재 팀 기준 처리"""

    def test_eligibility_conditions(self):
        """적용 조건 검증"""
        # EPM > 0, MPG >= 12, 출전율 > 1/3

    def test_expected_win_prob_calculation(self):
        """기대 승률 계산: 0.5 - (opp_epm * 0.03)"""

    def test_performance_based_impact(self):
        """성과 기반 영향력 계산"""
        # played_avg - missed_avg

    def test_prob_shift_formula(self):
        """prob_shift = EPM * 0.02 * normalized_diff"""

    def test_fallback_for_no_missed_games(self):
        """미출전 데이터 없을 때 폴백"""
        # prob_shift = EPM * 0.02

    def test_no_limit_on_adjustment(self):
        """부상 조정 한도 없음"""
```

#### 3.3.5 GTD 세분화 테스트 (v1.1.0)
```python
class TestGTDStatusWeights:
    """GTD 상태 세분화 검증 (ESPN + NBA PDF 병합)"""

    def test_out_status_100_percent_weight(self):
        """Out 상태 → 100% 결장 가중치"""
        # applied_shift = prob_shift * 1.0

    def test_doubtful_status_75_percent_weight(self):
        """Doubtful 상태 → 75% 결장 가중치"""
        # applied_shift = prob_shift * 0.75

    def test_questionable_status_50_percent_weight(self):
        """Questionable 상태 → 50% 결장 가중치"""
        # applied_shift = prob_shift * 0.50

    def test_probable_status_25_percent_weight(self):
        """Probable 상태 → 25% 결장 가중치"""
        # applied_shift = prob_shift * 0.25

    def test_espn_gtd_default_50_percent(self):
        """ESPN Day-To-Day (NBA PDF 없음) → 기본 50%"""
        # NBA PDF에서 세부 상태 못 찾으면 50% 적용

    def test_nba_pdf_overrides_espn_gtd(self):
        """NBA PDF 상태가 ESPN GTD 오버라이드"""
        # ESPN: Day-To-Day + NBA: Questionable → 50% 적용

    def test_injury_status_enum_play_probability(self):
        """InjuryStatus enum의 play_probability 속성"""
        # AVAILABLE: 1.0, PROBABLE: 0.75, QUESTIONABLE: 0.50
        # DOUBTFUL: 0.25, OUT: 0.0
```

#### 3.3.6 ESPN + NBA PDF 병합 테스트
```python
class TestESPNNBAPDFMerge:
    """ESPN과 NBA PDF 부상 정보 병합 검증"""

    def test_espn_as_primary_source(self):
        """ESPN이 기본 데이터 소스"""
        # ESPN 부상자 리스트가 기준

    def test_nba_pdf_enriches_gtd_status(self):
        """NBA PDF로 GTD 상태 세분화"""
        # ESPN GTD → NBA PDF에서 Probable/Questionable/Doubtful 확인

    def test_player_name_matching_for_merge(self):
        """병합 시 선수 이름 매칭"""
        # 정확 매칭 또는 부분 매칭

    def test_team_abbr_matching_for_merge(self):
        """병합 시 팀 약어 매칭"""
        # ESPN team_abbr == NBA team_abbr

    def test_merge_result_structure(self):
        """병합 결과 구조"""
        # {player_name_team: {espn_status, nba_status, final_status, play_probability}}

    def test_nba_pdf_not_found_uses_espn_only(self):
        """NBA PDF 없으면 ESPN만 사용"""
        # NBA PDF 조회 실패 → ESPN 기본값 사용

    def test_player_not_in_nba_pdf_uses_default(self):
        """NBA PDF에 선수 없으면 기본값 사용"""
        # ESPN GTD + NBA PDF에 없음 → 50% 적용
```

#### 3.3.7 NBA Injury Client 테스트
```python
class TestNBAInjuryClient:
    """NBA 공식 Injury Report 클라이언트 검증"""

    def test_pdf_url_pattern(self):
        """PDF URL 패턴 검증"""
        # https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{HH}PM.pdf

    def test_update_times(self):
        """PDF 업데이트 시간"""
        # ["01PM", "05PM", "07PM", "09PM"]

    def test_get_latest_pdf_url(self):
        """최신 PDF URL 조회"""
        # 최신 시간부터 역순 확인

    def test_pdf_parsing_extracts_injuries(self):
        """PDF 파싱으로 부상 정보 추출"""
        # player_name, team_abbr, status, reason

    def test_player_name_normalization(self):
        """선수 이름 정규화 (LastName,FirstName → FirstName LastName)"""

    def test_team_name_to_abbr_mapping(self):
        """팀명 → 약어 매핑"""
        # "LosAngelesLakers" → "LAL"

    def test_injury_status_enum(self):
        """InjuryStatus enum 파싱"""
        # Available, Probable, Questionable, Doubtful, Out

    def test_fetch_injuries_returns_team_grouped(self):
        """부상 정보 팀별 그룹화 반환"""
        # {team_abbr: [NBAInjury, ...]}

    def test_get_player_status(self):
        """특정 선수 상태 조회"""
        # player_name + team_abbr → InjuryStatus
```

#### 3.3.8 부상 조정 적용 테스트
```python
class TestInjuryAdjustmentApplication:
    """부상 조정 적용 검증"""

    def test_no_adjustment_when_no_injuries(self):
        """부상 없을 때 확률 변화 없음"""

    def test_home_injury_decreases_home_prob(self):
        """홈팀 부상 → 홈 승률 감소"""

    def test_away_injury_increases_home_prob(self):
        """원정팀 부상 → 홈 승률 증가"""

    def test_probability_bounds_maintained(self):
        """확률 경계 유지: 1% ~ 99%"""

    def test_large_shift_applied_without_cap(self):
        """큰 부상 영향도 한도 없이 적용"""

    def test_gtd_weighted_adjustment(self):
        """GTD 가중치 적용 조정"""
        # Doubtful: 75% 적용, Questionable: 50% 적용

    def test_multiple_gtd_players_cumulative(self):
        """다중 GTD 선수 누적 영향"""
        # 각 선수별 가중치 적용 후 합산

    def test_out_and_gtd_combined(self):
        """Out + GTD 혼합 상황"""
        # Out: 100% + Questionable: 50% 각각 적용
```

### 3.6 캐시 & TTL 정책 테스트

> **캐시 정책 요약 (2025-12-03 기준)**
>
> | 데이터 | TTL | 자동 갱신 | 수동 갱신 |
> |--------|-----|----------|----------|
> | ESPN 부상 | 30분 | TTL 만료 시 | ✅ |
> | NBA PDF | - | 같은 날 새 PDF 감지 시 | ✅ |
> | Team EPM | 60분 | TTL 만료 시 | ✅ |
> | 배당 정보 | 세션 | - | ✅ |
>
> **갱신 트리거:**
> - 자동: TTL 만료 / 새 PDF 발견
> - 수동: 🔄 강제 새로고침 버튼

#### 3.6.1 TTL 기반 자동 갱신 테스트
```python
class TestCacheTTLPolicy:
    """캐시 TTL 정책 검증"""

    # TTL 설정값
    ESPN_TTL_MINUTES = 30
    TEAM_EPM_TTL_MINUTES = 60

    def test_espn_cache_ttl_30_minutes(self):
        """ESPN 부상 캐시 30분 TTL"""
        # 1. 첫 번째 호출 → API 호출
        # 2. 즉시 두 번째 호출 → 캐시 반환
        # 3. 30분 후 호출 → API 재호출

    def test_espn_cache_expires_after_ttl(self):
        """ESPN 캐시 TTL 만료 시 자동 재호출"""
        # _cache_timestamp + 30분 경과 시 _is_cache_valid() == False

    def test_team_epm_cache_ttl_60_minutes(self):
        """Team EPM 캐시 60분 TTL"""
        # 1. 첫 번째 호출 → DNT API 호출
        # 2. 즉시 두 번째 호출 → 캐시 반환
        # 3. 60분 후 호출 → DNT API 재호출

    def test_team_epm_cache_expires_after_ttl(self):
        """Team EPM 캐시 TTL 만료 시 자동 재호출"""
        # _team_epm_timestamp + 60분 경과 시 _is_team_epm_cache_valid() == False

    def test_cache_timestamp_updated_on_fetch(self):
        """API 호출 시 캐시 타임스탬프 갱신"""
        # fetch 후 _cache_timestamp == datetime.now()

    def test_ttl_check_uses_current_time(self):
        """TTL 검사에 현재 시간 사용"""
        # 시간 mock으로 TTL 경과 시뮬레이션
```

#### 3.6.2 NBA PDF 자동 감지 테스트
```python
class TestNBAPDFAutoRefresh:
    """NBA PDF 자동 갱신 검증"""

    # PDF 업데이트 시간 (ET)
    UPDATE_TIMES = ["01PM", "05PM", "07PM", "09PM"]

    def test_pdf_time_tracking(self):
        """마지막 조회 PDF 시간 기록"""
        # _last_pdf_time[date] == "01PM"

    def test_newer_pdf_detection(self):
        """더 최신 PDF 존재 감지"""
        # 01PM 캐시 → 05PM PDF 발견 → True

    def test_auto_refresh_on_newer_pdf(self):
        """새 PDF 발견 시 자동 갱신"""
        # _has_newer_pdf() == True → 자동 재조회

    def test_no_refresh_when_latest_cached(self):
        """최신 PDF 캐시 시 재조회 안함"""
        # 09PM 캐시 → 더 이상 새 PDF 없음 → 캐시 반환

    def test_pdf_time_sequence(self):
        """PDF 시간 순서 확인"""
        # 01PM → 05PM → 07PM → 09PM

    def test_same_day_auto_refresh_only(self):
        """같은 날만 자동 갱신"""
        # target_date == date.today() 일 때만 자동 감지

    def test_past_date_uses_cache(self):
        """과거 날짜는 캐시 사용"""
        # 과거 날짜 조회 → 캐시 반환 (자동 갱신 안함)
```

#### 3.6.3 수동 갱신 (새로고침 버튼) 테스트
```python
class TestManualRefresh:
    """수동 갱신 (새로고침 버튼) 검증"""

    def test_refresh_button_clears_all_caches(self):
        """새로고침 버튼 → 모든 캐시 초기화"""
        # clear_all_caches() 호출 검증

    def test_clear_all_api_caches_called(self):
        """clear_all_api_caches() 호출 검증"""
        # Team EPM + ESPN + NBA PDF + Odds 모두 초기화

    def test_espn_cache_cleared(self):
        """ESPN 캐시 초기화"""
        # espn_client._cache == {}
        # espn_client._cache_timestamp == None

    def test_nba_pdf_cache_cleared(self):
        """NBA PDF 캐시 초기화"""
        # nba_injury_client._cache == {}
        # nba_injury_client._last_pdf_time == {}

    def test_team_epm_cache_cleared(self):
        """Team EPM 캐시 초기화"""
        # _team_epm_date_cache == {}
        # _team_epm_timestamp == None

    def test_odds_cache_cleared(self):
        """배당 캐시 초기화"""
        # _odds_cache == None

    def test_streamlit_caches_cleared(self):
        """Streamlit 캐시 초기화"""
        # st.cache_data.clear()
        # st.cache_resource.clear()

    def test_refresh_forces_api_recall(self):
        """갱신 후 API 재호출"""
        # 새로고침 → 다음 요청 시 모든 API 재호출

    def test_last_refresh_time_updated(self):
        """마지막 갱신 시간 업데이트"""
        # session_state["last_refresh_time"] 갱신
```

#### 3.6.4 캐시 독립성 테스트
```python
class TestCacheIsolation:
    """캐시 독립성 검증"""

    def test_espn_ttl_independent_of_team_epm(self):
        """ESPN TTL과 Team EPM TTL 독립"""
        # ESPN 만료 시 Team EPM 영향 없음

    def test_nba_pdf_independent_of_espn(self):
        """NBA PDF 캐시와 ESPN 캐시 독립"""
        # 각각 별도 캐시 유지

    def test_date_based_cache_isolation(self):
        """날짜별 캐시 격리"""
        # 2025-12-03 캐시 ≠ 2025-12-04 캐시

    def test_team_epm_date_key_isolation(self):
        """Team EPM 날짜 키 격리"""
        # cache_key별 독립 저장
```

#### 3.6.5 에러 상황 캐시 테스트
```python
class TestCacheErrorHandling:
    """캐시 에러 처리 검증"""

    def test_api_failure_uses_stale_cache(self):
        """API 실패 시 만료된 캐시 사용 (fallback)"""
        # API 에러 → 이전 캐시 반환

    def test_api_failure_logs_warning(self):
        """API 실패 시 경고 로그"""
        # logger.warning 호출 확인

    def test_empty_response_clears_cache(self):
        """빈 응답 시 캐시 초기화 안함"""
        # 빈 응답 → 기존 캐시 유지

    def test_partial_failure_handling(self):
        """부분 실패 처리"""
        # 일부 API 실패 → 성공한 API 캐시 유지
```

#### 3.6.6 프론트엔드 갱신 표시 테스트
```python
class TestRefreshUIDisplay:
    """프론트엔드 갱신 표시 검증"""

    def test_last_refresh_time_displayed(self):
        """마지막 갱신 시간 표시"""
        # "🔄 마지막 갱신: HH:MM:SS"

    def test_current_time_displayed(self):
        """현재 시간 표시"""
        # "현재: HH:MM KST"

    def test_initial_refresh_time_on_app_start(self):
        """앱 시작 시 초기 갱신 시간 설정"""
        # session_state["last_refresh_time"] 초기화

    def test_refresh_time_updates_on_button_click(self):
        """버튼 클릭 시 갱신 시간 업데이트"""
        # 이전 시간 → 새 시간으로 변경
```

### 3.7 Level 4: E2E 통합 테스트

#### 3.7.1 전체 예측 파이프라인 테스트
```python
class TestFullPredictionPipeline:
    """전체 예측 파이프라인 E2E 검증"""

    def test_e2e_single_game_prediction(self):
        """단일 경기 예측 전체 플로우"""
        # 1. 팀 EPM 로드 (DNT API)
        # 2. 경기 스케줄 조회 (NBA Stats API)
        # 3. V5.4 피처 생성
        # 4. 기본 예측 확률 계산
        # 5. 부상 정보 조회 (ESPN API)
        # 6. 부상 조정 적용
        # 7. 최종 확률 반환

    def test_e2e_multiple_games_prediction(self):
        """다중 경기 예측 일관성"""
        # 모든 경기 예측 완료 및 유효 범위 확인

    def test_e2e_finished_game_accuracy(self):
        """종료된 경기 적중률 검증"""
        # 최소 40% 이상 (랜덤보다 나음)

    def test_e2e_with_injuries(self):
        """부상 정보 포함 예측"""
        # 부상 조정 후 확률 변화 확인

    def test_e2e_odds_integration(self):
        """배당 정보 통합"""
        # 경기별 배당 데이터 병합
```

#### 3.7.2 프론트엔드 데이터 플로우 테스트
```python
class TestFrontendDataFlow:
    """프론트엔드 데이터 플로우 검증"""

    def test_game_card_data_structure(self):
        """게임 카드 렌더링 데이터"""
        # game_id, game_time, teams, scores, b2b

    def test_prediction_display_data(self):
        """예측 표시 데이터"""
        # home_win_prob, away_win_prob, predicted_winner, confidence

    def test_injury_display_data(self):
        """부상 표시 데이터"""
        # out_players, gtd_players, prob_shift

    def test_odds_display_data(self):
        """배당 표시 데이터"""
        # spread, moneyline, total
```

#### 3.7.3 시나리오 기반 테스트
```python
class TestScenarioBasedE2E:
    """시나리오 기반 E2E 테스트"""

    def test_scenario_strong_vs_weak_team(self):
        """시나리오: 강팀 vs 약팀"""
        # OKC @ WAS → 높은 원정 승률

    def test_scenario_b2b_away_team(self):
        """시나리오: 원정팀 B2B"""
        # 홈팀 유리

    def test_scenario_star_player_injured(self):
        """시나리오: 스타 선수 부상"""
        # 유의미한 확률 변화

    def test_scenario_multiple_injuries(self):
        """시나리오: 다중 부상"""
        # 누적 영향 계산

    def test_scenario_no_games_day(self):
        """시나리오: 경기 없는 날"""
        # 빈 리스트 반환
```

---

## 4. 테스트 데이터 전략

### 4.1 실시간 데이터 (Live Data)
- **대상**: API 통합 테스트, E2E 테스트
- **특징**: 실제 API 호출, 최신 데이터 사용
- **주의**: Rate Limit 준수, API 키 필요

### 4.2 Mock 데이터
- **대상**: 단위 테스트, 경계값 테스트
- **용도**: API 의존성 제거, 특수 케이스 테스트

```python
# Mock 데이터 예시
MOCK_TEAM_EPM = {
    1610612738: {  # BOS
        "team_epm": 8.5,
        "team_oepm": 7.2,
        "team_depm": 1.3,
        "sos": 0.5,
        "team_alias": "BOS"
    },
    1610612765: {  # DET
        "team_epm": -6.2,
        "team_oepm": -4.8,
        "team_depm": -1.4,
        "sos": -0.3,
        "team_alias": "DET"
    }
}

MOCK_V54_FEATURES = {
    "team_epm_diff": 8.0,
    "sos_diff": 0.5,
    "bench_strength_diff": 3.0,
    "top5_epm_diff": 4.0,
    "ft_rate_diff": 0.02
}
```

### 4.3 Fixture 데이터
```python
@pytest.fixture(scope="module")
def predictor():
    """V5.4 예측 서비스 (모듈 범위)"""
    model_dir = Path("bucketsvision_v4/models")
    return V5PredictionService(model_dir)

@pytest.fixture(scope="module")
def loader():
    """데이터 로더 (모듈 범위)"""
    data_dir = Path("data")
    return DataLoader(data_dir)

@pytest.fixture
def et_today():
    """미국 동부 시간 기준 오늘"""
    import pytz
    et = pytz.timezone('America/New_York')
    return datetime.now(et).date()
```

---

## 5. 테스트 실행 가이드

### 5.1 실행 명령어

```bash
# 전체 테스트 실행
python -m pytest tests/ -v

# 특정 테스트 파일 실행
python -m pytest tests/test_v5_4_comprehensive.py -v

# 특정 클래스 실행
python -m pytest tests/test_integration.py::TestDNTAPIIntegration -v

# 특정 테스트 케이스 실행
python -m pytest tests/test_integration.py::TestDNTAPIIntegration::test_team_epm_returns_30_teams -v

# 마킹된 테스트 실행
python -m pytest -m "smoke" -v  # Smoke 테스트만
python -m pytest -m "api" -v     # API 테스트만
python -m pytest -m "slow" -v    # 느린 테스트만

# 직접 실행 (pytest 없이)
python tests/test_v5_4_comprehensive.py
```

### 5.2 테스트 마커

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: 핵심 기능 빠른 검증")
    config.addinivalue_line("markers", "api: 외부 API 연동 테스트")
    config.addinivalue_line("markers", "slow: 실행 시간이 긴 테스트")
    config.addinivalue_line("markers", "e2e: End-to-End 테스트")
    config.addinivalue_line("markers", "accuracy: 모델 정확도 검증")
```

### 5.3 환경 설정

```bash
# 필수 환경변수
export DNT_API_KEY="your_dnt_api_key"
export ODDS_API_KEY="your_odds_api_key"

# 테스트 환경
export TEST_ENV="development"
export LOG_LEVEL="DEBUG"
```

---

## 6. 테스트 품질 기준

### 6.1 통과 기준

| 테스트 유형 | 통과 기준 |
|------------|----------|
| 단위 테스트 | 100% 통과 |
| API 통합 테스트 | 95% 이상 (API 장애 허용) |
| 서비스 통합 테스트 | 100% 통과 |
| E2E 테스트 | 95% 이상 |
| 정확도 검증 | 기준 정확도 유지 (±2%) |

### 6.2 정확도 기준 (V5.4)

| 메트릭 | 기준값 | 허용 범위 |
|--------|--------|----------|
| 전체 정확도 | 78.05% | 76% ~ 80% |
| 고신뢰(≥70%) 정확도 | 87.88% | 85% ~ 90% |
| 저신뢰(<70%) 정확도 | 71.43% | 68% ~ 75% |
| 확률 범위 | 8.2% ~ 94.8% | 5% ~ 95% |

### 6.3 성능 기준

| 항목 | 기준 |
|------|------|
| 단일 예측 응답 시간 | < 500ms |
| 전체 경기 예측 (10경기) | < 5초 |
| 팀 EPM 로드 | < 2초 |
| 메모리 사용량 | < 500MB |

---

## 7. 테스트 자동화

### 7.1 CI/CD 파이프라인

```yaml
# .github/workflows/test.yml
name: Integration Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 5 * * *'  # 매일 05:00 UTC

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run Smoke Tests
      run: pytest tests/ -m smoke -v
      env:
        DNT_API_KEY: ${{ secrets.DNT_API_KEY }}

    - name: Run Full Tests
      run: pytest tests/ -v --cov=app --cov=src
      env:
        DNT_API_KEY: ${{ secrets.DNT_API_KEY }}
        ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
```

### 7.2 LaunchD 스케줄 (macOS)

```xml
<!-- com.bucketsvision.integration-tests.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.bucketsvision.integration-tests</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>-m</string>
        <string>pytest</string>
        <string>tests/test_v5_4_comprehensive.py</string>
        <string>-v</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/stlee/Desktop/bucketsvision</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
</dict>
</plist>
```

---

## 8. 테스트 보고서

### 8.1 보고서 형식

```
================================================================================
  BucketsVision V5.4 통합 테스트 결과
  실행 시간: 2025-12-03 14:30:00
================================================================================

[1] API 데이터 수집 테스트
--------------------------------------------------
  ✓ DNT API - 팀 EPM (30팀)
  ✓ DNT API - 선수 EPM (300+)
  ✓ NBA Stats API - 경기 로드
  ✓ ESPN API - 부상 정보

[2] 날짜/시즌 로직 테스트
--------------------------------------------------
  ✓ 현재 시즌 = 2026
  ✓ 2025-11-28 → 2026
  ✓ 2026-03-15 → 2026

[3] V5.4 피처 빌드 테스트
--------------------------------------------------
  ✓ 피처 5개 생성
  ✓ team_epm_diff 존재
  ✓ sos_diff 존재
  ✓ bench_strength_diff 존재
  ✓ top5_epm_diff 존재
  ✓ ft_rate_diff 존재

[4] 모델 예측 로직 테스트
--------------------------------------------------
  ✓ 모델 로드 완료
  ✓ 스케일러 로드 완료
  ✓ 모델 버전 5.4.0
  ✓ 피처 수 5개
  ✓ 균형 경기 ~50%
  ✓ 강팀 홈 >70%

[5] Injury Impact 테스트
--------------------------------------------------
  ✓ 부상 조정 한도 없음 (15% 적용)
  ✓ 확률 하한 1%

[6] E2E 통합 테스트
--------------------------------------------------
  ✓ E2E 파이프라인 완료

    샘플 경기: UTA @ HOU
    기본 확률: 58.2%
    최종 확률: 61.5%

================================================================================
  [테스트 결과 요약]
================================================================================
  ✓ 통과: 24
  ✗ 실패: 0
  ⚠ 스킵: 0
================================================================================
```

### 8.2 실패 분석 템플릿

```markdown
## 테스트 실패 분석

### 실패 테스트
- 테스트명: test_team_epm_returns_30_teams
- 클래스: TestDNTAPIIntegration
- 파일: tests/test_integration.py:87

### 오류 내용
```
AssertionError: Expected 30 teams, got 28
```

### 원인 분석
- DNT API 응답에서 2개 팀 데이터 누락
- 영향받은 팀: TBD (확인 필요)

### 조치 사항
1. DNT API 응답 로깅 추가
2. 누락 팀 식별
3. API 제공자에 문의 (필요 시)

### 해결 상태
- [ ] 분석 완료
- [ ] 수정 완료
- [ ] 재테스트 통과
```

---

## 9. 트러블슈팅

### 9.1 일반적인 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| API 타임아웃 | 네트워크 지연 | 타임아웃 증가, 재시도 로직 |
| Rate Limit 초과 | 빠른 연속 요청 | 요청 간격 0.7초 이상 |
| 팀 데이터 누락 | API 응답 불완전 | 기본값 사용, 에러 로깅 |
| 선수 매칭 실패 | 이름 불일치 | 퍼지 매칭, Fallback 로직 |
| 확률 범위 초과 | 극단적 피처값 | 확률 경계 클리핑 |

### 9.2 디버깅 가이드

```python
# 로깅 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# API 응답 확인
import json
response = dnt_client.get_team_epm()
print(json.dumps(response, indent=2))

# 피처 값 검증
features = loader.build_v5_4_features(home_id, away_id, team_epm, date)
for name, value in features.items():
    print(f"{name}: {value}")

# 예측 분석
prob = predictor.predict_proba(features)
print(f"Base Probability: {prob:.4f}")
```

---

## 10. 부록

### 10.1 테스트 파일 구조

```
tests/
├── __init__.py
├── conftest.py                    # Pytest 설정 및 공통 Fixture
├── test_v5_4_comprehensive.py     # V5.4 종합 테스트 (기존)
├── test_e2e.py                    # E2E 테스트 (기존)
├── test_integration.py            # 통합 테스트 (신규)
│   ├── TestDNTAPIIntegration
│   ├── TestNBAStatsAPIIntegration
│   ├── TestESPNAPIIntegration
│   ├── TestOddsAPIIntegration
│   ├── TestDataLoaderIntegration
│   ├── TestV54FeatureBuild
│   ├── TestV54ModelPrediction
│   ├── TestInjuryImpactCalculation
│   └── TestFullPredictionPipeline
├── test_unit.py                   # 단위 테스트 (신규)
│   ├── TestDateSeasonLogic
│   ├── TestFourFactorsCalculation
│   └── TestTeamInfoMapping
└── fixtures/
    ├── mock_team_epm.json
    ├── mock_player_epm.json
    └── mock_game_data.json
```

### 10.2 참조 문서

- [CLAUDE.md](/CLAUDE.md) - 프로젝트 가이드
- [MODEL_COMPARISON_REPORT.md](/bucketsvision_v4/docs/MODEL_COMPARISON_REPORT.md) - 모델 비교
- [v5_4_metadata.json](/bucketsvision_v4/models/v5_4_metadata.json) - 모델 메타데이터
- [injury_impact_v1_metadata.json](/bucketsvision_v4/models/injury_impact_v1_metadata.json) - 부상 영향 메타데이터

### 10.3 버전 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.3.0 | 2025-12-03 | Phase 2 리팩토링 검증 완료 (StatCalculator, InjuryService, GameScheduleService) |
| 1.2.0 | 2025-12-03 | 인저리 임팩트 v1.1.0 반영 (GTD 세분화, ESPN+NBA PDF 병합) |
| 1.1.0 | 2025-12-03 | 캐시 & TTL 정책 테스트 추가 (섹션 3.6) |
| 1.0.0 | 2025-12-03 | 초기 버전 |

### 10.4 최신 테스트 실행 결과 (2025-12-03)

```
================================================================================
  Phase 2 리팩토링 검증 통합 테스트 결과
================================================================================

테스트 파일: test_v5_4_comprehensive.py, test_integration.py, test_e2e.py
실행 시간: 68.27초
총 테스트: 92개

================================================================================
  [Level 1: 단위 테스트]
================================================================================
  ✓ TestDateSeasonLogic: 4/4 통과
    - test_current_season_is_2026
    - test_season_calculation_october
    - test_season_calculation_before_october
    - test_data_loader_uses_correct_season

  ✓ TestV54FeatureBuild: 6/6 통과
    - test_feature_names_match_model
    - test_build_v5_4_features_returns_all
    - test_team_epm_diff_calculation
    - test_bench_strength_diff_calculation
    - test_top5_epm_diff_calculation
    - test_ft_rate_diff_range

  ✓ TestFrontendData: 4/4 통과
    - test_team_info_complete
    - test_abbr_to_id_mapping
    - test_game_card_data_structure
    - test_prediction_data_for_frontend

================================================================================
  [Level 2: API 통합 테스트]
================================================================================
  ✓ TestAPIDataCollection: 5/5 통과
    - test_dnt_api_team_epm
    - test_dnt_api_player_epm
    - test_nba_stats_api_games
    - test_nba_stats_api_team_game_logs
    - test_espn_injury_api

  ✓ TestDNTAPIIntegration: 3/3 통과
  ✓ TestNBAStatsAPIIntegration: 4/4 통과
  ✓ TestESPNAPIIntegration: 2/2 통과
  ✓ TestOddsAPIIntegration: 1/1 통과

================================================================================
  [Level 3: 서비스 통합 테스트]
================================================================================
  ✓ TestModelPrediction: 6/6 통과
    - test_model_loaded
    - test_model_info
    - test_predict_proba_range
    - test_predict_strong_vs_weak
    - test_prediction_monotonicity
    - test_probability_range_matches_metadata

  ✓ TestInjuryImpact: 6/6 통과
    - test_injury_calculator_loads
    - test_injury_impact_version (v1.1.0)
    - test_injury_adjustment_no_limit
    - test_injury_adjustment_probability_bounds
    - test_injury_summary_structure
    - test_injury_impact_conditions

  ✓ TestDataLoaderIntegration: 3/3 통과
  ✓ TestV54ModelPrediction: 7/7 통과
  ✓ TestInjuryImpactCalculation: 3/3 통과
  ✓ TestInjuryAdjustmentApplication: 5/5 통과

================================================================================
  [Level 4: E2E 통합 테스트]
================================================================================
  ✓ TestE2EIntegration: 3/3 통과
    - test_full_prediction_pipeline
    - test_multiple_games_prediction
    - test_finished_game_accuracy

  ✓ TestFullPredictionPipeline: 3/3 통과
  ✓ TestScenarioBasedE2E: 2/2 통과
  ✓ TestPredictionPipeline: 3/3 통과
  ✓ test_e2e.py: 8/8 통과

================================================================================
  [Phase 2 리팩토링 신규 서비스 검증]
================================================================================
  ✓ StatCalculator
    - calc_efg(): 0.589 (정상)
    - calc_ft_rate(): 0.400 (정상)
    - calc_streak(): 3 (정상)
    - default_team_stats(): 정상

  ✓ PlayerStatCalculator
    - get_team_players(): 6명 (정상)
    - calc_rotation_epm(): 2.304 (정상)
    - calc_bench_strength(): -1.000 (정상)
    - calc_top5_epm(): 2.300 (정상)

  ✓ InjuryService
    - 초기화: 정상
    - clear_cache(): 정상
    - get_injuries(): 정상
    - get_gtd_players(): 정상

  ✓ GameScheduleService
    - 초기화: 정상
    - GAME_STATUS 상수: 정상
    - get_games(): 9개 경기 (정상)
    - _should_use_cache(): 정상

  ✓ DataLoader 위임
    - _game_schedule_service: GameScheduleService
    - _injury_service: InjuryService
    - clear_injury_caches() 위임: 정상

================================================================================
  [테스트 결과 요약]
================================================================================
  ✓ 통과: 92
  ✗ 실패: 0
  ⚠ 경고: 85 (sklearn feature name 경고, pytest return 경고 - 무해)
================================================================================

검증 완료:
- V5.4 모델 예측 파이프라인: 정상
- Phase 2 리팩토링 서비스: 정상 작동
- 모든 API 연동: 정상
- 부상 영향 계산 (v1.1.0): 정상
```

---

*이 문서는 BucketsVision 프로젝트의 품질 보증을 위한 통합 테스트 가이드입니다.*
