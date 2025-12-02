# BucketsVision - Claude Code 가이드

## 핵심 정보

### 현재 시즌 정보 (중요!)
- **현재 진행 시즌**: 2025-26 시즌 (25-26)
- **시즌 시작일**: 2025년 10월
- **시즌 코드**: 2026 (NBA API 기준 종료 연도)
- **예시**: 2025년 11월 경기 → 2026 시즌 데이터 사용

### 시즌 계산 규칙
```python
# 10월 이후는 다음 해 시즌
def get_season_from_date(game_date):
    if game_date.month >= 10:
        return game_date.year + 1
    return game_date.year

# 2025-11-28 → 2026 시즌
# 2026-01-15 → 2026 시즌
# 2026-06-20 → 2026 시즌 (플레이오프)
```

### 절대 하지 말아야 할 실수
1. 2024년 시즌 데이터를 현재 시즌으로 착각하지 말 것
2. `date(2024, ...)` 대신 `date(2025, ...)`를 사용할 것
3. 시즌 테스트 시 반드시 현재 날짜 기준으로 확인

## 모델 학습/검증 원칙 (필수 준수)

### 데이터 분리 규칙
```
Train Set (과거 시즌)        Validation Set (현재 시즌)
─────────────────────────────────────────────────────
2023, 2024, 2025            2026
(22-23, 23-24, 24-25)       (25-26, 현재 진행중)

모델 학습용                  성능 검증용 (학습 금지!)
```

### 절대 준수 사항
1. **현재 시즌(25-26, season=2026)은 검증용으로만 사용** - 학습셋에 절대 포함 금지
2. **시간순 분리(Temporal Split) 유지** - 미래 데이터로 과거 예측하는 데이터 누수(leakage) 방지
3. **모델 재학습 시에도 동일 원칙 적용** - 새 시즌 시작 시 이전 시즌들만 학습에 사용
4. **검증 정확도의 신뢰성** - 이 원칙을 지켜야 74.61% 정확도가 실제 예측 성능을 반영함

### 새 시즌 전환 시 (예: 26-27 시즌 시작 시)
```python
# 26-27 시즌(2027)이 시작되면:
train_seasons = [2023, 2024, 2025, 2026]  # 25-26 시즌을 학습에 추가
val_season = 2027                          # 26-27 시즌을 검증용으로 사용
```

## 프로젝트 구조

### 핵심 파일
- `app/main.py` - Streamlit 메인 앱
- `app/services/data_loader.py` - 데이터 로딩 및 V5.2 피처 생성
- `app/services/predictor_v5.py` - V5.2 예측 모델
- `src/utils/helpers.py` - 유틸리티 함수 (시즌 계산 등)

### 모델 파일
- `bucketsvision_v4/models/v5_2_model.pkl` - XGBoost 모델
- `bucketsvision_v4/models/v5_2_scaler.pkl` - StandardScaler
- `bucketsvision_v4/models/v5_2_feature_names.json` - 피처명 (11개)
- `bucketsvision_v4/models/v5_2_metadata.json` - 모델 메타데이터

### 학습/재학습 스크립트
- `bucketsvision_v4/train_v5_2_with_rest.py` - V5.2 학습 스크립트
- `bucketsvision_v4/save_v5_2_model.py` - 모델 저장 스크립트

### 문서
- `bucketsvision_v4/docs/V5_2_MODEL_DOCUMENTATION.md` - 모델 상세 문서

### 데이터 소스
- **DNT API**: Team EPM, Player EPM
- **NBA Stats API**: 경기 결과, 팀 스탯, 스케줄, B2B/휴식일
- **ESPN API**: 부상/결장 정보
- **Odds API**: 배당 정보

## V5.2 모델 아키텍처

### 알고리즘
- **XGBoost** with Hyperparameter Tuning
- 저신뢰도(<70%) 정확도: **74.61%** (+0.76%p vs V5.0)

### 피처 (11개)
| 카테고리 | 피처 | 설명 |
|----------|------|------|
| EPM (4개) | team_epm_diff | 팀 EPM 차이 |
| | player_rotation_epm_diff | 로테이션 EPM 차이 |
| | bench_strength_diff | 벤치 EPM 차이 |
| | top5_epm_diff | 상위 5인 EPM 차이 |
| Four Factors (3개) | efg_pct_diff | eFG% 차이 |
| | ft_rate_diff | FT Rate 차이 |
| | orb_pct_diff | 공격 리바운드율 차이 |
| 모멘텀 (2개) | last3_win_pct_diff | 최근 3경기 승률 차이 |
| | last5_win_pct_diff | 최근 5경기 승률 차이 |
| 피로도 (2개) | b2b_diff | B2B 상태 차이 (-1, 0, +1) |
| | rest_days_diff | 휴식일 차이 |

### 후행 지표 (Post-prediction Adjustment)
| 지표 | 적용 방식 |
|------|----------|
| 부상 영향 | EPM 기반 prob_shift 계산, 최대 ±10%p 조정 |

## 서비스 실행

```bash
# Streamlit 앱 실행
streamlit run app/main.py
# 또는
python3 -m streamlit run app/main.py

# E2E 테스트
python test_v5_2_e2e.py
```

## 주의사항

### 모델 관련
1. V5.2는 B2B와 휴식일을 **모델 피처**로 학습 (후행 지표 아님)
2. 부상 영향만 후행 지표로 예측 후 적용
3. Scaler와 모델은 반드시 함께 사용
4. 피처 순서는 `v5_2_feature_names.json` 참조

### 시간대 관련
- 경기 날짜: 미국 동부 시간 (ET) 기준
- 한국 시간 표시: ET + 1일 (KST)
- 예: ET 11/28 경기 → KST 11/29 표시

### API 캐시
- 오늘/내일 경기: 캐시 미사용 (실시간 반영)
- 과거 경기: 24시간 캐시

## 개발 시 체크리스트
- [ ] 시즌 연도가 올바른지 확인 (2025-26 시즌 = 2026)
- [ ] 날짜 테스트 시 2025년 기준 사용
- [ ] 모델 피처 순서가 v5_2_feature_names.json과 일치하는지 확인
- [ ] B2B/휴식일 피처가 올바르게 계산되는지 확인
- [ ] 부상 조정이 ±10%p 한도 내에서 적용되는지 확인
