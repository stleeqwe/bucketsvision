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

## 실시간 API 데이터 사용 원칙 (필수 준수)

### 엄격히 금지되는 사항
1. **로컬 파일에 저장된 과거 데이터 사용 금지** - 모든 EPM/스탯 데이터는 API 실시간 호출 필수
2. **parquet/csv 등 캐시 파일로 데이터 로드 금지** - 오래된 데이터로 예측 정확도 저하
3. **수동 수집 스크립트로 저장한 데이터 사용 금지** - 트레이드/부상/EPM 변동 미반영

### 모든 데이터는 실시간 API 호출
| 데이터 | API | 메서드 |
|--------|-----|--------|
| Team EPM | DNT API | `dnt_client.get_team_epm()` |
| Player EPM | DNT API | `dnt_client.get_player_epm()` |
| 경기 스케줄 | NBA Stats API | `nba_client.get_scoreboard()` |
| 부상 정보 | ESPN API | `espn_client.get_injuries()` |
| 배당 정보 | Odds API | `odds_client.get_all_games_odds()` |

### 허용되는 캐시
- **인스턴스 레벨 메모리 캐시**: 같은 요청 내 중복 API 호출 방지용 (OK)
- **Streamlit @st.cache_resource**: 세션 내 모델 재로드 방지용 (OK)

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
4. **검증 정확도의 신뢰성** - 이 원칙을 지켜야 78.05% 정확도가 실제 예측 성능을 반영함

### 새 시즌 전환 시 (예: 26-27 시즌 시작 시)
```python
# 26-27 시즌(2027)이 시작되면:
train_seasons = [2023, 2024, 2025, 2026]  # 25-26 시즌을 학습에 추가
val_season = 2027                          # 26-27 시즌을 검증용으로 사용
```

## 프로젝트 구조

### 핵심 파일
- `app/main.py` - Streamlit 메인 앱
- `app/services/data_loader.py` - 데이터 로딩 및 V5.4 피처 생성
- `app/services/predictor_v5.py` - V5.4 예측 모델
- `src/utils/helpers.py` - 유틸리티 함수 (시즌 계산 등)

### 모델 파일
- `bucketsvision_v4/models/v5_4_model.pkl` - Logistic Regression 모델
- `bucketsvision_v4/models/v5_4_scaler.pkl` - StandardScaler
- `bucketsvision_v4/models/v5_4_feature_names.json` - 피처명 (5개)
- `bucketsvision_v4/models/v5_4_metadata.json` - 모델 메타데이터

### 학습/재학습 스크립트
- `bucketsvision_v4/train_v5_2_with_rest.py` - 학습 스크립트 (피처 빌드용)
- `bucketsvision_v4/optimal_feature_search.py` - V5.4 피처 최적화 스크립트

### 문서
- `bucketsvision_v4/docs/MODEL_COMPARISON_REPORT.md` - 모델 비교 문서

### 데이터 소스
- **DNT API**: Team EPM, Player EPM, SOS (Strength of Schedule)
- **NBA Stats API**: 경기 결과, 팀 스탯, 스케줄
- **ESPN API**: 부상/결장 정보
- **Odds API**: 배당 정보

## V5.4 모델 아키텍처

### 알고리즘
- **Logistic Regression** (C=0.01, L2 정규화)
- 전체 정확도: **78.05%**
- 저신뢰도(<70%) 정확도: **71.43%**
- 고신뢰도(≥70%) 정확도: **87.88%**

### 피처 (5개)
| 피처 | 설명 | 계수 |
|------|------|------|
| team_epm_diff | 팀 EPM 차이 (홈-원정) | +0.407 |
| top5_epm_diff | 상위 5인 EPM 차이 | +0.361 |
| bench_strength_diff | 벤치 선수 EPM 차이 | +0.259 |
| ft_rate_diff | Free Throw Rate 차이 | +0.066 |
| sos_diff | Strength of Schedule 차이 | +0.016 |

### 확률 범위
- **최소**: 8.2%
- **최대**: 94.8%
- **압축 없음** (XGBoost 대비 넓은 확률 범위)

### 후행 지표: Injury Impact v1.0.0
| 항목 | 값 |
|------|-----|
| 버전 | 1.0.0 (2025-12-03) |
| 알고리즘 | Performance-based (출전 vs 미출전 성과 비교) |
| 공식 | `prob_shift = player_epm × 0.02 × normalized_diff` |
| 폴백 | 결장 0경기 시 `prob_shift = player_epm × 0.02` |
| 한도 | 없음 (확률 경계 1%~99%만 유지) |

**적용 조건:**
- EPM > 0, MPG ≥ 12, 출전율 > 1/3

**메타데이터:** `bucketsvision_v4/models/injury_impact_v1_metadata.json`

## 서비스 실행

```bash
# Streamlit 앱 실행
streamlit run app/main.py
# 또는
python3 -m streamlit run app/main.py
```

## 주의사항

### 모델 관련
1. V5.4는 5개 피처만 사용 (단순하지만 강력)
2. 부상 영향은 후행 지표로 예측 후 적용
3. Scaler와 모델은 반드시 함께 사용
4. 피처 순서는 `v5_4_feature_names.json` 참조

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
- [ ] 모델 피처 순서가 v5_4_feature_names.json과 일치하는지 확인

## V5.4 vs 이전 모델 비교

| 모델 | 피처수 | 전체 정확도 | 저신뢰(<70%) | 고신뢰(≥70%) |
|------|--------|-------------|--------------|--------------|
| V4.4 | 13개 | 76.02% | 66.18% | 88.18% |
| V5.3 | 9개 | 76.02% | 66.18% | 88.18% |
| **V5.4** | **5개** | **78.05%** | **71.43%** | 87.88% |
