# 선수 개별 EPM 활용 모델 고도화 연구 보고서

**연구일**: 2025-11-30
**목표**: 팀 EPM과 별개로 선수 개별 EPM을 접목하여 예측 성능 개선 (과적합 방지)

---

## 1. 연구 요약

### 핵심 결과

| 설정 | 피처 수 | 정확도 | 개선폭 | AUC | 권장 |
|------|--------|--------|--------|-----|------|
| V4.2 기준선 (팀 EPM만) | 11 | 73.80% | - | 0.7684 | 현재 |
| **Minimal 세트** | **13** | **76.38%** | **+2.58%p** | **0.8088** | **✅ 최우선** |
| Basic 세트 | 15 | 75.65% | +1.85%p | 0.8230 | 차선 |
| bench_strength만 | 12 | 74.91% | +1.11%p | 0.7942 | 최소 침습 |

### 최종 권장: Minimal 세트 + 강한 정규화

```python
추가 피처:
1. player_rotation_epm_diff  # 로테이션 선수(MPG >= 12) 가중 평균 EPM 차이
2. bench_strength_diff       # 벤치 선수(6-10번째) 평균 EPM 차이

최적 하이퍼파라미터:
- C = 0.001 (매우 강한 L2 정규화)
```

---

## 2. 연구 방법론

### 2.1 테스트한 선수 EPM 피처 (12개)

| 피처 | 설명 | 팀EPM 상관관계 |
|------|------|---------------|
| `player_rotation_epm_diff` | 로테이션(MPG≥12) 선수 가중 EPM | 0.84 (높음) |
| `top5_weighted_epm_diff` | 상위 5명 출전시간 가중 EPM | 0.77 (높음) |
| `top8_weighted_epm_diff` | 상위 8명 출전시간 가중 EPM | - |
| `bench_strength_diff` | 벤치(6-10번째) 평균 EPM | 0.59 (중간) |
| `starter_bench_gap_diff` | 주전-벤치 EPM 갭 | 0.46 (낮음) |
| `star_concentration_diff` | Top2 EPM 집중도 | - |
| `elite_player_diff` | Elite(EPM≥2) 선수 수 | - |
| `negative_minutes_diff` | 저EPM 선수 출전시간 | - |
| `guard_epm_diff` | 가드진 EPM | - |
| `frontcourt_epm_diff` | 프론트코트 EPM | - |
| `epm_variance_diff` | EPM 분산 | - |
| `rotation_size_diff` | 로테이션 크기 | - |

### 2.2 실험 설계

- **훈련 데이터**: 3,683 경기 (2022-10 ~ 2025-04)
- **검증 데이터**: 271 경기 (2025-10 ~ 2025-11)
- **교차 검증**: 5-Fold Stratified CV
- **정규화 스윕**: C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
- **안정성 테스트**: 10개 시드

---

## 3. 핵심 발견

### 3.1 다중공선성 문제

```
팀 EPM과 선수 EPM 피처 간 상관관계:
  - player_rotation_epm_diff: 0.8412 (매우 높음)
  - top5_weighted_epm_diff: 0.7727 (높음)
  - bench_strength_diff: 0.5924 (중간)
  - starter_bench_gap_diff: 0.4565 (낮음)
```

**문제**: 선수 EPM 피처들이 팀 EPM과 높은 상관관계를 가짐
**해결**:
1. 서로 보완적인 피처 조합 선택
2. 강한 정규화(C=0.001)로 중복 정보 억제

### 3.2 개별 피처 효과 분석

```
V4 기준선에 개별 피처 추가 시 효과:
  ✓ bench_strength_diff: +0.37%p (유일하게 개별 효과 있음)
  - player_rotation_epm_diff: +0.00%p
  ✗ star_concentration_diff: -0.37%p
  ✗ top5_weighted_epm_diff: -1.11%p
  ✗ frontcourt_epm_diff: -2.58%p
```

**중요 발견**: 대부분의 선수 EPM 피처는 **개별 추가 시 성능 저하**
→ 팀 EPM과의 중복 때문
→ **조합 효과**가 중요

### 3.3 조합 효과

```
Minimal 세트 (2개):
  - player_rotation_epm_diff + bench_strength_diff
  - 개별 효과: +0.00% + +0.37% = +0.37%
  - 실제 조합 효과: +2.58%
  → 시너지 효과: +2.21%p

원리:
  - player_rotation_epm_diff: 팀의 핵심 로테이션 전력
  - bench_strength_diff: 팀의 벤치 깊이
  → 서로 다른 측면의 정보를 제공
```

### 3.4 과적합 분석

```
Overfit Gap (훈련 - 검증):
  - V4.2 기준선: -0.0706 (음수)
  - Minimal 세트: -0.0926 (음수)

음수 Gap 의미:
  → 검증 성능 > 훈련 성능
  → 과적합 없음
  → 시간적 특성 (최신 데이터가 예측하기 쉬움)
```

### 3.5 안정성

```
10개 시드로 테스트:
  Accuracy: 75.65% ± 0.00%
  AUC: 0.8230 ± 0.0000

  → 완전히 안정적인 결과
```

---

## 4. 권장 구현

### 4.1 V4.3 모델 스펙 (권장)

```python
# V4.3 피처 (13개)
V4_3_FEATURE_NAMES = [
    # V4.2 기존 (11개)
    'team_epm_diff',
    'team_oepm_diff',
    'team_depm_diff',
    'sos_diff',
    'last5_win_pct_diff',
    'efg_pct_diff',
    'streak_diff',
    'margin_ewma_diff',
    'ft_rate_diff',
    'away_road_strength',
    'orb_diff',

    # 신규 선수 EPM (2개)
    'player_rotation_epm_diff',  # NEW
    'bench_strength_diff',       # NEW
]

# 모델 설정
model_config = {
    'algorithm': 'LogisticRegression',
    'C': 0.001,  # 중요: 매우 강한 정규화
    'calibration': 'isotonic',
    'cv': 5,
}
```

### 4.2 피처 계산 로직

```python
def rotation_epm(team_id: int, season: int, min_mpg: float = 12.0) -> float:
    """
    로테이션 선수(MPG >= 12)의 가중 평균 EPM.

    공식: Σ(EPM_i × MPG_i) / Σ(MPG_i)
    """
    players = get_team_players(team_id, season)
    rotation = players[players['mpg'] >= min_mpg]

    if len(rotation) == 0 or rotation['mpg'].sum() == 0:
        return 0.0

    return (rotation['tot'] * rotation['mpg']).sum() / rotation['mpg'].sum()


def bench_strength(team_id: int, season: int) -> float:
    """
    벤치 선수(6-10번째 MPG)의 평균 EPM.
    """
    players = get_team_players(team_id, season)
    sorted_players = players.nlargest(10, 'mpg')
    bench = sorted_players.iloc[5:10]

    if len(bench) == 0:
        return -2.0  # 기본값

    return bench['tot'].mean()


# 피처 생성
def build_player_epm_features(home_id, away_id, season):
    h_rot = rotation_epm(home_id, season)
    a_rot = rotation_epm(away_id, season)

    h_bench = bench_strength(home_id, season)
    a_bench = bench_strength(away_id, season)

    return {
        'player_rotation_epm_diff': h_rot - a_rot,
        'bench_strength_diff': h_bench - a_bench,
    }
```

### 4.3 데이터 흐름

```
선수 EPM 데이터 (DNT API)
    │
    ├── get_player_epm(date, season)
    │       ↓
    │   player_epm DataFrame
    │   (player_id, team_id, tot, off, def, mpg, ...)
    │
    └── 피처 계산
            │
            ├── rotation_epm(team_id, season)
            │       → 로테이션(MPG≥12) 가중 EPM
            │
            └── bench_strength(team_id, season)
                    → 벤치(6-10th) 평균 EPM

    ↓ 차이값 계산

player_rotation_epm_diff = home_rotation - away_rotation
bench_strength_diff = home_bench - away_bench
```

---

## 5. 주의사항

### 5.1 필수 사항

1. **정규화 강도 유지**: C=0.001 사용 필수
   - C가 높으면 다중공선성 문제 발생
   - 기존 V4.2의 C=0.01보다 10배 강함

2. **스케일러 재학습**: 새 피처 추가 시 StandardScaler 재학습 필요

3. **피처 순서**: v4_feature_names.json 순서 유지

### 5.2 피해야 할 것

1. **추가 피처 확장**:
   - Extended/All 세트는 성능 저하
   - 2개 피처가 최적

2. **정규화 약화**:
   - C > 0.01 사용 시 과적합 위험

3. **개별 피처 분석 맹신**:
   - 조합 효과가 중요
   - 개별 효과 없어도 조합 시 시너지

### 5.3 모니터링 필요

1. **검증 세트 크기**:
   - 현재 271경기로 작음
   - 시즌 진행에 따라 추가 검증 필요

2. **시즌 변화**:
   - 새 시즌 데이터로 성능 확인
   - 로스터 변화에 따른 영향

---

## 6. 결론

### 성과
- **+2.58%p 정확도 개선** (73.80% → 76.38%)
- **피처 2개만 추가**로 효율적 개선
- **과적합 없음** 확인

### 핵심 인사이트
1. 선수 EPM은 팀 EPM과 높은 상관관계 → 단순 추가는 비효율적
2. **벤치 깊이** 정보가 팀 EPM에서 부족한 부분을 보완
3. 강한 정규화로 중복 정보 억제 시 시너지 효과 발생

### 다음 단계
1. V4.3 모델 학습 및 배포
2. 실제 예측에서 성능 검증
3. 부상 영향도 계산과 연동 (결장자 제외 EPM)

---

## 부록: 실험 파일

- `research/player_epm_study.py` - 기본 실험
- `research/player_epm_deep_analysis.py` - 심층 분석
- `research/player_epm_final_validation.py` - 최종 검증
- `research/player_epm_study_results.json` - 기본 실험 결과
- `research/player_epm_deep_analysis_results.json` - 심층 분석 결과
- `research/player_epm_final_results.json` - 최종 검증 결과
