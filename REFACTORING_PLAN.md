# BucketsVision 리팩토링 계획서

## 개요

- **총 코드량**: 156개 Python 파일, 46,530줄
- **데드 코드**: ~12,000줄 (26%)
- **리팩토링 대상**: ~15,000줄
- **예상 결과**: 코드베이스 35% 축소, 구조 개선

---

## Phase 1: 데드 코드 제거 (안전)

> 핵심 기능 영향: 없음 | 위험도: 낮음

### 1.1 Scripts Archive 삭제
**경로**: `/scripts/archive/`
**용량**: 8개 파일, 3,668줄

| 파일 | 줄 수 | 상태 |
|------|-------|------|
| analyze_fatigue_factors.py | 715 | 미사용 |
| train_with_fatigue.py | 586 | 미사용 |
| train_with_injury.py | 525 | 미사용 |
| validate_fatigue_impact.py | 467 | 미사용 |
| feature_engineering_experiment.py | 406 | 미사용 |
| train_with_four_factors.py | 401 | 미사용 |
| test_injury_adjustment.py | 165 | 미사용 |
| verify_altitude_effect.py | 255 | 미사용 |

**작업**:
```bash
rm -rf scripts/archive/
```

### 1.2 V2 레거시 삭제
**경로**: `/bucketsvision_v2/`
**용량**: 37개 파일, ~5,000줄

**포함 내용**:
- 중복된 feature pipeline
- 중복된 config
- 중복된 model 구현
- 사용하지 않는 calibration 모듈

**작업**:
```bash
rm -rf bucketsvision_v2/
```

### 1.3 TheHalf Archive 삭제
**경로**: `/archive/thehalf/`
**용량**: 23개 파일, ~3,000줄

**작업**:
```bash
rm -rf archive/thehalf/
```

### 1.4 빈 테스트 디렉토리 정리
**경로**: `/tests/`
**상태**: `__init__.py`만 존재, 실제 테스트 없음

**작업**:
```bash
rm -rf tests/
```

### Phase 1 결과
- **삭제**: ~11,668줄
- **파일 수**: 156 → 83개 (-73개)
- **위험도**: 없음 (프로덕션 코드에서 import 없음)

---

## Phase 2: 대형 파일 분리

> 핵심 기능 영향: 없음 | 위험도: 중간 (테스트 필요)

### 2.1 data_loader.py 분리 (848줄 → 4개 모듈)

**현재**: `/app/services/data_loader.py`
**문제**: 데이터 로딩, 피처 생성, 캐싱, API 호출이 혼재

**분리 계획**:

```
app/services/
├── data_loader.py          (메인 오케스트레이터, ~150줄)
├── data_fetchers.py        (API 데이터 가져오기, ~200줄)
├── feature_builder.py      (V4.2/V4.3 피처 생성, ~300줄)
└── cache_manager.py        (캐시 관리, ~200줄)
```

**새 구조**:

```python
# data_fetchers.py
class TeamDataFetcher:
    """팀 EPM, 게임 로그 데이터 가져오기"""
    def load_team_epm(self, target_date: date) -> Dict
    def load_team_game_logs(self, season: int) -> pd.DataFrame

class PlayerDataFetcher:
    """선수 EPM 데이터 가져오기"""
    def load_player_epm(self, season: int) -> pd.DataFrame

class GameDataFetcher:
    """경기 스케줄 및 결과 가져오기"""
    def get_games(self, game_date: date) -> List[Dict]
```

```python
# feature_builder.py
class V4FeatureBuilder:
    """V4.2/V4.3 피처 생성"""
    def build_v4_2_features(self, home_id, away_id, team_epm, target_date) -> Dict
    def build_v4_3_features(self, home_id, away_id, team_epm, target_date) -> Dict

    # Private methods
    def _calc_rotation_epm(self, team_id, season) -> float
    def _calc_bench_strength(self, team_id, season) -> float
    def _calc_last5_stats(self, team_id, logs) -> Dict
```

```python
# cache_manager.py
class DataCacheManager:
    """통합 캐시 관리"""
    def __init__(self, data_dir: Path)
    def get_team_epm(self, date: date) -> Optional[Dict]
    def set_team_epm(self, date: date, data: Dict) -> None
    def get_player_epm(self, season: int) -> Optional[pd.DataFrame]
    def clear_cache(self) -> None
```

```python
# data_loader.py (리팩토링 후)
class DataLoader:
    """데이터 로딩 오케스트레이터"""
    def __init__(self, data_dir: Path):
        self.fetchers = {
            'team': TeamDataFetcher(data_dir),
            'player': PlayerDataFetcher(data_dir),
            'game': GameDataFetcher(data_dir),
        }
        self.feature_builder = V4FeatureBuilder(self.fetchers)
        self.cache = DataCacheManager(data_dir)
```

### 2.2 game_card.py 분리 (516줄 → 3개 모듈)

**현재**: `/app/components/game_card.py`

**분리 계획**:
```
app/components/
├── game_card.py           (메인 카드, ~200줄)
├── game_metrics.py        (지표 표시, ~150줄)
└── game_status.py         (상태 배지, ~100줄)
```

### 2.3 ridge_model.py 분리 (503줄 → 3개 모듈)

**현재**: `/src/models/ridge_model.py`
**문제**: 3개의 독립적인 모델이 한 파일에 존재

**분리 계획**:
```
src/models/
├── base.py                 (기존 유지)
├── ridge.py                (RidgeModel, ~170줄)
├── elasticnet.py           (ElasticNetModel, ~170줄)
└── huber.py                (HuberRegressionModel, ~170줄)
```

---

## Phase 3: 중복 코드 통합

> 핵심 기능 영향: 없음 | 위험도: 중간

### 3.1 Injury 로직 통합 (4개 → 1개)

**현재 상태**:
| 파일 | 줄 수 | 용도 |
|------|-------|------|
| src/features/injury_impact.py | 235 | 기본 부상 영향 |
| src/features/data_driven_injury.py | 478 | 데이터 기반 부상 |
| src/features/injury_ensemble.py | 468 | 앙상블 부상 |
| src/prediction/injury_adjustment.py | ? | 예측 조정 |

**통합 계획**:
```
src/features/
├── injury/
│   ├── __init__.py
│   ├── base.py              (기본 InjuryImpactFeature)
│   ├── data_driven.py       (DataDrivenInjury)
│   └── ensemble.py          (InjuryEnsemble)
```

### 3.2 API 클라이언트 정리

**현재**:
- `/src/data_collection/dnt_client.py` (548줄, async)
- `/app/services/dnt_api.py` (120줄, sync wrapper)

**통합**:
```python
# app/services/dnt_api.py를 제거하고
# src/data_collection/dnt_client.py에 sync 메서드 추가
class DNTClient:
    async def get_team_epm_async(self, date) -> Dict
    def get_team_epm(self, date) -> Dict  # sync wrapper 내장
```

---

## Phase 4: 캐싱 최적화

> 핵심 기능 영향: 성능 개선 | 위험도: 낮음

### 4.1 현재 캐싱 문제점

1. **비일관성**: Streamlit cache + dict + DiskCache 혼용
2. **메모리 누수**: 캐시 크기 제한 없음
3. **만료 정책 없음**: 오래된 데이터 유지
4. **API 응답 미캐싱**: NBA Stats API 호출 반복

### 4.2 통합 캐싱 전략

```python
# src/utils/cache.py 개선

class UnifiedCache:
    """통합 캐시 관리자"""

    def __init__(self, cache_dir: Path, max_memory_mb: int = 100):
        self.disk_cache = DiskCache(cache_dir)
        self.memory_cache = LRUCache(maxsize=1000)
        self.max_memory = max_memory_mb * 1024 * 1024

    def get(self, key: str, ttl: int = 3600) -> Optional[Any]:
        """메모리 → 디스크 순서로 조회"""
        # 1. 메모리 캐시 확인
        if key in self.memory_cache:
            return self.memory_cache[key]

        # 2. 디스크 캐시 확인
        data = self.disk_cache.get(key, ttl=ttl)
        if data:
            self.memory_cache[key] = data
        return data

    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """메모리 + 디스크에 저장"""
        self.memory_cache[key] = value
        if persist:
            self.disk_cache.set(key, value)

    def cleanup(self, max_age_days: int = 7) -> int:
        """오래된 캐시 정리"""
        return self.disk_cache.cleanup(max_age_days)
```

### 4.3 API 응답 캐싱

```python
# NBA Stats API 캐싱 추가
class NBAStatsClient:
    def __init__(self, cache: UnifiedCache):
        self.cache = cache

    def get_scoreboard(self, date: date) -> Dict:
        cache_key = f"nba_scoreboard_{date}"

        # 오늘/내일: 5분 캐시, 과거: 24시간 캐시
        ttl = 300 if date >= date.today() else 86400

        cached = self.cache.get(cache_key, ttl=ttl)
        if cached:
            return cached

        data = self._fetch_scoreboard(date)
        self.cache.set(cache_key, data)
        return data
```

---

## Phase 5: 메모리 최적화

> 핵심 기능 영향: 성능 개선 | 위험도: 낮음

### 5.1 DataFrame 메모리 최적화

```python
# src/utils/memory.py (신규)

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 메모리 사용량 최적화"""
    for col in df.columns:
        col_type = df[col].dtype

        # int64 → int32/int16/int8
        if col_type == 'int64':
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')

        # float64 → float32
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')

        # object → category (반복값이 많은 경우)
        elif col_type == 'object':
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')

    return df
```

### 5.2 Lazy Loading 적용

```python
# 현재: 앱 시작 시 모든 데이터 로드
# 개선: 필요할 때만 로드

class LazyDataLoader:
    def __init__(self):
        self._team_epm = None
        self._player_epm = None
        self._game_logs = None

    @property
    def team_epm(self):
        if self._team_epm is None:
            self._team_epm = self._load_team_epm()
        return self._team_epm

    def clear(self):
        """메모리 해제"""
        self._team_epm = None
        self._player_epm = None
        self._game_logs = None
        gc.collect()
```

---

## Phase 6: 코드 품질 개선

> 핵심 기능 영향: 없음 | 위험도: 낮음

### 6.1 타입 힌트 추가

```python
# 현재
def build_features(home_id, away_id, team_epm, date):
    ...

# 개선
def build_features(
    home_id: int,
    away_id: int,
    team_epm: Dict[int, TeamEPM],
    date: date
) -> Dict[str, float]:
    ...
```

### 6.2 Docstring 표준화

```python
def build_v4_3_features(
    self,
    home_team_id: int,
    away_team_id: int,
    team_epm: Dict[int, Dict],
    target_date: date
) -> Dict[str, float]:
    """
    V4.3 모델용 피처 생성.

    Args:
        home_team_id: 홈팀 ID
        away_team_id: 원정팀 ID
        team_epm: 팀별 EPM 데이터
        target_date: 경기 날짜

    Returns:
        13개 피처가 포함된 딕셔너리:
        - team_epm_diff: 팀 EPM 차이
        - team_oepm_diff: 공격 EPM 차이
        - ...

    Raises:
        ValueError: 팀 ID가 유효하지 않은 경우
    """
```

### 6.3 불필요한 import 정리

```python
# 사용하지 않는 import 제거
# isort로 import 정렬

# Before
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime, timedelta
import json
import os

# After
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
```

---

## 실행 순서 및 체크리스트

### Step 1: Phase 1 실행 (데드 코드 제거)
- [ ] `scripts/archive/` 삭제
- [ ] `bucketsvision_v2/` 삭제
- [ ] `archive/thehalf/` 삭제
- [ ] `tests/` 삭제
- [ ] 앱 실행 테스트

### Step 2: Phase 2 실행 (파일 분리)
- [ ] `data_loader.py` 분리
- [ ] import 경로 수정
- [ ] 앱 실행 테스트
- [ ] `game_card.py` 분리 (선택)
- [ ] `ridge_model.py` 분리 (선택)

### Step 3: Phase 3 실행 (중복 통합)
- [ ] Injury 로직 통합
- [ ] API 클라이언트 정리
- [ ] 앱 실행 테스트

### Step 4: Phase 4 실행 (캐싱 최적화)
- [ ] UnifiedCache 구현
- [ ] API 캐싱 적용
- [ ] 성능 테스트

### Step 5: Phase 5 실행 (메모리 최적화)
- [ ] DataFrame 최적화 적용
- [ ] Lazy Loading 적용
- [ ] 메모리 사용량 측정

### Step 6: Phase 6 실행 (코드 품질)
- [ ] 타입 힌트 추가
- [ ] Docstring 표준화
- [ ] import 정리

---

## 예상 결과

### 코드 축소
| 항목 | Before | After | 감소 |
|------|--------|-------|------|
| 파일 수 | 156개 | 83개 | -47% |
| 총 줄 수 | 46,530줄 | ~30,000줄 | -35% |
| 데드 코드 | 12,000줄 | 0줄 | -100% |

### 성능 개선
| 항목 | Before | After |
|------|--------|-------|
| 앱 시작 시간 | ~3초 | ~2초 |
| 메모리 사용량 | ~500MB | ~300MB |
| API 호출 수 | 매번 호출 | 캐시 활용 |

### 유지보수성
| 항목 | Before | After |
|------|--------|-------|
| 파일당 평균 줄 수 | 298줄 | ~180줄 |
| 최대 파일 크기 | 848줄 | ~300줄 |
| 코드 중복 | 높음 | 낮음 |

---

## 주의사항

1. **백업 필수**: 각 Phase 실행 전 git commit
2. **점진적 실행**: 한 번에 하나의 Phase만 실행
3. **테스트 필수**: 각 Phase 후 앱 실행 확인
4. **롤백 준비**: 문제 발생 시 즉시 복구 가능하도록

---

## 명령어 요약

```bash
# Phase 1: 데드 코드 제거
rm -rf scripts/archive/
rm -rf bucketsvision_v2/
rm -rf archive/thehalf/
rm -rf tests/

# 검증
python3 -m streamlit run app/main.py
```

---

*작성일: 2025-11-30*
*버전: 1.0*
