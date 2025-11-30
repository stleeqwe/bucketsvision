"""
메모리 최적화 유틸리티.

DataFrame 메모리 사용량 최적화 및 모니터링 기능.
"""

import gc
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import logger


def optimize_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    DataFrame 메모리 사용량 최적화.

    - int64 → int32/int16/int8
    - float64 → float32
    - object → category (반복값이 많은 경우)

    Args:
        df: 최적화할 DataFrame
        verbose: 상세 로깅 여부

    Returns:
        최적화된 DataFrame
    """
    if df.empty:
        return df

    start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)

    for col in df.columns:
        col_type = df[col].dtype

        # 정수형 최적화
        if col_type in ['int64', 'int32']:
            c_min = df[col].min()
            c_max = df[col].max()

            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype('uint8')
                elif c_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif c_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype('int8')
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype('int16')
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype('int32')

        # 실수형 최적화
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')

        # 문자열 → 카테고리 (반복값이 50% 이하일 때)
        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    reduction = (start_mem - end_mem) / start_mem * 100

    if verbose:
        logger.info(f"DataFrame optimized: {start_mem:.2f}MB → {end_mem:.2f}MB ({reduction:.1f}% reduction)")

    return df


def get_memory_usage() -> dict:
    """
    현재 메모리 사용량 조회.

    Returns:
        메모리 사용량 딕셔너리
    """
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
    }


def force_gc() -> int:
    """
    강제 가비지 컬렉션.

    Returns:
        수집된 객체 수
    """
    gc.collect()
    return gc.collect()


class MemoryTracker:
    """메모리 사용량 추적기"""

    def __init__(self):
        self._baseline: Optional[dict] = None

    def start(self) -> dict:
        """추적 시작"""
        force_gc()
        self._baseline = get_memory_usage()
        return self._baseline

    def check(self) -> dict:
        """현재 사용량 및 변화량 반환"""
        current = get_memory_usage()

        if self._baseline is None:
            return current

        return {
            'current_rss_mb': current['rss_mb'],
            'current_vms_mb': current['vms_mb'],
            'delta_rss_mb': current['rss_mb'] - self._baseline['rss_mb'],
            'delta_vms_mb': current['vms_mb'] - self._baseline['vms_mb'],
        }

    def reset(self) -> int:
        """추적 초기화 및 GC 실행"""
        self._baseline = None
        return force_gc()
