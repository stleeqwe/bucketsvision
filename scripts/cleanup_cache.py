#!/usr/bin/env python3
"""
캐시 정리 스크립트.

오래된 캐시 파일을 정리하여 디스크 공간을 확보합니다.

사용법:
    # 7일 이상 된 캐시 정리
    python scripts/cleanup_cache.py

    # 3일 이상 된 캐시 정리
    python scripts/cleanup_cache.py --days 3

    # 전체 캐시 삭제
    python scripts/cleanup_cache.py --all

    # 캐시 통계만 확인
    python scripts/cleanup_cache.py --stats
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_cache_stats(cache_dir: Path) -> dict:
    """캐시 통계 수집"""
    stats = {
        'total_files': 0,
        'total_size_mb': 0,
        'expired_files': 0,
        'by_age': {
            '1d': 0,
            '7d': 0,
            '30d': 0,
            'older': 0,
        }
    }

    if not cache_dir.exists():
        return stats

    now = datetime.now().timestamp()
    one_day = 86400
    seven_days = 7 * one_day
    thirty_days = 30 * one_day

    for cache_file in cache_dir.rglob("*.json"):
        stats['total_files'] += 1
        stats['total_size_mb'] += cache_file.stat().st_size / (1024 * 1024)

        try:
            with open(cache_file, 'r') as f:
                entry = json.load(f)
            created_at = entry.get('created_at', 0)
            ttl = entry.get('ttl')
            age = now - created_at

            # 만료 체크
            if ttl and age > ttl:
                stats['expired_files'] += 1

            # 나이별 분류
            if age < one_day:
                stats['by_age']['1d'] += 1
            elif age < seven_days:
                stats['by_age']['7d'] += 1
            elif age < thirty_days:
                stats['by_age']['30d'] += 1
            else:
                stats['by_age']['older'] += 1

        except (json.JSONDecodeError, KeyError):
            stats['expired_files'] += 1

    return stats


def cleanup_old_cache(cache_dir: Path, max_age_days: int = 7) -> int:
    """오래된 캐시 파일 정리"""
    if not cache_dir.exists():
        return 0

    deleted = 0
    cutoff = datetime.now().timestamp() - (max_age_days * 86400)

    for cache_file in cache_dir.rglob("*.json"):
        try:
            with open(cache_file, 'r') as f:
                entry = json.load(f)
            created_at = entry.get('created_at', 0)

            if created_at < cutoff:
                cache_file.unlink()
                deleted += 1

        except (json.JSONDecodeError, KeyError):
            cache_file.unlink()
            deleted += 1

    return deleted


def cleanup_expired(cache_dir: Path) -> int:
    """TTL 만료된 캐시 정리"""
    if not cache_dir.exists():
        return 0

    deleted = 0
    now = datetime.now().timestamp()

    for cache_file in cache_dir.rglob("*.json"):
        try:
            with open(cache_file, 'r') as f:
                entry = json.load(f)
            created_at = entry.get('created_at', 0)
            ttl = entry.get('ttl')

            if ttl and (now - created_at) > ttl:
                cache_file.unlink()
                deleted += 1

        except (json.JSONDecodeError, KeyError):
            cache_file.unlink()
            deleted += 1

    return deleted


def clear_all_cache(cache_dir: Path) -> int:
    """전체 캐시 삭제"""
    if not cache_dir.exists():
        return 0

    deleted = 0
    for cache_file in cache_dir.rglob("*.json"):
        cache_file.unlink()
        deleted += 1

    return deleted


def print_stats(stats: dict):
    """통계 출력"""
    print(f"\n{'='*50}")
    print("캐시 통계")
    print(f"{'='*50}")
    print(f"  총 파일 수:     {stats['total_files']:,}개")
    print(f"  총 용량:        {stats['total_size_mb']:.2f} MB")
    print(f"  만료된 파일:    {stats['expired_files']:,}개")
    print(f"\n[나이별 분포]")
    print(f"  1일 미만:       {stats['by_age']['1d']:,}개")
    print(f"  1-7일:          {stats['by_age']['7d']:,}개")
    print(f"  7-30일:         {stats['by_age']['30d']:,}개")
    print(f"  30일 이상:      {stats['by_age']['older']:,}개")


def main():
    parser = argparse.ArgumentParser(description='캐시 정리')
    parser.add_argument('--days', type=int, default=7, help='정리할 캐시 최대 나이 (일)')
    parser.add_argument('--all', action='store_true', help='전체 캐시 삭제')
    parser.add_argument('--stats', action='store_true', help='통계만 출력')
    parser.add_argument('--expired', action='store_true', help='만료된 캐시만 정리')

    args = parser.parse_args()

    cache_dir = PROJECT_ROOT / "data" / "cache"

    # 통계 출력
    stats = get_cache_stats(cache_dir)
    print_stats(stats)

    if args.stats:
        return

    # 정리 수행
    if args.all:
        print(f"\n전체 캐시를 삭제합니다...")
        deleted = clear_all_cache(cache_dir)
    elif args.expired:
        print(f"\n만료된 캐시를 정리합니다...")
        deleted = cleanup_expired(cache_dir)
    else:
        print(f"\n{args.days}일 이상 된 캐시를 정리합니다...")
        deleted = cleanup_old_cache(cache_dir, args.days)

    print(f"  삭제된 파일: {deleted:,}개")

    # 정리 후 통계
    if deleted > 0:
        new_stats = get_cache_stats(cache_dir)
        freed_mb = stats['total_size_mb'] - new_stats['total_size_mb']
        print(f"  확보된 용량: {freed_mb:.2f} MB")


if __name__ == "__main__":
    main()
