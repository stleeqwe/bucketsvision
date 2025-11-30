#!/usr/bin/env python3
"""
Collect NBA Games Data.

NBA Stats API를 사용하여 경기 데이터를 수집합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.utils.logger import logger
from config.settings import settings

# NBA Stats API 헤더
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com"
}


def get_season_string(season: int) -> str:
    """시즌 연도를 NBA 형식으로 변환 (예: 2024 -> 2023-24)"""
    return f"{season-1}-{str(season)[-2:]}"


def fetch_season_games(season: int, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    시즌 경기 데이터 가져오기.

    Args:
        season: 시즌 연도 (예: 2024 = 23-24 시즌)
        season_type: "Regular Season" 또는 "Playoffs"

    Returns:
        경기 DataFrame
    """
    season_str = get_season_string(season)

    url = "https://stats.nba.com/stats/leaguegamefinder"
    params = {
        "PlayerOrTeam": "T",
        "LeagueID": "00",
        "Season": season_str,
        "SeasonType": season_type
    }

    logger.info(f"Fetching games for {season_str} {season_type}...")

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # 결과 파싱
        result_sets = data.get("resultSets", [])
        if not result_sets:
            logger.warning("No result sets found")
            return pd.DataFrame()

        headers = result_sets[0]["headers"]
        rows = result_sets[0]["rowSet"]

        df = pd.DataFrame(rows, columns=headers)

        logger.info(f"  Fetched {len(df)} team-game records")

        return df

    except requests.RequestException as e:
        logger.error(f"Failed to fetch games: {e}")
        return pd.DataFrame()


def process_games(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    팀-게임 데이터를 게임 단위로 변환.

    각 팀별 레코드를 홈/어웨이 매칭하여 게임 단위로 변환합니다.
    """
    if team_games.empty:
        return pd.DataFrame()

    # 필요한 컬럼 확인
    required_cols = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE",
                     "MATCHUP", "WL", "PTS"]

    missing = [c for c in required_cols if c not in team_games.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}")
        return pd.DataFrame()

    games = []
    processed_game_ids = set()

    for game_id, group in team_games.groupby("GAME_ID"):
        if game_id in processed_game_ids:
            continue

        if len(group) != 2:
            continue

        processed_game_ids.add(game_id)

        # 홈/어웨이 구분 (MATCHUP에 "@"가 있으면 어웨이)
        row1, row2 = group.iloc[0], group.iloc[1]

        if "@" in str(row1["MATCHUP"]):
            away_row, home_row = row1, row2
        else:
            home_row, away_row = row1, row2

        games.append({
            "game_id": game_id,
            "game_date": home_row["GAME_DATE"],
            "home_team_id": int(home_row["TEAM_ID"]),
            "away_team_id": int(away_row["TEAM_ID"]),
            "home_team": home_row["TEAM_ABBREVIATION"],
            "away_team": away_row["TEAM_ABBREVIATION"],
            "home_score": int(home_row["PTS"]),
            "away_score": int(away_row["PTS"]),
            "margin": int(home_row["PTS"]) - int(away_row["PTS"])
        })

    games_df = pd.DataFrame(games)

    if not games_df.empty:
        games_df["game_date"] = pd.to_datetime(games_df["game_date"])
        games_df = games_df.sort_values("game_date").reset_index(drop=True)

    logger.info(f"Processed {len(games_df)} unique games")

    return games_df


def collect_all_seasons(
    seasons: List[int],
    output_dir: Path,
    delay: float = 2.0
) -> Dict[int, pd.DataFrame]:
    """
    여러 시즌의 경기 데이터 수집.

    Args:
        seasons: 시즌 리스트
        output_dir: 출력 디렉토리
        delay: API 호출 간 지연 (초)

    Returns:
        시즌 -> DataFrame 딕셔너리
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_games = {}

    for season in seasons:
        logger.info(f"\n{'='*50}")
        logger.info(f"Collecting season {season}...")
        logger.info(f"{'='*50}")

        # 레귤러 시즌
        team_games = fetch_season_games(season, "Regular Season")

        if team_games.empty:
            logger.warning(f"No data for season {season}")
            time.sleep(delay)
            continue

        # 게임 단위로 변환
        games = process_games(team_games)
        games["season"] = season

        # 저장
        output_path = output_dir / f"season_{season}.parquet"
        games.to_parquet(output_path, index=False)
        logger.info(f"Saved to {output_path}")

        all_games[season] = games

        time.sleep(delay)

    # 통합 데이터 저장
    if all_games:
        combined = pd.concat(all_games.values(), ignore_index=True)
        combined_path = output_dir / "all_seasons.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info(f"\nSaved combined data ({len(combined)} games) to {combined_path}")

    return all_games


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Collect NBA Games Data")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025],
                        help="Seasons to collect (e.g., 2024 = 23-24 season)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between API calls")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else settings.data_dir / "raw" / "nba_stats" / "games"

    logger.info("NBA Games Data Collection")
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Output: {output_dir}")

    collect_all_seasons(args.seasons, output_dir, args.delay)

    logger.info("\nCollection complete!")


if __name__ == "__main__":
    main()
