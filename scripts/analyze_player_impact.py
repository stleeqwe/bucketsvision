#!/usr/bin/env python3
"""
Player Impact Analysis Pipeline.

부상 영향도 분석의 전체 파이프라인을 실행합니다.

단계:
1. 선수 경기 출전 데이터 수집 (NBA Stats API)
2. On/Off 분석 (Adjusted)
3. Mixed Effects Regression
4. Bayesian Hierarchical Model
5. 앙상블 통합
6. 결과 저장 및 리포트 생성

사용법:
    python scripts/analyze_player_impact.py --seasons 2025 2026 --min-games 10
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import List, Optional

import pandas as pd
import numpy as np

from src.utils.logger import logger
from src.utils.helpers import ensure_dir
from config.settings import settings


def collect_player_games(data_dir: Path, seasons: List[int]) -> pd.DataFrame:
    """선수 경기 출전 데이터 수집"""
    from src.data_collection.player_game_collector import PlayerGameCollector

    logger.info("=" * 70)
    logger.info("Step 1: Collecting Player Game Data")
    logger.info("=" * 70)

    collector = PlayerGameCollector(
        data_dir=data_dir,
        cache_dir=data_dir / "cache" / "nba_stats"
    )

    try:
        df = collector.collect_multiple_seasons(seasons, save_individual=True)
        logger.info(f"Collected {len(df)} player game records")
        return df
    finally:
        collector.close()


def run_on_off_analysis(
    data_dir: Path,
    season: int,
    min_games: int = 10
) -> pd.DataFrame:
    """On/Off 분석 실행"""
    from src.analysis.player_on_off_analyzer import PlayerOnOffAnalyzer

    logger.info("=" * 70)
    logger.info(f"Step 2: On/Off Analysis (Season {season})")
    logger.info("=" * 70)

    # 데이터 로드
    games_path = data_dir / "raw" / "nba_stats" / "games" / f"season_{season}.parquet"
    player_games_path = data_dir / "raw" / "nba_stats" / "player_games" / f"season_{season}.parquet"
    team_epm_path = data_dir / "raw" / "dnt" / "team_epm" / f"season_{season}.parquet"

    if not games_path.exists():
        logger.error(f"Games data not found: {games_path}")
        return pd.DataFrame()

    if not player_games_path.exists():
        logger.error(f"Player games data not found: {player_games_path}")
        return pd.DataFrame()

    games_df = pd.read_parquet(games_path)
    player_games_df = pd.read_parquet(player_games_path)
    team_epm_df = pd.read_parquet(team_epm_path) if team_epm_path.exists() else None

    logger.info(f"  Games: {len(games_df)}")
    logger.info(f"  Player game records: {len(player_games_df)}")

    # 분석 실행
    analyzer = PlayerOnOffAnalyzer(
        games_df=games_df,
        player_games_df=player_games_df,
        team_epm_df=team_epm_df,
        min_games_on=min_games
    )

    results_df = analyzer.get_all_players_impact()

    # 저장
    output_dir = data_dir / "processed" / "player_impact"
    ensure_dir(output_dir)
    output_path = output_dir / f"season_{season}.parquet"
    results_df.to_parquet(output_path, index=False)

    logger.info(f"  Analyzed {len(results_df)} players")
    logger.info(f"  Saved to {output_path}")

    # 상위 선수 출력
    if not results_df.empty:
        logger.info("\n  Top 10 Impact Players:")
        for _, row in results_df.head(10).iterrows():
            sig = "*" if row.get("is_significant", False) else ""
            logger.info(
                f"    {row['player_name']:<20} ({row.get('team_abbr', ''):<3}): "
                f"{row['adjusted_impact']:+.2f}{sig} "
                f"(On={row['games_on']}, Off={row['games_off']})"
            )

    return results_df


def run_mixed_effects(
    data_dir: Path,
    season: int,
    min_games: int = 5
) -> pd.DataFrame:
    """Mixed Effects 모델 실행"""
    from src.analysis.mixed_effects_model import MixedEffectsPlayerModel

    logger.info("=" * 70)
    logger.info(f"Step 3: Mixed Effects Regression (Season {season})")
    logger.info("=" * 70)

    # 데이터 로드
    games_path = data_dir / "raw" / "nba_stats" / "games" / f"season_{season}.parquet"
    player_games_path = data_dir / "raw" / "nba_stats" / "player_games" / f"season_{season}.parquet"

    if not games_path.exists() or not player_games_path.exists():
        logger.warning("Required data not found, skipping Mixed Effects")
        return pd.DataFrame()

    games_df = pd.read_parquet(games_path)
    player_games_df = pd.read_parquet(player_games_path)

    # 모델 적합
    model = MixedEffectsPlayerModel(min_games_for_effect=min_games)

    try:
        model_data, target_players = model.prepare_data(games_df, player_games_df)

        if len(target_players) == 0:
            logger.warning("No target players found")
            return pd.DataFrame()

        logger.info(f"  Target players: {len(target_players)}")
        logger.info(f"  Games: {len(model_data)}")

        result = model.fit(model_data, target_players)

        logger.info(f"  Model fit complete")
        logger.info(f"    Log-likelihood: {result.log_likelihood:.2f}")
        logger.info(f"    AIC: {result.aic:.2f}")
        logger.info(f"    Team variance: {result.team_variance:.2f}")
        logger.info(f"    Residual variance: {result.residual_variance:.2f}")

        # 결과 저장
        effects_df = model.get_all_effects_df()

        output_dir = data_dir / "processed" / "mixed_effects"
        ensure_dir(output_dir)
        output_path = output_dir / f"season_{season}.parquet"
        effects_df.to_parquet(output_path, index=False)

        logger.info(f"  Saved {len(effects_df)} player effects to {output_path}")

        return effects_df

    except Exception as e:
        logger.error(f"Mixed Effects failed: {e}")
        return pd.DataFrame()


def run_bayesian_model(
    data_dir: Path,
    season: int,
    min_games: int = 5
) -> pd.DataFrame:
    """Bayesian 모델 실행"""
    from src.analysis.bayesian_impact import BayesianPlayerImpactModel

    logger.info("=" * 70)
    logger.info(f"Step 4: Bayesian Hierarchical Model (Season {season})")
    logger.info("=" * 70)

    # EPM 데이터 로드
    epm_path = data_dir / "raw" / "dnt" / "season_epm" / f"season_{season}.parquet"
    if not epm_path.exists():
        logger.warning(f"Player EPM not found: {epm_path}")
        return pd.DataFrame()

    player_epm_df = pd.read_parquet(epm_path)
    logger.info(f"  Player EPM records: {len(player_epm_df)}")

    # On/Off 결과 로드
    on_off_path = data_dir / "processed" / "player_impact" / f"season_{season}.parquet"
    if on_off_path.exists():
        on_off_df = pd.read_parquet(on_off_path)
    else:
        on_off_df = pd.DataFrame()

    logger.info(f"  On/Off results: {len(on_off_df)}")

    # 모델 적합
    model = BayesianPlayerImpactModel(min_games=min_games)
    results_df = model.fit(player_epm_df, on_off_df)

    # 저장
    output_dir = data_dir / "processed" / "bayesian_impact"
    ensure_dir(output_dir)
    output_path = output_dir / f"season_{season}.parquet"
    results_df.to_parquet(output_path, index=False)

    logger.info(f"  Saved {len(results_df)} Bayesian estimates to {output_path}")

    # 요약 통계
    if not results_df.empty:
        avg_reduction = results_df["uncertainty_reduction"].mean()
        high_conf = (results_df["prob_large_effect"] > 0.8).sum()
        logger.info(f"  Average uncertainty reduction: {avg_reduction:.2%}")
        logger.info(f"  High confidence impacts (P(|θ|>3) > 0.8): {high_conf}")

    return results_df


def run_ensemble(
    data_dir: Path,
    season: int
) -> pd.DataFrame:
    """앙상블 통합"""
    from src.features.injury_ensemble import InjuryImpactEnsemble

    logger.info("=" * 70)
    logger.info(f"Step 5: Ensemble Integration (Season {season})")
    logger.info("=" * 70)

    ensemble = InjuryImpactEnsemble.load(data_dir, season)

    results_df = ensemble.get_all_ensemble_impacts()

    # 저장
    output_dir = data_dir / "processed" / "ensemble_impact"
    ensure_dir(output_dir)
    output_path = output_dir / f"season_{season}.parquet"
    results_df.to_parquet(output_path, index=False)

    logger.info(f"  Ensemble results: {len(results_df)} players")
    logger.info(f"  Saved to {output_path}")

    # 상위 영향도 선수 출력
    if not results_df.empty:
        logger.info("\n  Top 15 Ensemble Impact Players:")
        logger.info("  " + "-" * 80)
        logger.info(f"  {'Player':<20} {'Team':<5} {'Impact':>8} {'Std':>6} {'Conf':>6} {'Sources':>8}")
        logger.info("  " + "-" * 80)

        for _, row in results_df.head(15).iterrows():
            logger.info(
                f"  {row['player_name']:<20} {row.get('team_abbr', ''):<5} "
                f"{row['ensemble_impact']:>+8.2f} {row['ensemble_std']:>6.2f} "
                f"{row['confidence']:>6.2f} {row['sources_count']:>8}"
            )

    return results_df


def generate_report(
    data_dir: Path,
    season: int,
    results_df: pd.DataFrame
):
    """결과 리포트 생성"""
    logger.info("=" * 70)
    logger.info("Step 6: Generating Report")
    logger.info("=" * 70)

    if results_df.empty:
        logger.warning("No results to report")
        return

    report_lines = [
        f"# Player Impact Analysis Report",
        f"## Season {season}",
        "",
        "### Summary Statistics",
        f"- Total players analyzed: {len(results_df)}",
        f"- Players with high confidence (>0.7): {(results_df['confidence'] > 0.7).sum()}",
        f"- Average ensemble impact: {results_df['ensemble_impact'].mean():.2f}",
        f"- Impact range: [{results_df['ensemble_impact'].min():.2f}, {results_df['ensemble_impact'].max():.2f}]",
        "",
        "### Top 20 Impact Players (Absence = Team Disadvantage)",
        "",
        "| Rank | Player | Team | Impact | Uncertainty | Confidence |",
        "|------|--------|------|--------|-------------|------------|",
    ]

    for i, (_, row) in enumerate(results_df.head(20).iterrows(), 1):
        report_lines.append(
            f"| {i} | {row['player_name']} | {row.get('team_abbr', '')} | "
            f"{row['ensemble_impact']:+.2f} | {row['ensemble_std']:.2f} | {row['confidence']:.2%} |"
        )

    report_lines.extend([
        "",
        "### Method Comparison (Top 10 Players)",
        "",
        "| Player | On/Off | Mixed Effects | Bayesian | EPM | Ensemble |",
        "|--------|--------|---------------|----------|-----|----------|",
    ])

    for _, row in results_df.head(10).iterrows():
        on_off = f"{row.get('on_off_impact', 0):+.2f}" if row.get('on_off_impact') is not None else "N/A"
        mixed = f"{row.get('mixed_effects_impact', 0):+.2f}" if row.get('mixed_effects_impact') is not None else "N/A"
        bayes = f"{row.get('bayesian_impact', 0):+.2f}" if row.get('bayesian_impact') is not None else "N/A"
        epm = f"{row.get('epm_impact', 0):+.2f}" if row.get('epm_impact') is not None else "N/A"

        report_lines.append(
            f"| {row['player_name']} | {on_off} | {mixed} | {bayes} | {epm} | "
            f"{row['ensemble_impact']:+.2f} |"
        )

    report_lines.extend([
        "",
        "### Notes",
        "- **Impact**: Expected margin change when player is absent (negative = team disadvantage)",
        "- **Uncertainty**: Standard deviation of the impact estimate",
        "- **Confidence**: Overall reliability score (0-1) based on data quality and method agreement",
        "",
        f"Generated with BucketsVision Player Impact Analysis Pipeline",
    ])

    # 저장
    report_dir = data_dir.parent / "reports"
    ensure_dir(report_dir)
    report_path = report_dir / f"player_impact_season_{season}.md"

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"  Report saved to {report_path}")


def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(description="Player Impact Analysis Pipeline")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2025, 2026],
        help="Seasons to analyze"
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=10,
        help="Minimum games for analysis"
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection (use existing)"
    )
    parser.add_argument(
        "--only-collect",
        action="store_true",
        help="Only collect data, skip analysis"
    )

    args = parser.parse_args()

    data_dir = settings.data_dir

    logger.info("=" * 70)
    logger.info("Player Impact Analysis Pipeline")
    logger.info("=" * 70)
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Min games: {args.min_games}")
    logger.info(f"Data dir: {data_dir}")
    logger.info("")

    # Step 1: 데이터 수집
    if not args.skip_collection:
        collect_player_games(data_dir, args.seasons)

    if args.only_collect:
        logger.info("Data collection complete. Exiting.")
        return

    # 시즌별 분석
    for season in args.seasons:
        logger.info("\n" + "=" * 70)
        logger.info(f"Analyzing Season {season}")
        logger.info("=" * 70)

        # Step 2: On/Off 분석
        on_off_df = run_on_off_analysis(data_dir, season, args.min_games)

        # Step 3: Mixed Effects
        mixed_df = run_mixed_effects(data_dir, season, min(5, args.min_games))

        # Step 4: Bayesian
        bayesian_df = run_bayesian_model(data_dir, season, min(5, args.min_games))

        # Step 5: 앙상블
        ensemble_df = run_ensemble(data_dir, season)

        # Step 6: 리포트
        generate_report(data_dir, season, ensemble_df)

    logger.info("\n" + "=" * 70)
    logger.info("Analysis Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
