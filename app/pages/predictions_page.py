"""
예측 페이지.

리팩토링 Phase 4: main.py에서 추출.
PredictionPipeline을 사용하여 예측 로직 단순화.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional

import streamlit as st
from scipy.stats import norm

from app.services.data_loader import DataLoader, TEAM_INFO
from app.services.predictor_v5 import V5PredictionService
from app.components.game_card_v2 import (
    render_game_card,
    render_day_summary,
    render_no_games,
)
from app.components.sidebar.date_picker import DateSelection
from app.theme import COLORS
from app.utils.date_utils import get_kst_date, get_weekday_kr


def render_predictions_page(
    loader: DataLoader,
    predictor: V5PredictionService,
    date_selection: DateSelection,
    team_epm: Dict[int, Dict],
    et_today: date,
) -> None:
    """
    예측 페이지 렌더링.

    Args:
        loader: 데이터 로더
        predictor: 예측 서비스
        date_selection: 날짜 선택 결과
        team_epm: 팀 EPM 데이터
        et_today: 오늘 날짜 (ET)
    """
    st.subheader(date_selection.header_text)

    # 날짜 범위의 모든 경기 가져오기
    all_games_by_date = _load_games_by_date(
        loader,
        date_selection.start_date,
        date_selection.end_date
    )

    total_games = sum(len(games) for games in all_games_by_date.values())

    if total_games == 0:
        render_no_games()
        return

    # 전체 통계 (다중 날짜 모드)
    if date_selection.mode != "daily":
        total_finished = sum(
            sum(1 for g in games if g.get("game_status") == 3)
            for games in all_games_by_date.values()
        )
        st.caption(f"총 {total_games}경기 | 종료 {total_finished}경기")

    # 예측 적중 추적
    grand_total_finished = 0
    grand_total_correct = 0
    grand_total_error = 0.0

    # 날짜별로 경기 렌더링
    sorted_dates = sorted(all_games_by_date.keys(), reverse=True)

    for game_date in sorted_dates:
        games = all_games_by_date[game_date]

        # 날짜 헤더 (다중 날짜 모드)
        if date_selection.mode != "daily":
            _render_date_header(game_date, len(games))

        # 일별 상태 요약
        if date_selection.mode == "daily":
            _render_daily_status(games)

        # 일별 적중 추적
        day_finished, day_correct, day_error = _render_games(
            games=games,
            game_date=game_date,
            loader=loader,
            predictor=predictor,
            team_epm=team_epm,
        )

        grand_total_finished += day_finished
        grand_total_correct += day_correct
        grand_total_error += day_error

        # 일별 요약 (다중 날짜 모드)
        if day_finished > 0 and date_selection.mode != "daily":
            accuracy = day_correct / day_finished * 100
            mae = day_error / day_finished
            st.caption(f"📊 {day_finished}경기 중 {day_correct}경기 적중 ({accuracy:.1f}%) | MAE: {mae:.1f}pt")
            st.markdown("---")

    # 전체 통계 요약
    _render_summary(
        date_selection.mode,
        grand_total_finished,
        grand_total_correct,
        grand_total_error,
    )


def _load_games_by_date(
    loader: DataLoader,
    start_date: date,
    end_date: date,
) -> Dict[date, List[Dict]]:
    """날짜별 경기 로딩"""
    all_games_by_date = {}

    with st.spinner("경기 일정 로딩 중..."):
        current_date = start_date
        while current_date <= end_date:
            games = loader.get_games(current_date)
            if games:
                all_games_by_date[current_date] = games
            current_date += timedelta(days=1)

    return all_games_by_date


def _render_date_header(game_date: date, game_count: int) -> None:
    """날짜 헤더 렌더링"""
    kst_date = get_kst_date(game_date)
    weekday_kr = get_weekday_kr(kst_date)
    st.markdown(
        f"### {kst_date.strftime('%m월 %d일')} ({weekday_kr}) - {game_count}경기"
    )


def _render_daily_status(games: List[Dict]) -> None:
    """일별 경기 상태 요약"""
    live_count = sum(1 for g in games if g.get("game_status") == 2)
    scheduled_count = sum(1 for g in games if g.get("game_status") == 1)
    finished_count = sum(1 for g in games if g.get("game_status") == 3)

    status_parts = []
    if live_count > 0:
        status_parts.append(f"🔴 진행 {live_count}")
    if scheduled_count > 0:
        status_parts.append(f"⏰ 예정 {scheduled_count}")
    if finished_count > 0:
        status_parts.append(f"✅ 종료 {finished_count}")
    if status_parts:
        st.caption(" | ".join(status_parts))


def _render_games(
    games: List[Dict],
    game_date: date,
    loader: DataLoader,
    predictor: V5PredictionService,
    team_epm: Dict[int, Dict],
) -> tuple:
    """
    경기 렌더링 및 통계 수집.

    Returns:
        (finished_count, correct_count, total_error) 튜플
    """
    day_finished = 0
    day_correct = 0
    day_error = 0.0

    for game in games:
        game_status = game.get("game_status", 1)

        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})

        home_abbr = home_info.get("abbr", "UNK")
        away_abbr = away_info.get("abbr", "UNK")

        # B2B 정보
        home_b2b = game.get("home_b2b", False)
        away_b2b = game.get("away_b2b", False)

        # V5.4 피처 생성
        features = loader.build_v5_4_features(
            home_id, away_id, team_epm, game_date
        )

        # 기본 예측
        base_prob = predictor.predict_proba(features)

        # 점수
        home_score = game.get("home_score")
        away_score = game.get("away_score")

        # 부상 분석 (예정된 경기만)
        home_injury_summary = None
        away_injury_summary = None
        home_prob_shift = 0.0
        away_prob_shift = 0.0

        if game_status == 1:
            try:
                home_injury_summary = loader.get_injury_summary(
                    home_abbr, game_date, team_epm
                )
                away_injury_summary = loader.get_injury_summary(
                    away_abbr, game_date, team_epm
                )
                home_prob_shift = home_injury_summary.get("total_prob_shift", 0.0)
                away_prob_shift = away_injury_summary.get("total_prob_shift", 0.0)
            except Exception:
                pass

        # 부상 보정
        home_win_prob = predictor.apply_injury_adjustment(
            base_prob, home_prob_shift, away_prob_shift
        )

        # 마진 계산
        raw_margin = norm.ppf(home_win_prob) * 12.0
        if abs(home_win_prob - 0.5) > 0.25:
            predicted_margin = raw_margin * 0.85
        else:
            predicted_margin = raw_margin

        # 적중률 계산 (종료된 경기)
        if game_status == 3 and home_score is not None and away_score is not None:
            day_finished += 1
            predicted_home_win = home_win_prob >= 0.5
            actual_home_win = home_score > away_score
            actual_margin = home_score - away_score

            if predicted_home_win == actual_home_win:
                day_correct += 1

            day_error += abs(predicted_margin - actual_margin)

        # 배당 정보 (예정된 경기만)
        odds_info = None
        if game_status == 1:
            odds_info = loader.get_game_odds(home_abbr, away_abbr)

        # 게임 카드 렌더링
        game_id = game.get("game_id", f"{home_abbr}_{away_abbr}")
        render_game_card(
            home_team=home_abbr,
            away_team=away_abbr,
            home_name=home_info.get("name", "Unknown"),
            away_name=away_info.get("name", "Unknown"),
            home_color=home_info.get("color", COLORS["home"]),
            away_color=away_info.get("color", COLORS["away"]),
            game_time=game["game_time"],
            predicted_margin=round(predicted_margin, 1),
            home_win_prob=home_win_prob,
            game_status=game_status,
            home_score=home_score,
            away_score=away_score,
            home_b2b=home_b2b,
            away_b2b=away_b2b,
            hide_result=(game_status == 2),
            odds_info=odds_info,
            game_id=game_id,
            enable_custom_input=(game_status == 1),
            home_injury_summary=home_injury_summary,
            away_injury_summary=away_injury_summary,
        )

    return day_finished, day_correct, day_error


def _render_summary(
    date_mode: str,
    total_finished: int,
    total_correct: int,
    total_error: float,
) -> None:
    """통계 요약 렌더링"""
    if total_finished == 0:
        return

    if date_mode == "daily":
        mae = total_error / total_finished
        render_day_summary(total_finished, total_correct, mae)
    else:
        accuracy = total_correct / total_finished * 100
        mae = total_error / total_finished
        acc_color = COLORS['success'] if accuracy >= 50 else COLORS['error']

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
                border: 1px solid #2d4a6f;
                border-radius: 12px;
                padding: 24px;
                margin: 20px 0;
                text-align: center;
            ">
                <div style="font-size: 1rem; color: {COLORS['text_secondary']}; margin-bottom: 12px;">
                    📊 전체 예측 성과
                </div>
                <div style="display: flex; justify-content: center; gap: 40px;">
                    <div>
                        <div style="font-size: 0.8rem; color: {COLORS['text_muted']};">적중률</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {acc_color};">
                            {accuracy:.1f}%
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; color: {COLORS['text_muted']};">평균 오차</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['text_secondary']};">
                            {mae:.1f}pt
                        </div>
                    </div>
                </div>
                <div style="font-size: 0.9rem; color: {COLORS['text_muted']}; margin-top: 16px;">
                    {total_finished}경기 중 {total_correct}경기 적중
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
