"""
프로젝트 전역 상수.

NBA 팀 정보, ID 매핑 등 프로젝트 전반에서 사용되는 상수를 정의합니다.
"""

from typing import Dict


# NBA 팀 정보 (team_id -> {abbr, name, color})
TEAM_INFO: Dict[int, Dict[str, str]] = {
    1610612737: {"abbr": "ATL", "name": "Atlanta Hawks", "color": "#E03A3E"},
    1610612738: {"abbr": "BOS", "name": "Boston Celtics", "color": "#007A33"},
    1610612751: {"abbr": "BKN", "name": "Brooklyn Nets", "color": "#000000"},
    1610612766: {"abbr": "CHA", "name": "Charlotte Hornets", "color": "#1D1160"},
    1610612741: {"abbr": "CHI", "name": "Chicago Bulls", "color": "#CE1141"},
    1610612739: {"abbr": "CLE", "name": "Cleveland Cavaliers", "color": "#860038"},
    1610612742: {"abbr": "DAL", "name": "Dallas Mavericks", "color": "#00538C"},
    1610612743: {"abbr": "DEN", "name": "Denver Nuggets", "color": "#0E2240"},
    1610612765: {"abbr": "DET", "name": "Detroit Pistons", "color": "#C8102E"},
    1610612744: {"abbr": "GSW", "name": "Golden State Warriors", "color": "#1D428A"},
    1610612745: {"abbr": "HOU", "name": "Houston Rockets", "color": "#CE1141"},
    1610612754: {"abbr": "IND", "name": "Indiana Pacers", "color": "#002D62"},
    1610612746: {"abbr": "LAC", "name": "LA Clippers", "color": "#C8102E"},
    1610612747: {"abbr": "LAL", "name": "Los Angeles Lakers", "color": "#552583"},
    1610612763: {"abbr": "MEM", "name": "Memphis Grizzlies", "color": "#5D76A9"},
    1610612748: {"abbr": "MIA", "name": "Miami Heat", "color": "#98002E"},
    1610612749: {"abbr": "MIL", "name": "Milwaukee Bucks", "color": "#00471B"},
    1610612750: {"abbr": "MIN", "name": "Minnesota Timberwolves", "color": "#0C2340"},
    1610612740: {"abbr": "NOP", "name": "New Orleans Pelicans", "color": "#0C2340"},
    1610612752: {"abbr": "NYK", "name": "New York Knicks", "color": "#006BB6"},
    1610612760: {"abbr": "OKC", "name": "Oklahoma City Thunder", "color": "#007AC1"},
    1610612753: {"abbr": "ORL", "name": "Orlando Magic", "color": "#0077C0"},
    1610612755: {"abbr": "PHI", "name": "Philadelphia 76ers", "color": "#006BB6"},
    1610612756: {"abbr": "PHX", "name": "Phoenix Suns", "color": "#1D1160"},
    1610612757: {"abbr": "POR", "name": "Portland Trail Blazers", "color": "#E03A3E"},
    1610612758: {"abbr": "SAC", "name": "Sacramento Kings", "color": "#5A2D81"},
    1610612759: {"abbr": "SAS", "name": "San Antonio Spurs", "color": "#C4CED4"},
    1610612761: {"abbr": "TOR", "name": "Toronto Raptors", "color": "#CE1141"},
    1610612762: {"abbr": "UTA", "name": "Utah Jazz", "color": "#002B5C"},
    1610612764: {"abbr": "WAS", "name": "Washington Wizards", "color": "#002B5C"},
}

# 팀 약어 -> team_id 매핑
ABBR_TO_ID: Dict[str, int] = {v["abbr"]: k for k, v in TEAM_INFO.items()}

# team_id -> 팀 약어 매핑
ID_TO_ABBR: Dict[int, str] = {k: v["abbr"] for k, v in TEAM_INFO.items()}
