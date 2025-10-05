from pathlib import Path
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime

import rankings as rankings_app


def main(csv_path: Path):
    df = pd.read_csv(str(csv_path))
    df = df.iloc[1:]
    existing_players = set(rankings_app.get_players_df()[rankings_app.PLAYER_DATABASE_NAME_COLUMN])

    for index, row in df.iterrows():
        for team_str in [row["Team A"], row["Team B"]]:
            players = team_str.split("\n")
            for p in players:
                if p not in existing_players:
                    name = p
                    rankings_app.add_player_command(-1, name, [])
                    existing_players.add(p)
        team_a_players = rankings_app.identify_players(row["Team A"].split("\n"))
        team_b_players = rankings_app.identify_players(row["Team B"].split("\n"))
        if len(team_a_players) != len(team_b_players):
            continue

        games_won_a = 0
        games_won_b = 0
        for score in row["Score (A:B)"].strip().split("\n"):
            score_parts = score.split(":")
            score_a = int(score_parts[0].strip())
            score_b = int(score_parts[1].strip())
            if score_a >= score_b + 2:
                games_won_a += 1
            if score_b >= score_a + 2:
                games_won_b += 1

        date = datetime.strptime(row["Date"] + " 20:00", "%d.%m.%Y %H:%M")
        rankings_app.add_matches([
            {
                rankings_app.MATCHES_DATABASE_DATETIME_COLUMN: date,
                rankings_app.MATCHES_DATABASE_HOME_TEAM_COLUMN: team_a_players.values(),
                rankings_app.MATCHES_DATABASE_AWAY_TEAM_COLUMN: team_b_players.values(),
                rankings_app.MATCHES_DATABASE_HOME_GOALS_COLUMN: games_won_a,
                rankings_app.MATCHES_DATABASE_AWAY_GOALS_COLUMN: games_won_b,
            }
        ])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('csv', type=Path, help='Path to csv file')
    args = parser.parse_args()
    main(args.csv)
