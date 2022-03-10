from pathlib import Path
import pandas
import click
import editdistance
from typing import List


CSV_SEPARATOR = ";"
CSV_LIST_SEPARATOR = "<!>"

PLAYERS_DATASET_TAG = "df_players"
MATCHES_DATASET_TAG = "df_matches"
RATING_CHANGES_DATASET_TAG = "df_rating_changes"

ROOT_DATA_PATH = Path(__file__).parent / "DATA"
PLAYER_DATABASE_CSV_PATH = ROOT_DATA_PATH / "players.csv"
MATCHES_DATABASE_CSV_PATH = ROOT_DATA_PATH  / "matches.csv"
RATING_CHANGES_DATABASE_CSV_PATH = ROOT_DATA_PATH / "changes.csv"
DATABASE_CSV_PATH_DICTIONARY = {
    PLAYERS_DATASET_TAG : PLAYER_DATABASE_CSV_PATH,
    MATCHES_DATASET_TAG: MATCHES_DATABASE_CSV_PATH,
    RATING_CHANGES_DATASET_TAG: RATING_CHANGES_DATABASE_CSV_PATH,
}

PLAYER_DATABASE_INDEX = "player_id"
MATCHES_DATABASE_INDEX = "match_id"
RATING_CHANGES_DATABASE_INDEX = "rating_change_id"
DATABASE_INDEX_DICTIONARY = {
    PLAYERS_DATASET_TAG: PLAYER_DATABASE_INDEX,
    MATCHES_DATASET_TAG: MATCHES_DATABASE_INDEX,
    RATING_CHANGES_DATASET_TAG: RATING_CHANGES_DATABASE_INDEX,
}

PLAYER_DATABASE_NAME_COLUMN = "name"
PLAYER_DATABASE_NICKNAMES_COLUMN = "nicknames"
PLAYER_DATABASE_RATING_COLUMN = "rating"
PLAYER_DATABASE_NON_INDEX_COLUMNS = [PLAYER_DATABASE_NAME_COLUMN, PLAYER_DATABASE_NICKNAMES_COLUMN, PLAYER_DATABASE_RATING_COLUMN]
MATCHES_DATABASE_NON_INDEX_COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals"]

RATING_CHANGES_DATABASE_NON_INDEX_COLUMNS = ["date", "player_id", "rating_change"]
DATABASE_NON_INDEX_COLUMNS_DICTIONARY = {
    PLAYERS_DATASET_TAG: PLAYER_DATABASE_NON_INDEX_COLUMNS,
    MATCHES_DATASET_TAG: MATCHES_DATABASE_NON_INDEX_COLUMNS,
    RATING_CHANGES_DATASET_TAG: RATING_CHANGES_DATABASE_NON_INDEX_COLUMNS,
}

DEFAULT_INITIAL_PLAYER_RATING = 2000
INITIAL_PLAYER_RATING_QUANTILE = 0.3
STRING_MATCHING_SEPARATION_FACTOR_FOR_EXACT_MATCH = 1.5

def load_df(df_tag):
    df_path = DATABASE_CSV_PATH_DICTIONARY[df_tag]
    index_column = DATABASE_INDEX_DICTIONARY[df_tag]
    if df_path.exists():
        df = pandas.read_csv(DATABASE_CSV_PATH_DICTIONARY[df_tag], sep=CSV_SEPARATOR, index_col=index_column)
    else:
        df = pandas.DataFrame(
            columns=[index_column] + DATABASE_NON_INDEX_COLUMNS_DICTIONARY[df_tag])
        df = df.set_index(index_column)
    return df


def save_df(df_tag, df: pandas.DataFrame):
    if not ROOT_DATA_PATH.is_dir():
        ROOT_DATA_PATH.mkdir()
    df_path = DATABASE_CSV_PATH_DICTIONARY[df_tag]
    df.to_csv(
        df_path, sep=CSV_SEPARATOR, columns=DATABASE_NON_INDEX_COLUMNS_DICTIONARY[df_tag],
         header=True, index=True, index_label=DATABASE_INDEX_DICTIONARY[df_tag])


def get_players_df():
    return load_df(PLAYERS_DATASET_TAG)


def set_players_df(df_players):
    save_df(PLAYERS_DATASET_TAG, df_players)


def get_matches_df():
    return load_df(MATCHES_DATASET_TAG)


def add_from_records(df_tag, records: List[dict], df, persist_into_database=True):
    if df is None:
        df = load_df(df_tag)
    index_col = DATABASE_INDEX_DICTIONARY[df_tag]
    max_id = df.index.max() if len(df.index) > 0 else 0
    new_index = []

    for record in records:
        for column_name in DATABASE_NON_INDEX_COLUMNS_DICTIONARY[df_tag]:
            assert column_name in record
        if index_col in record:
            _record_id = record[index_col]
            assert _record_id not in df.index and _record_id not in new_index
        else:
            record[index_col] = max_id + 1
            max_id += 1
        new_index.append(record[index_col])

    df_new = pandas.DataFrame.from_records(records, index=index_col)
    df_new.fillna("", inplace=True)
    df = pandas.concat([df, df_new], axis=0)
    if persist_into_database:
        save_df(df_tag, df)
    return df


def round_rating(rating):
    if isinstance(rating, pandas.Series):
        return rating.round()
    else:
        return round(rating)


def add_players(player_records: List[dict], df_players=None, persist_into_database=True):
    if df_players is None:
        df_players = get_players_df()

    ratings = df_players[PLAYER_DATABASE_RATING_COLUMN]
    if len(ratings) == 0:
        initial_rating = DEFAULT_INITIAL_PLAYER_RATING
    else:
        initial_rating = round_rating(ratings.quantile(INITIAL_PLAYER_RATING_QUANTILE))

    for record in player_records:
        if PLAYER_DATABASE_RATING_COLUMN not in record:
            record[PLAYER_DATABASE_RATING_COLUMN] = initial_rating

    return add_from_records(PLAYERS_DATASET_TAG, player_records, df_players, persist_into_database)


def list_players(df_players=None):
    if df_players is None:
        df_players = get_players_df()
    click.echo(df_players.to_markdown())


def search_pattern_for_players(df_players, query_string):
    keywords_per_player = df_players.apply(
        lambda player_row: player_row[PLAYER_DATABASE_NICKNAMES_COLUMN].split(CSV_LIST_SEPARATOR) + player_row[PLAYER_DATABASE_NAME_COLUMN])
    edlib_distances = keywords_per_player.apply(
        lambda keywords: min([editdistance.eval(keyword, query_string) for keyword in keywords])
    )
    return edlib_distances


def is_search_patter_exact(search_pattern):
    return (STRING_MATCHING_SEPARATION_FACTOR_FOR_EXACT_MATCH * search_pattern.min() > search_pattern).sum() <= 1


def remove_players(identifiers):
    df_players = get_players_df()

    index_identifiers = []
    name_indentifiers = []
    remaining_identifiers = []
    for identifier in identifiers:
        if identifier.isdigit():
            index_identifiers.append(int(identifier))
        elif (df_players[PLAYER_DATABASE_NAME_COLUMN] == identifier).any():
            name_indentifiers.append(identifier)
        else:
            remaining_identifiers.append(identifier)

    remove_by_id = df_players.index[df_players.index.isin(index_identifiers)].to_series()
    remove_by_name = df_players.index[df_players[PLAYER_DATABASE_NAME_COLUMN] == identifier].to_series()

    remove_by_pattern_matching = pandas.Series([], dtype=int)
    pattern_issues = []
    search_patterns = [search_pattern_for_players(df_players, identifier) for identifier in remaining_identifiers]
    for keyword, search_pattern in zip(remaining_identifiers, search_patterns):
        if is_search_patter_exact(search_pattern):
            remove_by_pattern_matching.append(search_pattern.idxmin)
        else:
            pattern_issues.append(
                (keyword, df_players[search_pattern < STRING_MATCHING_SEPARATION_FACTOR_FOR_EXACT_MATCH * search_pattern.min()]))

    remove_indices = pandas.concat([remove_by_id, remove_by_name, remove_by_pattern_matching], axis=0)
    removed_players = df_players.loc[remove_indices]
    nr_removed = len(removed_players.index)
    if nr_removed > 0:
        df_players.drop(remove_indices, axis=0, inplace=True)
        set_players_df(df_players)
        click.echo(f"Removed {nr_removed} player{'s' if nr_removed > 1 else ''}:")
        click.echo(f"{removed_players.to_markdown()}")
    else:
        click.echo(f"Could not find appropriate players to remove")

@click.group()
def rankings():
    pass


@rankings.group()
def add():
    '''Adds data to the database'''

@add.command()
@click.option("--rating", "-r", "rating", default=-1)
@click.argument("name", nargs=1)
@click.argument("nicknames", nargs=-1)
def player(rating, name, nicknames):
    '''Adds a new player player with given elo and list of nicknames'''
    player_records = [{
        PLAYER_DATABASE_NAME_COLUMN: name,
        PLAYER_DATABASE_NICKNAMES_COLUMN: CSV_SEPARATOR.join([nick.lower() for nick in nicknames]),
    }]
    if rating > 0:
        player_records[PLAYER_DATABASE_RATING_COLUMN] = round_rating(rating)
    add_players(player_records)


@rankings.group()
def remove():
    '''Removes ???'''


@remove.command()
@click.argument("identifiers", nargs=-1)
def player(identifiers):
    remove_players(identifiers)


@rankings.group()
def list():
    '''Lists data from the database'''
    pass


@list.command()
@click.option("--rating/--no-rating", default=False)
def players(rating):
    list_players()


@rankings.command()
@click.option("--size", "-s", default=5)
def draft(size):
    '''Draft teams'''
    pass


@rankings.command()
def score():
    pass


if __name__ == "__main__":
    rankings()
