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

PATTERN_MATCHING_SEPARATION_FACTOR_FOR_EXACT_MATCH = 1.5
PATTERN_MATCHING_NO_MATCH_STRING = "NO_MATCH"
PATTERN_MATCHING_MULTIPLE_MATCHES ="MULTIPLE_MATCHES"
PATTERN_MATCHING_MAX_DISTANCE = 3

TEAM_SEPARTION_STRINGS = ["vs", "vs.", "against", "-", "<>", "<->", ":", "|"]


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
    players_df = load_df(PLAYERS_DATASET_TAG)
    players_df.loc[:, PLAYER_DATABASE_NICKNAMES_COLUMN] = players_df[PLAYER_DATABASE_NICKNAMES_COLUMN].fillna('')
    return players_df


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
    return df, df_new


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


def player_search_vector_for_query(df_players, query_string):
    keywords_per_player = df_players.apply(
        lambda player_row: player_row[PLAYER_DATABASE_NICKNAMES_COLUMN].split(CSV_LIST_SEPARATOR) + [player_row[PLAYER_DATABASE_NAME_COLUMN]], axis=1)
    edlib_distances = keywords_per_player.apply(
        lambda keywords: min([editdistance.eval(keyword, query_string) for keyword in keywords])
    )
    return edlib_distances


def is_search_pattern_precise(search_vector):
    return search_vector.min() <= PATTERN_MATCHING_MAX_DISTANCE

def is_search_vector_exact(search_vector):
    return (PATTERN_MATCHING_SEPARATION_FACTOR_FOR_EXACT_MATCH * search_vector.min() > search_vector).sum() <= 1


def get_match_from_search_vector(search_vector):
    return search_vector.index[search_vector.argmin()]


def match_queries_to_player_ids(df_players, queries):
    results = {}
    for query in queries:
        search_vector = player_search_vector_for_query(df_players, query)
        if not is_search_pattern_precise(search_vector):
            id_to_match = PATTERN_MATCHING_NO_MATCH_STRING
        elif not is_search_vector_exact(search_vector):
            id_to_match = PATTERN_MATCHING_MULTIPLE_MATCHES
        else:
            id_to_match = get_match_from_search_vector(search_vector)
        results[query] = id_to_match
    return results


def remove_players(identifiers):
    df_players = get_players_df()

    index_identifiers = []
    name_indentifiers = []
    remaining_identifiers = []
    invalid_index_identifiers = []
    for identifier in identifiers:
        if identifier.isdigit():
            player_id = int(identifier)
            if player_id in df_players.index:
                index_identifiers.append(player_id)
            else:
                invalid_index_identifiers.append(player_id)
        elif (df_players[PLAYER_DATABASE_NAME_COLUMN] == identifier).any():
            name_indentifiers.append(identifier)
        else:
            remaining_identifiers.append(identifier)

    remove_by_id = df_players.index[df_players.index.isin(index_identifiers)].to_series()
    remove_by_name = df_players.index[df_players[PLAYER_DATABASE_NAME_COLUMN] == identifier].to_series()

    matching_dict = match_queries_to_player_ids(df_players, remaining_identifiers)
    remove_by_pattern_matching = pandas.Series(
        [player_id for player_id in matching_dict.values() if player_id not in [PATTERN_MATCHING_MULTIPLE_MATCHES, PATTERN_MATCHING_NO_MATCH_STRING]])

    unrecognized_identifiers, undecided_identifiers = [
        [identifier for identifier, player_id in matching_dict.items() if player_id == const]
        for const in [PATTERN_MATCHING_NO_MATCH_STRING, PATTERN_MATCHING_MULTIPLE_MATCHES]]

    remove_indices = pandas.concat([remove_by_id, remove_by_name, remove_by_pattern_matching], axis=0)
    removed_players = df_players.loc[remove_indices]
    nr_removed = len(removed_players.index)
    if nr_removed > 0:
        df_players.drop(remove_indices, axis=0, inplace=True)
        set_players_df(df_players)
        click.echo(f"Removed {nr_removed} player{'s' if nr_removed > 1 else ''}:")
        click.echo(f"{removed_players.to_markdown()}")

    if len(invalid_index_identifiers) > 0:
        click.echo(f"No players with indices {invalid_index_identifiers} in the database")
    if len(unrecognized_identifiers) > 0:
        click.echo(f"Could not match identifiers {unrecognized_identifiers}")
    if len(undecided_identifiers) > 0:
        click.echo(f"Multiple players matched identifiers {undecided_identifiers}. Please be more specific or match players by name or id instead")


@click.group()
def rankings():
    pass


@rankings.group()
def add():
    '''Adds data to the database.'''


@add.command()
@click.option("--rating", "-r", "rating", default=-1, type=int, help="Initial ranking for the player")
@click.argument("name", nargs=1)
@click.argument("nicknames", nargs=-1)
def players(rating, name, nicknames):
    '''Creates a new player. Takes player's name as first argument and treats other arguments as player's nicknames.'''
    player_record = {
        PLAYER_DATABASE_NAME_COLUMN: name,
        PLAYER_DATABASE_NICKNAMES_COLUMN: CSV_SEPARATOR.join([nick.lower() for nick in nicknames]),
    }
    if rating > 0:
        player_record[PLAYER_DATABASE_RATING_COLUMN] = round_rating(rating)
    df, df_new = add_players([player_record])
    click.echo("Added the following player")
    click.echo(df_new.to_markdown())


@add.command(help=
f'''Adds a match to the database. Takes names or nicknames of players as arguments.
Teams need to be separated by any of the following {TEAM_SEPARTION_STRINGS}.''')
@click.option("--date", "-d", "date", type=str, default="")
@click.argument("agrs", nargs=-1)
def match(date, args):
    click.echo("Not implemented")


@rankings.group()
def remove():
    '''Removes data from the database.'''
    pass


@remove.command()
@click.argument("identifiers", nargs=-1)
def players(identifiers):
    '''Removes players from the database. Takes a list of identifiers which are matched with players' ids, '''\
    '''names and nicknames to find corresponding players.'''
    remove_players(identifiers)


@rankings.group()
def list():
    '''Lists data from the database.'''
    pass


@list.command()
@click.option("--rating/--no-rating", default=False)
def players(rating):
    '''Lists all active players.'''
    list_players()


@list.command()
def matches():
    '''Lists all matches.'''
    pass


@rankings.group(help='''Updates database.''')
def update():
    click.echo("Not implemented!")


@update.command(help='''Updates player data.''')
@click.option("--player-id", "-i", "-id", "--id", type=int, required=True)
@click.option("--column", "-c", type=click.Choice(PLAYER_DATABASE_NON_INDEX_COLUMNS, case_sensitive=False))
@click.argument("new_value", nargs=1)
def players(player_id, column_name, new_value):
    click.echo("Not implemented!")


@update.command(help='''Updates match data.''')
@click.option("--match-id", "-i", "-id", "--id", type=int, required=True)
@click.option("--column", "-c", type=click.Choice(MATCHES_DATABASE_NON_INDEX_COLUMNS, case_sensitive=False))
@click.argument("new_value", nargs=1)
def match(match_id, column_name, new_value):
    click.echo("Not implemented!")


@rankings.command()
@click.option("--size", "-s", default=5)
@click.argument("players", nargs=-1)
def draft(team_size, players):
    '''Separates players into two teams of approximate same rating. Takes names of players as arguments.'''
    pass


@rankings.command()
@click.option("--match_id", "-id", type=int, default=-1)
@click.argument("home_score", nargs=1)
@click.argument("away_score", nargs=1)
def score(match_id, home_score, away_score):
    '''Adds score to one of the non-scored matches and updates rankings of all players participating. '''\
    '''Takes the score for first ("home") and second ("away") team as arguments.'''
    pass


if __name__ == "__main__":
    rankings()
