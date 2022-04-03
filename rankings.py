import builtins
from datetime import datetime
from pathlib import Path
import pandas
import numpy
import click
import editdistance
from typing import List
import math
from rich.console import Console as RichConsole
from rich.table import Table as RichTable
import itertools

RICH_CONSOLE = RichConsole()

CSV_SEPARATOR = ";"
CSV_LIST_SEPARATOR = "|"

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

MATCHES_DATABASE_DATETIME_COLUMN = "UTC"
MATCHES_DATABASE_HOME_TEAM_COLUMN = "home_team"
MATCHES_DATABASE_AWAY_TEAM_COLUMN = "away_team"
MATCHES_DATABASE_HOME_GOALS_COLUMN = "home_goals"
MATCHES_DATABASE_AWAY_GOALS_COLUMN = "away_goals"
MATCHES_DATABASE_NON_INDEX_COLUMNS = [
    MATCHES_DATABASE_DATETIME_COLUMN,
    MATCHES_DATABASE_HOME_TEAM_COLUMN,
    MATCHES_DATABASE_AWAY_TEAM_COLUMN,
    MATCHES_DATABASE_HOME_GOALS_COLUMN,
    MATCHES_DATABASE_AWAY_GOALS_COLUMN]

RATING_CHANGES_DATABASE_DATETIME_COLUMN = "UTC"
RATING_CHANGES_PLAYER_ID_COLUMN = "player_id"
RATING_CHANGES_RATING_CHANGE_COLUMN = "rating_change"
RATING_CHANGES_DATABASE_NON_INDEX_COLUMNS = [
    RATING_CHANGES_DATABASE_DATETIME_COLUMN,
    RATING_CHANGES_PLAYER_ID_COLUMN,
    RATING_CHANGES_RATING_CHANGE_COLUMN]

DATABASE_NON_INDEX_COLUMNS_DICTIONARY = {
    PLAYERS_DATASET_TAG: PLAYER_DATABASE_NON_INDEX_COLUMNS,
    MATCHES_DATASET_TAG: MATCHES_DATABASE_NON_INDEX_COLUMNS,
    RATING_CHANGES_DATASET_TAG: RATING_CHANGES_DATABASE_NON_INDEX_COLUMNS,
}

DEFAULT_INITIAL_PLAYER_RATING = 2000
INITIAL_PLAYER_RATING_QUANTILE = 0.3
DEFAULT_RATING_DIFFERENCE_SO_THAT_ONE_PLAYER_WINS_TWICE_AS_OFTEN_THAN_THE_OTHER = 400
DEFAULT_NR_1_0_WINS_TO_GET_TWICE_AS_GOOD_AS_OPPONENT = 5
DRAFTING_RATING_DIFFERENCE_TO_OPTIMAL_TO_BE_TWO_TIMES_UNLIKELY = 15

PLAYER_IDENTIFICATION_NOT_IN_INDEX ="NOT_IN_INDEX"
PATTERN_MATCHING_SEPARATION_FACTOR_FOR_EXACT_MATCH = 1.5
PATTERN_MATCHING_NO_MATCH_STRING = "NO_MATCH"
PATTERN_MATCHING_MULTIPLE_MATCHES ="MULTIPLE_MATCHES"
PATTERN_MATCHING_MAX_DISTANCE = 3

TEAM_SEPARTION_STRINGS = ["vs", "vs.", "against", "-", "<>", "<->", ":", "|"]
REMOVE_NONESSENTIAL_MATCHES_STRING = "non-scored"


def display_string_to_user(string: str):
    RICH_CONSOLE.print(string)

def _raw_load_df(df_tag):
    df_path = DATABASE_CSV_PATH_DICTIONARY[df_tag]
    index_column = DATABASE_INDEX_DICTIONARY[df_tag]
    if df_path.exists():
        df = pandas.read_csv(DATABASE_CSV_PATH_DICTIONARY[df_tag], sep=CSV_SEPARATOR, index_col=index_column, dtype="str")
    else:
        df = pandas.DataFrame(
            columns=[index_column] + DATABASE_NON_INDEX_COLUMNS_DICTIONARY[df_tag])
        df = df.set_index(index_column)
    return df


def _raw_save_df(df_tag, df: pandas.DataFrame):
    if not ROOT_DATA_PATH.is_dir():
        ROOT_DATA_PATH.mkdir()
    df_path = DATABASE_CSV_PATH_DICTIONARY[df_tag]
    df.to_csv(
        df_path, sep=CSV_SEPARATOR, columns=DATABASE_NON_INDEX_COLUMNS_DICTIONARY[df_tag],
         header=True, index=True, index_label=DATABASE_INDEX_DICTIONARY[df_tag])


def _decode_into_list_of_strings(encoded_string):
    list_of_strings = encoded_string.split(CSV_LIST_SEPARATOR)
    if len(list_of_strings) == 0 and list_of_strings[0] == "":
        list_of_strings = []
    return list_of_strings


def _encode_list_of_strings(list_of_strings):
    return CSV_LIST_SEPARATOR.join(list_of_strings)


def _decode_into_list_of_ints(encoded_string):
    return [int(element) for element in _decode_into_list_of_strings(encoded_string)]


def _encode_list_of_ints(list_of_ints):
    return _encode_list_of_strings([str(element) for element in list_of_ints])


def get_players_df():
    df_players = _raw_load_df(PLAYERS_DATASET_TAG)
    df_players = df_players.astype({
        PLAYER_DATABASE_NAME_COLUMN: str,
        PLAYER_DATABASE_NICKNAMES_COLUMN: str,
        PLAYER_DATABASE_RATING_COLUMN: pandas.Int64Dtype(),
    })
    df_players.loc[:, PLAYER_DATABASE_NICKNAMES_COLUMN] = df_players[PLAYER_DATABASE_NICKNAMES_COLUMN].apply(_decode_into_list_of_strings)
    return df_players


def set_players_df(df_players):
    df_players.loc[:, PLAYER_DATABASE_NICKNAMES_COLUMN] = df_players[PLAYER_DATABASE_NICKNAMES_COLUMN].apply(_encode_list_of_strings)
    _raw_save_df(PLAYERS_DATASET_TAG, df_players)


def get_matches_df():
    df_matches = _raw_load_df(MATCHES_DATASET_TAG)
    df_matches = df_matches.astype({
        MATCHES_DATABASE_DATETIME_COLUMN: str,
        MATCHES_DATABASE_HOME_TEAM_COLUMN: str,
        MATCHES_DATABASE_AWAY_TEAM_COLUMN: str,
        MATCHES_DATABASE_HOME_GOALS_COLUMN: pandas.Int64Dtype(),
        MATCHES_DATABASE_AWAY_GOALS_COLUMN: pandas.Int64Dtype(),
        }
    )
    df_matches[MATCHES_DATABASE_DATETIME_COLUMN] = pandas.to_datetime(df_matches[MATCHES_DATABASE_DATETIME_COLUMN])
    for column_name in [MATCHES_DATABASE_HOME_TEAM_COLUMN, MATCHES_DATABASE_AWAY_TEAM_COLUMN]:
        df_matches.loc[:, column_name] = df_matches[column_name].apply(_decode_into_list_of_ints)
    return df_matches


def set_matches_df(df_matches):
    for column_name in [MATCHES_DATABASE_HOME_TEAM_COLUMN, MATCHES_DATABASE_AWAY_TEAM_COLUMN]:
        df_matches.loc[:, column_name] = df_matches[column_name].apply(_encode_list_of_ints)
    _raw_save_df(MATCHES_DATASET_TAG, df_matches)


def get_rating_changes_df():
    df_rating_changes = _raw_load_df(RATING_CHANGES_DATASET_TAG)
    df_rating_changes = df_rating_changes.astype({
        RATING_CHANGES_DATABASE_DATETIME_COLUMN: str,
        RATING_CHANGES_RATING_CHANGE_COLUMN: pandas.Int64Dtype(),
    })
    df_rating_changes[RATING_CHANGES_DATABASE_DATETIME_COLUMN] = pandas.to_datetime(df_rating_changes[RATING_CHANGES_DATABASE_DATETIME_COLUMN])
    return df_rating_changes


def set_rating_changes_df(df_rating_changes):
    _raw_save_df(RATING_CHANGES_DATASET_TAG, df_rating_changes)


_DF_TAG_TP_LOAD_FUNCTION = {
    PLAYERS_DATASET_TAG: get_players_df,
    MATCHES_DATASET_TAG: get_matches_df,
    RATING_CHANGES_DATASET_TAG: get_rating_changes_df,
}

_DF_TAG_TO_SAVE_FUNCTION = {
    PLAYERS_DATASET_TAG: set_players_df,
    MATCHES_DATASET_TAG: set_matches_df,
    RATING_CHANGES_DATASET_TAG: set_rating_changes_df,
}


def load_df(df_tag):
    return _DF_TAG_TP_LOAD_FUNCTION[df_tag]()


def save_df(df_tag, df):
    return _DF_TAG_TO_SAVE_FUNCTION[df_tag](df)


def display_players_df(df_players: pandas.DataFrame, is_sort_by_rating=False, is_display_nicknames=True) -> str:
    df_to_display = df_players
    if not is_display_nicknames:
        df_to_display = df_to_display.drop(PLAYER_DATABASE_NICKNAMES_COLUMN, axis=1)
    if is_sort_by_rating:
        df_to_display = df_to_display.sort_values(by=PLAYER_DATABASE_RATING_COLUMN, ascending=False)
    return df_to_display.to_markdown()


def _match_display_is_match_scored(match_row):
    return all([not pandas.isnull(match_row[team_score_col]) for team_score_col in [MATCHES_DATABASE_HOME_GOALS_COLUMN, MATCHES_DATABASE_AWAY_GOALS_COLUMN]])


def _match_display_table_get_score_strings(df_players, home_score, away_score, home_player_ids, away_player_ids):
    if pandas.isnull(home_score) or pandas.isnull(away_score):
        home_avg_rating, away_avg_rating = [df_players.loc[player_ids, PLAYER_DATABASE_RATING_COLUMN].mean() for player_ids in [home_player_ids, away_player_ids]]
        home_win_prob, away_win_prob = compute_win_probabilities(home_avg_rating, away_avg_rating, DEFAULT_RATING_DIFFERENCE_SO_THAT_ONE_PLAYER_WINS_TWICE_AS_OFTEN_THAN_THE_OTHER)
        return [f"{rating:.0f}\n({win_prob*100:.1f}%)" for rating, win_prob in [(home_avg_rating, home_win_prob), (away_avg_rating, away_win_prob)]]
    else:
        return f"{home_score}", f"{away_score}"


def display_matches_df(df_matches: pandas.DataFrame, df_players: pandas.DataFrame=None):
    if df_players is None:
        df_players = get_players_df()

    player_id_to_name_map = {row_index: row[PLAYER_DATABASE_NAME_COLUMN] for row_index, row in df_players.iterrows()}

    table = RichTable(show_lines=True)
    table.add_column("Id", vertical="middle")
    table.add_column("Date", vertical="middle")
    table.add_column("Home team")
    for _ in range(3):
        table.add_column("", vertical="middle", justify="center")
    table.add_column("Away team")

    for row_index, row in df_matches.iterrows():
        _date_string = "To play..." if pandas.isnull(row[MATCHES_DATABASE_DATETIME_COLUMN]) else f"{row[MATCHES_DATABASE_DATETIME_COLUMN].date()}"
        _home_team_string, away_team_string = [
            "\n".join([player_id_to_name_map[player_id] for player_id in row[team_column]])
            for team_column in [MATCHES_DATABASE_HOME_TEAM_COLUMN, MATCHES_DATABASE_AWAY_TEAM_COLUMN]
            ]
        _home_score_string, _away_score_string = _match_display_table_get_score_strings(
            df_players, row[MATCHES_DATABASE_HOME_GOALS_COLUMN], row[MATCHES_DATABASE_AWAY_GOALS_COLUMN],
            row[MATCHES_DATABASE_HOME_TEAM_COLUMN], row[MATCHES_DATABASE_AWAY_TEAM_COLUMN])
        _teams_separator = "-" if _match_display_is_match_scored(row) else "vs."
        table.add_row(f"{row_index}", _date_string, _home_team_string, _home_score_string, _teams_separator, _away_score_string, away_team_string)
    display_string_to_user(table)


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
    df = pandas.concat([df, df_new], axis=0)
    if persist_into_database:
        save_df(df_tag, df)
    return df, df_new


def round_rating(rating):
    if isinstance(rating, pandas.Series):
        return rating.round()
    else:
        return round(rating)


def update_multiple_player_ratings(player_ids, rating_changes):
    UTC = datetime.now()
    rating_changes_records = [{
            RATING_CHANGES_PLAYER_ID_COLUMN: player_id,
            RATING_CHANGES_RATING_CHANGE_COLUMN: rating_change,
            RATING_CHANGES_DATABASE_DATETIME_COLUMN: UTC,
    } for player_id, rating_change in zip(player_ids, rating_changes)]
    return add_from_records(RATING_CHANGES_DATASET_TAG, rating_changes_records, get_rating_changes_df())


def update_player_rating(player_id, rating_changes):
    return update_multiple_player_ratings([player_id], [rating_changes])


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

    df , df_new_players = add_from_records(PLAYERS_DATASET_TAG, player_records, df_players, persist_into_database)
    update_multiple_player_ratings(df_new_players.index, df_new_players[PLAYER_DATABASE_RATING_COLUMN])

    return df, df_new_players


def add_matches(match_records: List[dict], df_matches=None, df_players=None, persist_into_databse=True):
    if df_matches is None:
        df_matches = get_matches_df()

    for match_record in match_records:
        home_team_ids = match_record[MATCHES_DATABASE_HOME_TEAM_COLUMN]
        away_team_ids = match_record[MATCHES_DATABASE_AWAY_TEAM_COLUMN]

        assert len(home_team_ids) == len(away_team_ids)
        assert len(home_team_ids) == len(home_team_ids)
        assert len(away_team_ids) == len(away_team_ids)
        assert not any([id in away_team_ids for id in home_team_ids])

        if MATCHES_DATABASE_HOME_GOALS_COLUMN in match_record and MATCHES_DATABASE_AWAY_GOALS_COLUMN in match_record:
            if MATCHES_DATABASE_DATETIME_COLUMN not in match_record:
                match_record[MATCHES_DATABASE_DATETIME_COLUMN] = datetime.now()
            adjust_player_ratings(match_record, df_players)
        else:
            match_record.update({
                MATCHES_DATABASE_HOME_GOALS_COLUMN: None,
                MATCHES_DATABASE_AWAY_GOALS_COLUMN: None,
                MATCHES_DATABASE_DATETIME_COLUMN: None
                })
    df_matches, df_matches_new = add_from_records(MATCHES_DATASET_TAG, match_records, df_matches, persist_into_database=persist_into_databse)
    return df_matches, df_matches_new


def score_match(match_id, home_goals, away_goals, df_matches=None, df_players=None):
    if df_matches is None:
        df_matches = get_matches_df()

    assert not find_essential_matches_mask(df_matches).loc[match_id]

    df_matches.loc[match_id, MATCHES_DATABASE_HOME_GOALS_COLUMN] = home_goals
    df_matches.loc[match_id, MATCHES_DATABASE_AWAY_GOALS_COLUMN] = away_goals
    df_matches.loc[match_id, MATCHES_DATABASE_DATETIME_COLUMN] = datetime.now()
    match_record = df_matches.loc[match_id]
    adjustments = adjust_player_ratings(match_record, df_players)
    set_matches_df(df_matches)
    return match_record, adjustments


def compute_win_probabilities(home_rating, away_rating, rating_diff_twice_as_good):
    exponent_scale_factor = math.log(2) / rating_diff_twice_as_good
    _intermediate_exp = math.exp((away_rating - home_rating) * exponent_scale_factor)
    home_win_prob = 1 / (1 + _intermediate_exp)
    away_win_prob = 1 - home_win_prob
    return home_win_prob, away_win_prob


def compute_rating_adjustment(home_rating, away_rating, home_goals, away_goals,
                              rating_diff_twice_as_good=DEFAULT_RATING_DIFFERENCE_SO_THAT_ONE_PLAYER_WINS_TWICE_AS_OFTEN_THAN_THE_OTHER,
                              nr_1_0_wins_needed_to_get_twice_as_good=DEFAULT_NR_1_0_WINS_TO_GET_TWICE_AS_GOOD_AS_OPPONENT):
    home_win_prob, away_win_prob = compute_win_probabilities(home_rating, away_rating, rating_diff_twice_as_good)

    SCALED_GRADIENT_EQUAL_PLAYERS = 1 / 2

    if home_goals > away_goals:
        home_scaled_gradient = away_win_prob
        away_scaled_gradient = -away_win_prob
    elif home_goals == away_goals:
        home_scaled_gradient = (away_win_prob - home_win_prob) / 2
        away_scaled_gradient = (home_win_prob - away_win_prob) / 2
    elif home_goals < away_goals:
        home_scaled_gradient = -home_win_prob
        away_scaled_gradient = home_win_prob

    score_modifier = min(3, abs(home_goals - away_goals))
    rating_adjustment_modifier = (rating_diff_twice_as_good / 2) / nr_1_0_wins_needed_to_get_twice_as_good / SCALED_GRADIENT_EQUAL_PLAYERS
    home_rating_adjustment, away_rating_adjustment = [
        round_rating(gradient * rating_adjustment_modifier * score_modifier) for gradient in [home_scaled_gradient, away_scaled_gradient]]
    return home_rating_adjustment, away_rating_adjustment


def adjust_player_ratings(match_record, df_players=None):
    if df_players is None:
        df_players = get_players_df()

    df_home_players, df_away_players = [
        df_players.loc[match_record[column_name]]
        for column_name in [MATCHES_DATABASE_HOME_TEAM_COLUMN, MATCHES_DATABASE_AWAY_TEAM_COLUMN]]

    total_home_rating, total_away_rating = [_df[PLAYER_DATABASE_RATING_COLUMN].mean() for _df in [df_home_players, df_away_players]]
    home_goals, away_goals = [match_record[score_column] for score_column in [MATCHES_DATABASE_HOME_GOALS_COLUMN, MATCHES_DATABASE_AWAY_GOALS_COLUMN]]

    home_rating_adjustment, away_rating_adjustment = compute_rating_adjustment(total_home_rating, total_away_rating, home_goals, away_goals)

    home_adjustments, away_adjustments = [
        _df.apply(lambda _: rating_adjustment, axis=1)
        for _df, rating_adjustment in [(df_home_players, home_rating_adjustment), (df_away_players, away_rating_adjustment)]]

    for _df, _adjustments in [(df_home_players, home_adjustments), (df_away_players, away_adjustments)]:
        update_multiple_player_ratings(_df.index, _adjustments)

    adjustments = pandas.concat([home_adjustments, away_adjustments])
    df_players.loc[adjustments.index, PLAYER_DATABASE_RATING_COLUMN] += adjustments
    set_players_df(df_players)
    return adjustments


def player_search_vector_for_query(df_players, query_string):
    keywords_per_player = df_players.apply(
        lambda player_row: player_row[PLAYER_DATABASE_NICKNAMES_COLUMN] + [player_row[PLAYER_DATABASE_NAME_COLUMN]], axis=1)
    edlib_distances = keywords_per_player.apply(
        lambda keywords: min([editdistance.eval(keyword.lower(), query_string.lower()) for keyword in keywords])
    )
    return edlib_distances


def _exchange_players(this_team, other_team, this_to_exchange, other_to_exchange):
    return numpy.concatenate([
        this_team[[id for id in range(len(this_team)) if id not in this_to_exchange]],
        other_team[other_to_exchange]
    ])


def find_teams_separation_with_small_ratings_difference(player_ids, df_players=None):
    _player_id_dtype = numpy.int32
    player_ids = numpy.fromiter(player_ids, dtype=_player_id_dtype)
    assert len(player_ids) % 2 == 0
    assert len(player_ids) < 13 # TODO the algorithm below is expected to take too long
    team_size = len(player_ids) // 2
    assert team_size > 0
    if df_players is None:
        df_players = get_players_df()
    ratings_pd_series = df_players.loc[player_ids, PLAYER_DATABASE_RATING_COLUMN]

    initial_home_team = player_ids[ :team_size]
    initial_away_team = player_ids[team_size: ]
    initial_difference = ratings_pd_series.loc[initial_home_team].sum() - ratings_pd_series.loc[initial_away_team].sum()

    # Given an initial team separation, any other team separation can be reached by making at most team_size/2 (rounded down) exchanges with the other team.
    max_exchanges = team_size // 2
    home_all_possible_exchanges, away_all_possible_exchanges = [], []
    all_rating_differences = []
    for nr_exchanges in range(max_exchanges + 1):
        for home_exchanges in itertools.combinations(range(team_size), nr_exchanges):
            for away_exchanges in itertools.combinations(range(team_size), nr_exchanges):
                home_exchanges = builtins.list(home_exchanges)
                away_exchanges = builtins.list(away_exchanges)
                home_exchange_rating, away_exchange_rating = [
                    ratings_pd_series.loc[initial[builtins.list(exchanges)]].sum() for initial, exchanges in [
                        (initial_home_team, home_exchanges), (initial_away_team, away_exchanges)]
                    ]
                all_rating_differences.append(
                    initial_difference - 2 * (home_exchange_rating - away_exchange_rating)
                )
                home_all_possible_exchanges.append(home_exchanges)
                away_all_possible_exchanges.append(away_exchanges)

    all_rating_differences = numpy.array(all_rating_differences) / team_size
    _squared_and_scaled_differences = numpy.square(all_rating_differences / DRAFTING_RATING_DIFFERENCE_TO_OPTIMAL_TO_BE_TWO_TIMES_UNLIKELY)
    logits = _squared_and_scaled_differences.min() - _squared_and_scaled_differences
    probabilities = numpy.exp2(logits)
    probabilities = probabilities / probabilities.sum()
    distribution_function = numpy.cumsum(probabilities)

    _random_uniform = numpy.random.random()
    optimal_index = logits.argmax()
    selected_index = numpy.argmax(_random_uniform < distribution_function)
    selected_home_exchanges, selected_away_exchanges = [all_exchanges[selected_index] for all_exchanges in [home_all_possible_exchanges, away_all_possible_exchanges]]

    selected_home_team = _exchange_players(initial_home_team, initial_away_team, selected_home_exchanges, selected_away_exchanges)
    selected_away_team = _exchange_players(initial_away_team, initial_home_team, selected_away_exchanges, selected_home_exchanges)
    _absolute_differences = numpy.absolute(all_rating_differences)
    mean_rating_disbalance_to_optimal = _absolute_differences[selected_index] - _absolute_differences[optimal_index]
    selection_probability = probabilities[_squared_and_scaled_differences >= _squared_and_scaled_differences[selected_index]].sum()

    return selected_home_team, selected_away_team, mean_rating_disbalance_to_optimal, selection_probability


def find_current_average_rating_of_team(player_ids, df_players=None):
    if df_players is None:
        df_players = get_players_df()
    return df_players.loc[player_ids, PLAYER_DATABASE_RATING_COLUMN].mean()


def is_search_pattern_precise(search_vector):
    return search_vector.min() <= PATTERN_MATCHING_MAX_DISTANCE

def is_search_vector_exact(search_vector):
    return (PATTERN_MATCHING_SEPARATION_FACTOR_FOR_EXACT_MATCH * search_vector.min() >= search_vector).sum() <= 1


def get_match_from_search_vector(search_vector):
    return search_vector.index[search_vector.argmin()]


def _match_queries_to_player_ids(df_players, queries):
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

def identify_players(identifiers, df_players=None):
    if df_players is None:
        df_players = get_players_df()

    identification_dict = {}
    remaining_identifiers = []
    for identifier in identifiers:
        if identifier.isdigit():
            if int(identifier) in df_players.index:
                player_id = int(identifier)
            else:
                player_id = PLAYER_IDENTIFICATION_NOT_IN_INDEX
        else:
            _search_by_name = df_players.index[df_players[PLAYER_DATABASE_NAME_COLUMN] == identifier]
            if len(_search_by_name) == 1:
                player_id = _search_by_name[0]
            else:
                remaining_identifiers.append(identifier)
                continue
        identification_dict[identifier] = player_id

    pattern_matching_dict = _match_queries_to_player_ids(df_players, remaining_identifiers)
    identification_dict.update(pattern_matching_dict)

    return identification_dict

def remove_players(identifiers, df_players=None):
    if df_players is None:
        df_players = get_players_df()
    identification_dict = identify_players(identifiers, df_players)

    ids_to_remove, invalid_index_identifiers, unrecognized_identifiers, undecided_identifiers = [[] for _ in range(4)]
    for identifier, id in identification_dict.items():
        if id == PLAYER_IDENTIFICATION_NOT_IN_INDEX:
            invalid_index_identifiers.append(identifier)
        elif id == PATTERN_MATCHING_NO_MATCH_STRING:
            unrecognized_identifiers.append(identifier)
        elif id == PATTERN_MATCHING_MULTIPLE_MATCHES:
            undecided_identifiers.append(identifier)
        else:
            ids_to_remove.append(id)

    removed_players = df_players.loc[ids_to_remove]
    if len(removed_players.index) > 0:
        df_players.drop(ids_to_remove, axis=0, inplace=True)
        set_players_df(df_players)
    return removed_players, invalid_index_identifiers, unrecognized_identifiers, undecided_identifiers


def find_essential_matches_mask(df_matches):
    is_home_goals_nan, is_away_goals_nan = [
        df_matches.loc[:, column].isna() for column in(MATCHES_DATABASE_HOME_GOALS_COLUMN, MATCHES_DATABASE_AWAY_GOALS_COLUMN)]

    is_match_essential = ~is_home_goals_nan & ~is_away_goals_nan
    return is_match_essential


def remove_matches(match_ids, df_matches=None, is_remove_essential=False, is_persist_into_database=True):
    if df_matches is None:
        df_matches = get_matches_df()
    assert all([id in df_matches.index for id in match_ids])

    df_removed = df_matches.loc[match_ids].copy()
    if not is_remove_essential:
        assert all(~find_essential_matches_mask(df_removed))

    df_matches.drop(match_ids, inplace=True)
    if is_persist_into_database:
        set_matches_df(df_matches)
    return df_matches, df_removed


def add_player_command(rating, name, nicknames):
    player_record = {
        PLAYER_DATABASE_NAME_COLUMN: name,
        PLAYER_DATABASE_NICKNAMES_COLUMN: [nick.lower() for nick in nicknames],
    }
    if rating > 0:
        player_record[PLAYER_DATABASE_RATING_COLUMN] = round_rating(rating)
    df, df_new = add_players([player_record])
    display_string_to_user("Added the following player")
    display_string_to_user(display_players_df(df_new))


def add_match_command(datetime, args):
    teams_spearator_index = None
    teams_spearator_string = None
    for index in range(len(args)):
        if args[index] in TEAM_SEPARTION_STRINGS:
            if teams_spearator_index is not None:
                display_string_to_user(f"Multiple teams separators. Found {teams_spearator_string} as argument "\
                f"{teams_spearator_index} and {args[index]} as arguments {index}")
                exit(-331)
            else:
                teams_spearator_index = index
    if teams_spearator_index is None:
        display_string_to_user(f"Could not find teams separator. Please seaparte players in the teams with {TEAM_SEPARTION_STRINGS[0]}")
        exit(-1141)
    home_team_args = args[:teams_spearator_index]
    away_team_args = args[teams_spearator_index + 1:]

    df_players = get_players_df()
    identification_dict = identify_players(home_team_args + away_team_args, df_players)
    _multiple_matches = [identifier for identifier, player_id in identification_dict.items() if player_id == PATTERN_MATCHING_MULTIPLE_MATCHES]
    _no_matches = [identifier for identifier, player_id in identification_dict.items() if player_id == PATTERN_MATCHING_NO_MATCH_STRING]
    if len(_no_matches) > 0 or len(_multiple_matches) > 0:
        if len(_no_matches) > 0:
            _plural_suffix = "s" if (len(_no_matches) > 1) else ""
            display_string_to_user(f"Could not match identifier{_plural_suffix} {' '.join(_no_matches)} to player name{_plural_suffix}.")
        if len(_multiple_matches) > 0:
            _plural_suffix = "s" if (len(_multiple_matches) > 1) else ""
            display_string_to_user(f"Found multiple players matching identifier{_plural_suffix} {' '.join(_multiple_matches)}.")
        exit(-6551)

    home_team_ids_set, away_team_ids_set = [
        set(identification_dict[identifier] for identifier in team_identifiers)
        for team_identifiers in (home_team_args, away_team_args)]

    duplicate_ids = [id for id in home_team_ids_set if id in away_team_ids_set]
    if len(duplicate_ids) > 0:
        _plural_suffix = "s" if len(duplicate_ids) > 0 else ""
        display_string_to_user(f"Clash, found player{_plural_suffix} {' '.join([str(id) for id in duplicate_ids])} in both home and away team")
        exit(-192)

    if len(home_team_ids_set) != len(away_team_ids_set):
        display_string_to_user(f"Teams need to have the same number of unique players. Found {len(home_team_ids_set)} unique players for the home team and "\
            f"{len(away_team_ids_set)} for the away team")
        exit(-431)

    match_record = {
        MATCHES_DATABASE_HOME_TEAM_COLUMN: [e for e in home_team_ids_set],
        MATCHES_DATABASE_AWAY_TEAM_COLUMN: [e for e in away_team_ids_set]
    }
    if datetime is not None:
        match_record.update(pandas.to_datetime(datetime))
    df_matches, df_matches_new = add_matches([match_record])
    _plural_suffix = "s" if len(df_matches_new) > 0 else ""
    display_string_to_user(f"Added {len(df_matches_new.index)} match{_plural_suffix} to the database")
    display_string_to_user(f"{df_matches_new.to_markdown()}")


def list_players_command(df_players=None):
    if df_players is None:
        df_players = get_players_df()
    display_string_to_user(display_players_df(df_players))


def list_matches_command(df_matches=None):
    if df_matches is None:
        df_matches = get_matches_df()
    display_matches_df(df_matches)


def list_rating_changes_command(df_rating_changes=None):
    if df_rating_changes is None:
        df_rating_changes = get_rating_changes_df()
    display_string_to_user(df_rating_changes.to_markdown())


def score_match_command(match_id, home_goals, away_goals, df_matches=None, df_players=None):
    if home_goals < 0 or away_goals < 0:
        display_string_to_user("Home and away scores need to be non negative integers")
    if df_matches is None:
        df_matches = get_matches_df()

    if match_id not in df_matches.index:
        display_string_to_user(f"Match with id {match_id} does not exist")
        exit(-190)

    if find_essential_matches_mask(df_matches).loc[match_id]:
        display_string_to_user(f"Match with id {match_id} has already been scored and can't be scored again.")
        return
    else:
        match_record, adjustments = score_match(match_id, home_goals, away_goals, df_matches, df_players)
        display_string_to_user(f"Updates score for match {match_id}: {match_record[MATCHES_DATABASE_HOME_GOALS_COLUMN]}-{match_record[MATCHES_DATABASE_AWAY_GOALS_COLUMN]}.")
        display_string_to_user(f"Rating changes:\n{adjustments}")


def draft_command(player_identifiers, is_discard_match):
    if len(player_identifiers) == 0:
        click.echo("Please specify available players in arguments. These players will be separated into two teams of approximatly equal rating.")
        exit(-9184)
    if len(player_identifiers) % 2 == 1:
        click.echo("The number of players needs to be even so that they can be split into two teams.")
        exit(-331)
    df_players = get_players_df()
    identifiers_dict = identify_players(player_identifiers, df_players)
    for identifier, player_id in identifiers_dict.items():
        assert player_id in df_players.index

    home_team_ids, away_team_ids, rating_disbalance_to_optimal, selection_probability = find_teams_separation_with_small_ratings_difference(identifiers_dict.values(), df_players)
    suggested_match = {
        MATCHES_DATABASE_HOME_TEAM_COLUMN: builtins.list(home_team_ids),
        MATCHES_DATABASE_AWAY_TEAM_COLUMN: builtins.list(away_team_ids)
        }

    _, df_new_match = add_matches([suggested_match], persist_into_databse=(not is_discard_match))
    match_stored_description = "The match below has not been stored because '--discard' flag was used." if is_discard_match \
        else "The following match has been added. You can score this match using the 'score' command which will update the rating" \
            " of all players involved."

    home_mean_rating, away_mean_rating = [find_current_average_rating_of_team(team_ids) for team_ids in [home_team_ids, away_team_ids]]
    click.echo(f"Separated players into two teams with approximate same ratings, {home_mean_rating:.1f} vs. {away_mean_rating:.1f}."
        f"This division is {rating_disbalance_to_optimal:.1f} rating units from optimal. The probability of selecting equal or worse divisions"
        f" was {selection_probability * 100:.1f}%.\n"
        f"{match_stored_description}")
    click.echo(df_new_match.to_markdown())


def remove_players_command(identifiers, df_players=None):
    removed_players, invalid_index_identifiers, unrecognized_identifiers, undecided_identifiers = remove_players(identifiers, df_players)

    nr_removed = len(removed_players.index)
    if nr_removed > 0:
        display_string_to_user(f"Removed {nr_removed} player{'s' if nr_removed > 1 else ''}:")
        display_string_to_user(f"{display_players_df(removed_players)}")

    if len(invalid_index_identifiers) > 0:
        display_string_to_user(f"No players with ids {[int(index_identifier) for index_identifier in invalid_index_identifiers]} in the database")
    if len(unrecognized_identifiers) > 0:
        display_string_to_user(f"Could not match identifiers {unrecognized_identifiers}")
    if len(undecided_identifiers) > 0:
        display_string_to_user(f"Multiple players matched identifiers {undecided_identifiers}. Please be more specific or match players by name or id instead")


def remove_matches_command(match_ids, is_ignore_warnings=False, df_matches=None):
    if df_matches is None:
        df_matches = get_matches_df()

    ids_to_remove_set = set()
    indices_not_parsed = []
    for id in match_ids:
        if id == REMOVE_NONESSENTIAL_MATCHES_STRING:
            is_essential = find_essential_matches_mask(df_matches)
            ids_to_remove_set.update(df_matches.index[~is_essential])
        elif str.isdigit(id):
            ids_to_remove_set.add(int(id))
        else:
            indices_not_parsed.append(id)
    if len(indices_not_parsed) > 0:
        display_string_to_user(f"Could not parse {indices_not_parsed} as match indices. Please use integers or '{REMOVE_NONESSENTIAL_MATCHES_STRING}'")
        exit(-1515)
    match_ids = [id for id in ids_to_remove_set]

    non_existing_ids, existing_ids = [[] for _ in range(2)]
    for id in match_ids:
        if id in df_matches.index:
            existing_ids.append(id)
        else:
            non_existing_ids.append(id)
    if non_existing_ids:
        display_string_to_user(f"Indices {non_existing_ids} do not exist in the index.")

    df_to_remove = df_matches.loc[existing_ids].copy()
    is_essential = find_essential_matches_mask(df_to_remove)
    essential_indices = df_to_remove.index[is_essential]
    non_essential_indices = df_to_remove.index[~is_essential]

    ids_to_remove = [id for id in non_essential_indices]
    if not is_ignore_warnings and len(essential_indices):
        _plural_suffix = "es" if len(essential_indices) > 2 else ""
        display_string_to_user(f"Removing following match{_plural_suffix} will damage the consistency of the database (it will not be possible to recreate all the data).")
        display_string_to_user(f"{df_to_remove.loc[essential_indices, :].to_markdown()}")
        if click.confirm("Are you sure you want to delete them?"):
            display_string_to_user(f"Removed {len(essential_indices)} matches.")
            ids_to_remove.extend(essential_indices)

    if len(ids_to_remove) == 0:
        display_string_to_user(f"No matches removed.")
    else:
        df_matches, df_removed = remove_matches(ids_to_remove, df_matches, is_remove_essential=True, is_persist_into_database=True)
        display_string_to_user(f"Removed {len(df_removed.index)} matches.")
        display_string_to_user(f"{df_removed.to_markdown()}")

    return df_matches, df_to_remove


def show_ratings():
    df_players = get_players_df()
    display_string_to_user(display_players_df(df_players, True, is_display_nicknames=False))


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
def player(rating, name, nicknames):
    '''Creates a new player. Takes player's name as first argument and treats other arguments as player's nicknames.'''
    add_player_command(rating, name, nicknames)

@add.command(help=
f'''Adds a match to the database. Takes names or nicknames of players as arguments.
Teams need to be separated by any of the following {TEAM_SEPARTION_STRINGS}.''')
@click.option("--date", "--datetime", "-d", "datetime", type=str, default=None)
@click.argument("args", nargs=-1)
def match(datetime, args):
    add_match_command(datetime, args)


@rankings.group()
def remove():
    '''Removes data from the database.'''
    pass


@remove.command()
@click.argument("identifiers", nargs=-1)
def players(identifiers):
    '''Removes players from the database. Takes a list of identifiers which are matched with players' ids, '''\
    '''names and nicknames to find corresponding players.'''
    remove_players_command(identifiers)


@remove.command(help=
    f'''Removes matches specified by ids given through arguments from the database.
    Use '{REMOVE_NONESSENTIAL_MATCHES_STRING}' instead of ids to remove all matches that have not been scored.'''
)
@click.option("--ignore_warnings", "-i", "--ignore", "is_ignore_warnings", type=bool, default=False)
@click.argument("match_ids", nargs=-1)
def matches(match_ids, is_ignore_warnings):
    remove_matches_command(match_ids, is_ignore_warnings)

@rankings.group(name="list")
def list():
    '''Lists data from the database.'''
    pass


@list.command()
@click.option("--rating/--no-rating", default=False)
def players(rating):
    '''Lists all active players.'''
    list_players_command()


@list.command()
def matches():
    '''Lists all matches.'''
    list_matches_command()


@list.command()
def history():
    '''Lists all cahnges in ratings.'''
    list_rating_changes_command()


@rankings.group(help='''Updates database.''')
def update():
    display_string_to_user("Not implemented!")


@update.command(help='''Updates player data.''')
@click.option("--player-id", "-i", "-id", "--id", type=int, required=True)
@click.option("--column", "-c", type=click.Choice(PLAYER_DATABASE_NON_INDEX_COLUMNS, case_sensitive=False))
@click.argument("new_value", nargs=1)
def players(player_id, column_name, new_value):
    display_string_to_user("Not implemented!")


@update.command(help='''Updates match data.''')
@click.option("--match-id", "-i", "-id", "--id", type=int, required=True)
@click.option("--column", "-c", type=click.Choice(MATCHES_DATABASE_NON_INDEX_COLUMNS, case_sensitive=False))
@click.argument("new_value", nargs=1)
def match(match_id, column_name, new_value):
    display_string_to_user("Not implemented!")


@rankings.command()
@click.option("--discard", "is_discard_match", is_flag=True, default=False, help="generated match will not be stored")
@click.argument("player_identifiers", nargs=-1)
def draft(player_identifiers, is_discard_match):
    '''Separates players into two teams of approximate same rating. Takes names of players as arguments.'''
    draft_command(player_identifiers, is_discard_match)


@rankings.command()
@click.option("--match_id", "-id", type=int, default=-1)
@click.argument("home_score", nargs=1, type=int)
@click.argument("away_score", nargs=1, type=int)
def score(match_id, home_score, away_score):
    '''Adds score to one of the non-scored matches and updates rankings of all players participating. '''\
    '''Takes the score for first ("home") and second ("away") team as arguments.'''
    score_match_command(match_id, home_score, away_score)


@rankings.command()
def ratings():
    '''Show ratings leaderboard'''
    show_ratings()


if __name__ == "__main__":
    rankings()
