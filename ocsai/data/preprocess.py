import pingouin as pg
import hashlib
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from ..utils import mprint, can_render_md_html


def infer_question(prompt, language='eng'):
    '''A very basic way to infer the question from the prompt. Generally,
    the non-English versions are chatbot-written, so please send fixes
    to peter.organisciak@du.edu! Some bad grammar is likely fine, though may have
    small benefits.
    '''
    phrases = {
        'ara': lambda y: f"ما هو استخدام مفاجئ لـ {y}؟",  # Arabic
        'chi': lambda y: f"什么是{y}的一个令人惊讶的用途？",  # Chinese
        'dut': lambda y: f"Wat is een verrassend gebruik voor een {y}?",  # Dutch
        'eng': lambda y: f"What is a surprising use for {y}?",  # English
        'fre': lambda y: f"Quel est un usage surprenant pour un {y}?",  # French
        'ger': lambda y: f"Was ist eine überraschende Verwendung für ein {y}?",  # German
        'heb': lambda y: f"מהו שימוש מפתיע ל{y}?",  # Hebrew
        'ita': lambda y: f"Qual è un uso sorprendente per un {y}?",  # Italian
        'pol': lambda y: f"Jakie jest zaskakujące zastosowanie dla {y}?",  # Polish
        'rus': lambda y: f"Какое удивительное применение для {y}?",  # Russian
        'spa': lambda y: f"¿Cuál es un uso sorprendente para un {y}?"  # Spanish
    }

    # Get the phrase in the requested language, or fall back to English
    phrase = phrases.get(language, phrases['eng'])

    # Apply the phrase to the object
    return phrase(prompt.upper())

                
def simple_stats(df, rater_cols=False, name=None, save_dir=None):
    '''
    Prints simple stats about the data. Expects a DataFrame with the following
    columns:
    - prompt
    - participant
    - id
    - rating
    - (optional) rater columns, describing individual rater scores

    If rater_cols is provided, will also print ICC2k scores for each rater.
    '''
    stats_data = {
        "name": name,
        "no_of_prompts": len(df.prompt.unique()),
        "no_of_participants": len(df.participant.unique()),
        "no_of_data_points": len(df),
        "prompts": df.prompt.unique().tolist(),
        "ICC2k": None,
        "ICC2k_CI": None,
        "ICC3k": None,
        "rater_cols": None,
        "no_of_raters": None
    }
    
    if rater_cols:
        stats_data["rater_cols"] = rater_cols
        stats_data["no_of_raters"] = len(rater_cols)

    if rater_cols and len(rater_cols) > 1:
        try:
            x = df[['id']+rater_cols].melt(id_vars='id', var_name='rater',
                                           value_name='rating')
            icc = pg.intraclass_corr(data=x, targets='id', raters='rater',
                                     ratings='rating', nan_policy='omit')
            stats_data["ICC2k"] = icc[icc.Type == 'ICC2k'].ICC.round(2).values[0]
            stats_data["ICC2k_CI"] = icc[icc.Type == 'ICC2k']['CI95%'].values[0]
            stats_data["ICC2k_CI"] = str(stats_data["ICC2k_CI"][0]) + '-' + str(stats_data["ICC2k_CI"][1])
            stats_data['ICC3k'] = icc[icc.Type == 'ICC3k'].ICC.round(2).values[0]
        except AssertionError:
            mprint("WARNING: ICC has an undefined error for this dataset")

    # print stats
    print_str = ""
    for key, value in stats_data.items():
        print_str += f"- {key}: {value}\n"
    mprint(print_str)

    # Saving to DuckDB
    if save_dir:
        assert name, "Name must be provided to save stats"
        conn = duckdb.connect(f"{save_dir}/stats_db.duckdb")
        stats_df = pd.DataFrame([stats_data])
        # Create table if not exists
        conn.sql("CREATE TABLE IF NOT EXISTS stats (name TEXT, no_of_prompts INTEGER, "
                 "no_of_participants INTEGER, no_of_data_points INTEGER, "
                 "prompts TEXT, ICC2k REAL, ICC2k_CI TEXT, ICC3k REAL, "
                 "rater_cols TEXT, no_of_raters INTEGER)")
        
        # Checking if the row exists
        existing_rows = conn.sql(f"SELECT * FROM stats WHERE name = '{name}'").fetchall()
        if existing_rows:
            conn.sql(f"DELETE FROM stats WHERE name = '{name}'")
        stats_df.to_sql('stats', conn, if_exists='append', index=False)

        conn.close()


def prep_general(data, name, rater_cols=None, test_type='uses',
                 language='eng', range=None, return_full=False,
                 aggregate_scores=True, drop_noresponse=True,
                 print_stats=True, round_adjust=1,
                 null_marker=None, column_mappings=None,
                 replace_values=None,
                 question_mappings=None,
                 type_mappings=None,
                 include_rater_std=True, overwrite_q=False,
                 meta=None, save_dir=None):
    '''General cleaning that repeats for multiple datasets.

    data: DataFrame with the following columns:
        - type (optional) - type of test. Assumes the value of 
            the `test_type` parameter if there's no column (default 
            'uses'). The values can be anything, but stay consistent.
            In Ocsai training, we use: ['uses', 'instances',
            'consequences', 'completion', 'metaphor']
        - participant - an identifier for the participant
        - prompt - a 'short' version of the prompt. For example:
            for a uses question like "What is a surprising use for 
            a banana?", the prompt would be "banana"
        - response - the participant's answer
        - (optional) rater columns, describing individual rater scores
        - question (optional) - the full human-readable question, 
            including the prompt. Optional but recommended, because
            automatically inferred questions are limited to type=uses.
        - [rater columns] - columns describing individual human judges' scores.
            List the names of the columns with the rater_cols argument, else
            it is inferred.
        - response_num (optional) - a number for the participant's response order.
            This isn't used in training, but is kept in the data when possible.
        - language (optional) - the language of the data. Use ISO 639-2 Codes.
            If not provided as a column, this will use the language parameter,
            which defaults to eng.

        Some datasets may have differently named columns (e.g. `subject` instead
        of `participant`). If so, use the column_mappings parameter to map the
        column names (e.g. {'subject':'participant'}).

    name: A name string describing the source of the data. This will be used to
        identify the data in the final dataset.

    rater_cols: A list of column names that describe individual rater scores.

    language: A string describing the language of the data. Use the ISO-639-2
        codes (https://www.loc.gov/standards/iso639-2/php/code_list.php). This
        parameter is superseced by the `language` column in data, if provided.
        [default: eng]

    range: A tuple of the min and max of the input data's range. If not provided,
        this will use the min and max of the data.

    return_full: If True, return the full DataFrame, rather than just the
        columns needed for training.

    aggregate_scores: If True, aggregate scores across raters. If False, this
        will (eventually) return original individual rater scores, rather than
        multi-rater. I haven't implemented this disagregation yet because I'm
        not sure how well disagregated scored could be used in a proper
        training, without data leakage.

    drop_noresponse: If True, drop rows where response is NaN.

    print_stats: If True, print simple stats about the data.

    round_adjust: If True, add a tiny bit to avg rater to round in direction of
        the median. Only need if there are tiebreakers. This is different from
        the round_precision parameter in normalize_values, which rounds the
        scores in a *random* direction.

    null_marker: If provided, replace this value with NaN.

    column_mappings: A dictionary of column mappings. This is useful if the
        column names in the data don't match the expected column names.
    
    replace_values: A dictionary of values to replace, with the key as the columns
        name, and the value as a dictionary of old:new replacement values. This is
        done *after* column renaming, so use the new column names, but done *before*
        question_mappings and type_mappings.

    question_mappings: A dictionary of prompt-to-question mappings.

    type_mappings: A dictionary of prompt-to-type mappings.

    include_rater_std: If True, include a column for the standard deviation of
        rater scores. This can help identify items that are more difficult to
        rate by humans.

    overwrite_q: If True, overwrite the question column with a default
        question. This is useful if you have a custom question format that
        isn't captured by the default question format.

    meta: A dictionary of metadata. This is only used for logging.

    save_dir: If provided, save the data to this directory, as {name}.csv, 
        and stats as stats_db.duckdb
    '''
    if meta:
        if meta.get('inline'):
            mprint(f'### Loading *{meta["inline"]}*')
        if meta.get('citation'):
            mprint(meta['citation'])

    if not aggregate_scores:
        raise NotImplementedError("disagregated scores not yet supported")

    if not rater_cols:
        # assume columns that say 'rater'
        rater_cols = [col for col in data.columns if 'rater' in col.lower()]

    if column_mappings:
        mprint("- Renaming columns", column_mappings)
        data = data.rename(columns=column_mappings)

    if replace_values:
        for col, values in replace_values.items():
            assert col in data.columns, f"Column {col} not found for replacement"
            data[col] = data[col].replace(values)

    if question_mappings:
        mprint("- Inferring questions", question_mappings)
        data['question'] = data.prompt.replace(question_mappings)

        assert set(question_mappings.keys()) == set(data.prompt.unique()), "There are prompts that don't have a question mapping"

    if type_mappings:
        mprint("- Inferring types", type_mappings)
        data['type'] = data.prompt.replace(type_mappings)

    if null_marker:
        print(f"Replacing {null_marker} with NaN in response column")
        data.response = data.response.replace(null_marker, np.nan)
        for col in rater_cols:
            data[col] = data[col].replace(null_marker, np.nan)

    if 'language' not in data.columns:
        data['language'] = language

    if drop_noresponse:
        data = data[~data.response.isna()]

    data['src'] = name

    # coerce rater cols to numeric
    data[rater_cols] = data[rater_cols].apply(pd.to_numeric, errors='coerce')
    data['avg_rating'] = data[rater_cols].mean(1)
    data['rater_count'] = data[rater_cols].notna().sum(1)

    if round_adjust:
        # add a tiny bit to avg rater to round in direction of median. Only
        # need if there are tiebreakers
        data['median_rating'] = data[rater_cols].median(1)
        data['avg_rating'] = (data.avg_rating +
                              (data.median_rating - data.avg_rating).div(10**3)
                              )
    if include_rater_std:
        data['rating_std'] = data[rater_cols].std(1)

    # while normalize_values infers based on the average, we probably want to
    # make that inference here based on the mix/max across the original ratings
    if not range:
        range = (data[rater_cols].min().min(), data[rater_cols].max().max())
        mprint("- Inferred range of original data:", range)
    data['target'] = normalize_values(data.avg_rating, oldrange=range)

    missing = data.target.isna()
    if missing.sum():
        print(f'Dropping {missing.sum()} unrated items')
        data = data[~missing]

    data['participant'] = name + data['participant'].astype(str)

    if 'type' not in data.columns:
        data['type'] = test_type

    if ('question' not in data.columns) or overwrite_q:
        try:
            assert data.type.unique().tolist() == ['uses']
        except AssertionError:
            raise Exception("can't infer question format for anything other "
                            "than uses; please explicitly supply a 'question' "
                            "column")
        uses = data.type == 'uses'

        def q_from_prompt(x):
            return infer_question(x['prompt'], x['language'])
        data.loc[uses, 'question'] = data[uses].apply(q_from_prompt, axis=1)

    def hashfunc(x):
        return hashlib.md5(x.encode('utf-8')).hexdigest()[:6]
    idhash = (data.participant+data.response).apply(hashfunc)
    # task can be a custom string that is used instead of prompt
    # (e.g. 'g2_red' instead of 'red')
    col = 'task' if 'task' in data.columns else 'prompt'
    data['id'] = f'{name}_' + data[col].astype(str) + '-' + idhash

    if 'response_num' not in data.columns:
        data['response_num'] = None
    if print_stats:
        simple_stats(data, rater_cols, save_dir=save_dir, name=name)

    if return_full:
        final = data
    else:
        cols = ['type', 'src', 'question', 'prompt', 'response', 'id',
                'target', 'participant', 'response_num', 'language',
                'rater_count']
        if include_rater_std:
            cols += ['rating_std']
        final = data[cols]

    if save_dir:
        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        filename = save_dir / f'{name}.csv'
        final.to_csv(filename, index=False)
    return final


def combine_dupes(df):
    '''
    Combine all the rows of an input dataframe. 
    Used for grouping duplicates. Expects the input DataFrame
    to have the following columns:
    - src: name for the source of the row
    - response: the response text. Should be the 'same', approximately, 
        but if using some other strategy for fuzzing matching, does not
        need to be exact.
    - target: the target score
    - id: a unique identifier for the row
    - participant: the participant identifier
    

    '''
    # start with a dict of the first row
    outdict = df.iloc[0].to_dict()

    # assertions:
    if 'prompt' in df.columns:
        assert len(df.prompt.unique()) == 1, "Should only be one prompt"
    if 'type' in df.columns:
        assert len(df.type.unique()) == 1, "Should only be one type"
    if 'language' in df.columns:
        assert len(df.language.unique()) == 1, "Should only be one language"

    # If there is only one row, keep it intact
    if len(df) == 1:
        outdict['participant_list'] = [outdict['participant']]
        return pd.Series(outdict)
    else:
        # sort and concatenate all the unique src values
        src = df.src.unique()
        src = '/'.join(sorted(src))
        outdict['src'] = src

        # take the most common response from the group
        outdict['response'] = df.response.value_counts().index[0]

        # id: sort and concatenate all the unique id values and make into a unique hash
        id = sorted(df.id.unique())
        id = ''.join(id)
        id = hashlib.md5(id.encode()).hexdigest()
        outdict['id'] = id

        # average of target, weighted by rater_count
        outdict['target'] = np.average(df.target, weights=df.rater_count)
        outdict['participant'] = f'COMBINED_{id}'
        outdict['participant_list'] = df.participant.unique().tolist()
        outdict['response_num'] = np.nan
        if 'rater_count' in outdict:
            outdict['rater_count'] = df.rater_count.sum()
        else:
            outdict['rater_count'] = len(df)

        # Calculate combined standard deviation
        # If rating_std is known for all values, calculate an approximation based on sum of squares
        # (assuming the groups are independent and normally distributed)
        # If rating_std is not known for all values, return a basic std based on the current rows.
        if (('rating_std' in df.columns) 
           and ('rater_count' in df.columns)
           and (df.rating_std.isna().sum() == 0)):
            sum_of_squares = df.apply(lambda row: (row.rater_count - 1) * row.rating_std**2, axis=1).sum()
            outdict['rating_std'] = np.sqrt(sum_of_squares / (outdict['rater_count'] - len(df)))
        else:
            outdict['rating_std'] = df.target.std()
        return pd.Series(outdict)


def fingerprint_series(s, basic=False):
    ''' Fingerprint responses for fuzzy deduplication.
    Does *not* work well for chinese - there, run basic=True.

    Follow the OpenRefine Strategy:

    'remove leading and trailing whitespace
    change all characters to their lowercase representation
    remove all punctuation and control characters
    normalize extended western characters to their ASCII representation (for example "gödel" → "godel")
    split the string into whitespace-separated tokens
    sort the tokens and remove duplicates
    join the tokens back together'

    https://openrefine.org/docs/technical-reference/clustering-in-depth
    '''
    import re
    import unicodedata
    from unidecode import unidecode
    s = s.str.lower()
    s = s.str.replace(r'[^\w\s]', '', regex=True)
    if basic:
        return s
    s = s.apply(lambda x: unidecode(x))
    s = s.apply(lambda x: ' '.join(sorted(x.split())))
    return s


def normalize_values(series, outrange=(1, 5), oldrange=None,
                     round_precision=False):
    '''
    Normalize the range of score values to a standard scale.

    outrange: tuple of min and max of the new range. Default is 1-5.

    oldrange: tuple of min and max of the old range. If not provided, this
    will use the min and max of the data.

    round_precision: integer of the rounding precision. If False, no rounding
        will occur.
    '''
    min, max = outrange
    if oldrange:
        oldmin, oldmax = oldrange
        # doublecheck
        warning_msg =("!!\n\nWARNING: DATA GOES {} THAN EXPECTED"
                      "(you set {}; data: {})."
                      "Ensure that you set the range you wanted.\n\n!!")
        if oldmin > series.min():
            print(warning_msg.format('LOWER', oldmin, series.min()))
        if oldmax < series.max():
            print(warning_msg.format("HIGHER", oldmax, series.max()))
    else:
        oldmin, oldmax = series.min(), series.max()
        print("Inferred range of original data:", oldmin, oldmax)

    x = (series - oldmin)/(oldmax-oldmin)
    x = min + (max-min)*x
    if round_precision:
        # none of our data has even numbers of raters, so there shouldn't be
        # need for rounding tiebreakers but for future proofing, add or
        # subtract a tiny number randomly, so that rounding can go either way
        modifier = (2*np.random.randint(0, 2, size=x.shape)-1)/10**5
        x = np.round(x + modifier, round_precision)
    return x