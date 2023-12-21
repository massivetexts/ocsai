import pingouin as pg
import hashlib
import numpy as np
from pathlib import Path
from ..utils import mprint


def simple_stats(df, rater_cols=False):
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
    print("# of prompts", len(df.prompt.unique()))
    print("# of participants", len(df.participant.unique()))
    print("# of data points", len(df))
    print("Prompts", df.prompt.unique())
    if rater_cols:
        print("# of raters", len(rater_cols))
        print("Intraclass correlation coefficients (report ICC2k)")
        x = df[['id']+rater_cols].melt(id_vars='id', var_name='rater',
                                       value_name='rating')
        icc = pg.intraclass_corr(data=x, targets='id', raters='rater',
                                 ratings='rating', nan_policy='omit')
        print(icc.round(2))


def prep_general(data, name, rater_cols=None, test_type='uses',
                 language='eng', range=(1,5), return_full=False,
                 aggregate_scores=True, drop_noresponse=True,
                 print_stats=True, round_adjust=1,
                 null_marker=None, column_mappings=None,
                 replace_values=None,
                 include_rater_std=False, overwrite_q=False,
                 meta=None, save_dir=None):
    '''General cleaning that repeats for multiple datasets.

    data: DataFrame with the following columns:
        - type (optional) - type of test. Assumes the value of 
            the `test_type` parameter if there's no colum (default 
            'uses'). The values can be anything, but stay consistent.
            In Ocsai training, we use: ['uses', 'instances',
            'consequences', 'completion']
        - participant - an identifier for the participant
        - prompt - a 'short' version of the prompt. For example:
            for a uses question like "What is a surprising use for 
            a banana?", the prompt would be "banana"
        - response - the participant's answer
        - (optional) rater columns, describing individual rater scores
        - question (optional) - the full human-readable question, 
            including the prompt. Optional but recommended, because
            automatically inferred questions are limits to type=uses.
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
        codes (https://www.loc.gov/standards/iso639-2/php/code_list.php)
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
        done *after* column renaming, so use the new column names.

    include_rater_std: If True, include a column for the standard deviation of
        rater scores. This can help identify items that are more difficult to
        rate by humans.

    overwrite_q: If True, overwrite the question column with a default
        question. This is useful if you have a custom question format that
        isn't captured by the default question format.

    meta: A dictionary of metadata. This is only used for logging.

    save_dir: If provided, save the data to this directory, as {name}.csv.
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
        print("Renaming columns", column_mappings)
        data = data.rename(columns=column_mappings)

    if replace_values:
        for col, values in replace_values.items():
            assert col in data.columns, f"Column {col} not found for replacement"
            data[col] = data[col].replace(values)

    if null_marker:
        print(f"Replacing {null_marker} with NaN in response column")
        data.response = data.response.replace(null_marker, np.nan)
        for col in rater_cols:
            data[col] = data[col].replace(null_marker, np.nan)

    if 'language' not in data.columns:
        data['language'] = language
    
    print("Rater cols:", rater_cols)

    if drop_noresponse:
        data = data[~data.response.isna()]

    data['src'] = name
    data['avg_rating'] = data[rater_cols].mean(1)

    if round_adjust:
        # add a tiny bit to avg rater to round in direction of median. Only
        # need if there are tiebreakers
        data['median_rating'] = data[rater_cols].median(1)
        data['avg_rating'] = (data.avg_rating +
                              (data.median_rating - data.avg_rating).div(10**3)
                              )
    if include_rater_std:
        data['rating_std'] = data[rater_cols].std(1)

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
        data.loc[uses, 'question'] = data.loc[uses, 'prompt'].apply(
            lambda x: f"What is a surprising use for a {x.upper()}?")

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
        simple_stats(data, rater_cols)

    if return_full:
        final = data
    else:
        cols = ['type', 'src', 'question', 'prompt', 'response', 'id',
                'target', 'participant', 'response_num']
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
                      "(you set:{}; data:{})."
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