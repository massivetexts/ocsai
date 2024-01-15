import pingouin as pg
import hashlib
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from ..utils import mprint, can_render_md_html

# pre-defined question mappings
question_inference_mapping = {
    'ara': {'uses': {
        'علب الصفيح': 'ما هو الاستخدام المفاجئ لـ علب الصفيح؟'}},
    'chi': {'uses': {
        '光盘': '光盘有什么令人惊讶的用途？',
        '冰块': '冰块有什么令人惊讶的用途？',
        '勺子': '勺子有什么令人惊讶的用途？',
        '南瓜': '南瓜有什么令人惊讶的用途？',
        '卫生纸': '卫生纸有什么令人惊讶的用途？',
        '发带': '发带有什么令人惊讶的用途？',
        '发簪': '发簪有什么令人惊讶的用途？',
        '台灯': '台灯有什么令人惊讶的用途？',
        '吸管': '吸管有什么令人惊讶的用途？',
        '吹风机': '吹风机有什么令人惊讶的用途？',
        '咖啡': '咖啡有什么令人惊讶的用途？',
        '喇叭': '喇叭有什么令人惊讶的用途？',
        '围巾': '围巾有什么令人惊讶的用途？',
        '图钉': '图钉有什么令人惊讶的用途？',
        '土豆': '土豆有什么令人惊讶的用途？',
        '地图': '地图有什么令人惊讶的用途？',
        '塑料袋': '塑料袋有什么令人惊讶的用途？',
        '墨水': '墨水有什么令人惊讶的用途？',
        '头发': '头发有什么令人惊讶的用途？',
        '夹子': '夹子有什么令人惊讶的用途？',
        '字典': '字典的一个令人惊讶的用途是什么？',
        '小米': '小米的一个令人惊讶的用途是什么？',
        '床单': '床单的一个令人惊讶的用途是什么？',
        '弹弓': '弹弓的一个令人惊讶的用途是什么？',
        '戒指': '戒指的一个令人惊讶的用途是什么？',
        '扇子': '扇子的一个令人惊讶的用途是什么？',
        '手套': '手套的一个令人惊讶的用途是什么？',
        '扑克': '扑克的一个令人惊讶的用途是什么？',
        '报纸': '报纸的一个令人惊讶的用途是什么？',
        '拖鞋': '拖鞋的一个令人惊讶的用途是什么？',
        '擀面杖': '擀面杖的一个令人惊讶的用途是什么？',
        '无花果': '无花果的一个令人惊讶的用途是什么？',
        '易拉罐': '易拉罐的一个令人惊讶的用途是什么？',
        '曲别针': '曲别针的一个令人惊讶的用途是什么？',
        '木头': '木头的一个令人惊讶的用途是什么？',
        '杯子': '杯子的一个令人惊讶的用途是什么？',
        '柳条': '柳条的一个令人惊讶的用途是什么？',
        '柳树': '柳树的一个令人惊讶的用途是什么？',
        '柿子': '柿子的一个令人惊讶的用途是什么？',
        '核桃': '核桃的一个令人惊讶的用途是什么？',
        '梳子': '梳子的一个令人惊讶的用途是什么？',
        '棉签': '棉签的一个令人惊讶的用途是什么？',
        '椰子': '椰子的一个令人惊讶的用途是什么？',
        '橄榄油': '橄榄油的一个令人惊讶的用途是什么？',
        '橡皮擦': '橡皮擦的一个令人惊讶的用途是什么？',
        '毛巾': '毛巾的一个令人惊讶的用途是什么？',
        '毛笔': '毛笔的一个令人惊讶的用途是什么？',
        '气球': '气球的一个令人惊讶的用途是什么？',
        '水壶': '水壶的一个令人惊讶的用途是什么？',
        '浴缸': '浴缸的一个令人惊讶的用途是什么？',
        '温度计': '温度计的一个令人惊讶的用途是什么？',
        '漏斗': '漏斗的一个令人惊讶的用途是什么？',
        '灌木': '灌木的一个令人惊讶的用途是什么？',
        '火柴': '火柴的一个令人惊讶的用途是什么？',
        '牙刷': '牙刷的一个令人惊讶的用途是什么？',
        '牙签': '牙签的一个令人惊讶的用途是什么？',
        '牙膏': '牙膏的一个令人惊讶的用途是什么？',
        '狐狸': '狐狸的一个令人惊讶的用途是什么？',
        '玉米': '玉米的一个令人惊讶的用途是什么？',
        '球拍': '球拍的一个令人惊讶的用途是什么？',
        '生姜': '生姜有什么令人惊讶的用途？',
        '画像': '画像有什么令人惊讶的用途？',
        '白纸': '白纸有什么令人惊讶的用途？',
        '白酒': '白酒有什么令人惊讶的用途？',
        '皮带': '皮带有什么令人惊讶的用途？',
        '皮筋': '皮筋有什么令人惊讶的用途？',
        '盘子': '盘子有什么令人惊讶的用途？',
        '相机': '相机有什么令人惊讶的用途？',
        '砖头': '砖头有什么令人惊讶的用途？',
        '硬币': '硬币有什么令人惊讶的用途？',
        '磁铁': '磁铁有什么令人惊讶的用途？',
        '积木': '积木有什么令人惊讶的用途？',
        '窗帘': '窗帘有什么令人惊讶的用途？',
        '笛子': '笛子有什么令人惊讶的用途？',
        '筷子': '筷子有什么令人惊讶的用途？',
        '算盘': '算盘有什么令人惊讶的用途？',
        '红酒': '红酒有什么令人惊讶的用途？',
        '纸巾': '纸巾有什么令人惊讶的用途？',
        '纸杯': '纸杯有什么令人惊讶的用途？',
        '纸盒': '纸盒有什么令人惊讶的用途？',
        '纽扣': '纽扣的一个令人惊讶的用途是什么？',
        '耳机': '耳机的一个令人惊讶的用途是什么？',
        '耳机线': '耳机线的一个令人惊讶的用途是什么？',
        '胶水': '胶水的一个令人惊讶的用途是什么？',
        '船桨': '船桨的一个令人惊讶的用途是什么？',
        '芦荟': '芦荟的一个令人惊讶的用途是什么？',
        '花椒': '花椒的一个令人惊讶的用途是什么？',
        '花瓣': '花瓣的一个令人惊讶的用途是什么？',
        '花生': '花生的一个令人惊讶的用途是什么？',
        '茶壶': '茶壶的一个令人惊讶的用途是什么？',
        '荷叶': '荷叶的一个令人惊讶的用途是什么？',
        '蚊帐': '蚊帐的一个令人惊讶的用途是什么？',
        '蛋壳': '蛋壳的一个令人惊讶的用途是什么？',
        '蛋清': '蛋清的一个令人惊讶的用途是什么？',
        '蛋糕': '蛋糕的一个令人惊讶的用途是什么？',
        '蜡烛': '蜡烛的一个令人惊讶的用途是什么？',
        '衣架': '衣架的一个令人惊讶的用途是什么？',
        '袜子': '袜子的一个令人惊讶的用途是什么？',
        '西瓜': '西瓜的一个令人惊讶的用途是什么？',
        '西瓜皮': '西瓜皮的一个令人惊讶的用途是什么？',
        '西红柿': '西红柿的一个令人惊讶的用途是什么？',
        '贝壳': '贝壳的一个令人惊讶的用途是什么？',
        '轮胎': '轮胎的一个令人惊讶的用途是什么？',
        '酒瓶': '酒瓶的一个令人惊讶的用途是什么？',
        '酸奶': '酸奶的一个令人惊讶的用途是什么？',
        '钉子': '钉子的一个令人惊讶的用途是什么？',
        '钥匙': '钥匙的一个令人惊讶的用途是什么？',
        '钳子': '钳子的一个令人惊讶的用途是什么？',
        '铁链': '铁链的一个令人惊讶的用途是什么？',
        '铃铛': '铃铛的一个令人惊讶的用途是什么？',
        '铅笔': '铅笔的一个令人惊讶的用途是什么？',
        '银行卡': '银行卡的一个令人惊讶的用途是什么？',
        '锅': '锅的一个令人惊讶的用途是什么？',
        '镊子': '镊子的一个令人惊讶的用途是什么？',
        '面团': '面团的一个令人惊讶的用途是什么？',
        '靴子': '靴子的一个令人惊讶的用途是什么？',
        '鞋带': '鞋带的一个令人惊讶的用途是什么？',
        '韭菜': '韭菜的一个令人惊讶的用途是什么？',
        '音响': '音响的一个令人惊讶的用途是什么？',
        '领带': '领带的一个令人惊讶的用途是什么？',
        '风车': '风车的一个令人惊讶的用途是什么？',
        '香蕉': '香蕉的一个令人惊讶的用途是什么？',
        '马来貘': '马来貘的一个令人惊讶的用途是什么？',
        '鹅卵石': '鹅卵石的一个令人惊讶的用途是什么？',
        '黄金': '黄金的一个令人惊讶的用途是什么？'}},
    'dut': {'uses': {
        'BRICK': 'Wat is een verrassend gebruik voor een baksteen?',
        'FORK': 'Wat is een verrassend gebruik voor een vork?',
        'PAPERCLIP': 'Wat is een verrassend gebruik voor een paperclip?',
        'TOWEL': 'Wat is een verrassend gebruik voor een handdoek?'}},
    'eng': {
        'completion': {
            'GAMES': 'Complete this sentence in a surprising way: "At a sleepover, we..."',
            'LIBRARY': 'Complete this sentence in a surprising way: "When the kids were in the library, they found..."',
            'RAIN': 'Complete this sentence in a surprising way: "It started raining, and..."'},
        'consequences': {
            'KID PRESIDENT': 'What would be a surprising consequence if A KID WERE PRESIDENT?',
            'RAIN SODA': 'What would be a surprising consequence if RAIN WERE MADE OF SODA?'},
        'metaphors': {
            'BORING CLASS': 'Think of the most boring high school or college class you’ve ever taken. What was it like to sit through it?',
            'GROSS FOOD': 'Think about the most disgusting thing you have ever eaten or drunk. What was it like to eat or drink it?',
            'MESSY ROOM': 'Think of the messiest room in which you’ve ever lived. What was it like to live there?'},
        'uses': {
            'BOOK': 'What is a surprising use for a BOOK?',
            'BOTTLE': 'What is a surprising use for a BOTTLE?',
            'BOX': 'What is a surprising use for a BOX?',
            'BRICK': 'What is a surprising use for a BRICK?',
            'FORK': 'What is a surprising use for a FORK?',
            'KNIFE': 'What is a surprising use for a KNIFE?',
            'PAPERCLIP': 'What is a surprising use for a PAPERCLIP?',
            'ROPE': 'What is a surprising use for a ROPE?',
            'SHOE': 'What is a surprising use for a SHOE?',
            'SHOVEL': 'What is a surprising use for a SHOVEL?',
            'TABLE': 'What is a surprising use for a TABLE?',
            'TIRE': 'What is a surprising use for a TIRE?'}
        },
    'fre': {'uses': {
        'BROUETTE': 'Quel est un usage surprenant pour une BROUETTE?',
        'CEINTURE': 'Quel est un usage surprenant pour une CEINTURE?'}},
    'ger': {'uses': {
        'AXT': 'Was ist eine überraschende Verwendung für eine AXT?',
        'BÜROKLAMMER': 'Was ist eine überraschende Verwendung für eine BÜROKLAMMER?',
        'ERBSE': 'Was ist eine überraschende Verwendung für eine ERBSE?',
        'FLÖTE': 'Was ist eine überraschende Verwendung für eine FLÖTE?',
        'GEIGE': 'Was ist eine überraschende Verwendung für eine GEIGE?',
        'GURKE': 'Was ist eine überraschende Verwendung für eine GURKE?',
        'MÜLLTÜTE': 'Was ist eine überraschende Verwendung für eine MÜLLTÜTE?',
        'PAPRIKA': 'Was ist eine überraschende Verwendung für eine PAPRIKA?',
        'SCHAUFEL': 'Was ist eine überraschende Verwendung für eine SCHAUFEL?',
        'SCHRANK': 'Was ist eine überraschende Verwendung für einen SCHRANK?',
        'STUHL': 'Was ist eine überraschende Verwendung für einen STUHL?',
        'SÄGE': 'Was ist eine überraschende Verwendung für eine SÄGE?',
        'TISCH': 'Was ist eine überraschende Verwendung für einen TISCH?',
        'TOMATE': 'Was ist eine überraschende Verwendung für eine TOMATE?',
        'TROMMEL': 'Was ist eine überraschende Verwendung für eine TROMMEL?',
        'TROMPETE': 'Was ist eine überraschende Verwendung für eine TROMPETE?',
        'ZANGE': 'Was ist eine überraschende Verwendung für eine ZANGE?',
        'HAARFOEHN': 'Was ist eine überraschende Verwendung für einen HAARFOEHN?',
        'KONSERVENDOSE': 'Was ist eine überraschende Verwendung für eine KONSERVENDOSE?'}},
    'ita': {'uses': {
        'BOTTIGLIA DI PLASTICA': 'Qual è un uso sorprendente per una BOTTIGLIA DI PLASTICA?',
        'LAMPADINA': 'Qual è un uso sorprendente per una LAMPADINA?',
        'SEDIA': 'Qual è un uso sorprendente per una SEDIA?',
        'ACCETTA': "Qual è un uso sorprendente per un'ACCETTA?",
        'BANANA': 'Qual è un uso sorprendente per una BANANA?',
        'BICICLETTA': 'Qual è un uso sorprendente per una BICICLETTA?',
        'BORSA': 'Qual è un uso sorprendente per una BORSA?',
        'BOTTE': 'Qual è un uso sorprendente per una BOTTE?',
        'BOTTIGLIETTA': 'Qual è un uso sorprendente per una BOTTIGLIETTA?',
        'GRAFFETTA': 'Qual è un uso sorprendente per una GRAFFETTA?'}},
    'pol': {'uses': {
        'CEGŁA': 'Jakie jest zaskakujące zastosowanie dla CEGŁY?',
        'PUSZKA': 'Jakie jest zaskakujące zastosowanie dla PUSZKI?',
        'SZNUREK': 'Jakie jest zaskakujące zastosowanie dla SZNUREKA?'}},
    'rus': {'uses': {
        'ГАЗЕТА': 'Какое удивительное применение для ГАЗЕТЫ?',
        'ДЕРЕВЯННАЯ ЛИНЕЙКА': 'Какое удивительное применение для ДЕРЕВЯННОЙ ЛИНЕЙКИ?',
        'КАРТОННАЯ КОРОБКА': 'Какое удивительное применение для КАРТОННОЙ КОРОБКИ?'}}}


def infer_question(prompt, language='eng'):
    '''A very basic way to infer the question from the prompt. Generally,
    the non-English versions are chatbot-written, so please send fixes
    to peter.organisciak@du.edu! Some bad grammar is likely fine, though may have
    small benefits. Additional hard-coded prompts are in the question_inference_mapping,
    based on manual and ChatGPT-based corrections to the grammar.

    Also includes a prompt-to-question mapping if you want to override.
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
    # check in question mappings (currently only uses)
    if (language in question_inference_mapping and 
            prompt.upper() in question_inference_mapping[language]['uses']):
        return question_inference_mapping[language]['uses'][prompt.upper()]

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
                 round_precision=None,
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

    round_precision: If an integer, round the scores to this precision.

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

    # while normalize_values infers based on the average, we probably want to
    # make that inference here based on the mix/max across the original ratings
    if not range:
        range = (data[rater_cols].min().min(), data[rater_cols].max().max())
        mprint("- Inferred range of original data:", range)
    
    # replace anything below or above the range with na
    for col in rater_cols:
        clipped_col = data[col].apply(lambda x: x if range[0] <= x <= range[1] else np.nan)
        # if there are any clipped values, print a warning
        if clipped_col.isna().sum() > data[col].isna().sum():
            mprint(f"WARNING: {clipped_col.isna().sum()} out-of-range values clipped from {col}")
        data[col] = clipped_col
    
    data['avg_rating'] = data[rater_cols].mean(1, numeric_only=True, skipna=True)
    # count all values that are numeric, per row of data[rater_cols]
    data['rater_count'] = data[rater_cols].notna().sum(1)

    if round_adjust:
        # add a tiny bit to avg rater to round in direction of median. Only
        # need if there are tiebreakers
        data['median_rating'] = data[rater_cols].median(1)
        data['avg_rating'] = (data.avg_rating +
                              (data.median_rating - data.avg_rating).div(10**3)
                              )
    if include_rater_std:
        data['rating_std'] = data[rater_cols].std(1, numeric_only=True, skipna=True)

    data['target'] = normalize_values(data.avg_rating, oldrange=range,
                                      round_precision=round_precision)

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
                     round_precision=None):
    '''
    Normalize the range of score values to a standard scale.

    outrange: tuple of min and max of the new range. Default is 1-5.

    oldrange: tuple of min and max of the old range. If not provided, this
    will use the min and max of the data.

    round_precision: integer of the rounding precision. If None, no rounding
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