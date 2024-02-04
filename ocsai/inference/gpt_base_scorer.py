
import getpass
import openai
from tqdm.auto import tqdm
from pathlib import Path
import duckdb
import time
import logging
import os
import pandas as pd
import numpy as np


class GPT_Base_Scorer:
    def __init__(self, openai_key_path=False, model_dict=None,
                 cache=None, logger=None, prompter=None):
        self.logger = logger
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        if openai_key_path:
            openai.api_key_path = openai_key_path
        elif os.environ.get('OPENAI_API_KEY'):
            openai.api_key = os.environ.get('OPENAI_API_KEY')
        else:
            openai.api_key = getpass.getpass(prompt='Enter API Key:').strip()

        self._models = model_dict
        self.client = openai.OpenAI()

        self.cache_path = None
        if cache:
            self.cache_path = Path(cache)
            self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # ensure that subclasses set the prompter
        self.prompter = prompter
        assert self.prompter is not None

    def originality(self, target, response, question=None,
                    task_type='uses', language='eng',
                    model='first', raise_errs=False, **kwargs):
        '''Get originality score'''
        response_dict = self.score(target, response, question=question,
                                   task_type=task_type, language=language,
                                   model=model, raise_errs=raise_errs, **kwargs)
        return response_dict['score']

    def score(self, target, response, question=None,
              task_type='uses', language='eng',
              model='first', raise_errs=False, **kwargs):
        '''Get full score information: originality, flags, and confidence'''
        if model == 'first':
            model = self.models[0]
        prompt_str = self.prompter.craft_prompt(target, response, question=question,
                                                task_type=task_type, language=language)
        score_raw = self._score_gpt(prompt_str, model=model, just_final=True)[0]
        try:
            response_dict = self.prompter.parse_response(score_raw)
        except:
            if raise_errs:
                self.logger.exception(f"GPT prompt:{prompt_str.strip()}\nraw response:{score_raw}")
                raise
            return {'score': None, 'confidence': None, 'flags': None}
        return response_dict

    def add_model(self, name, finetunepath):
        self.models[name] = finetunepath

    def _score_gpt(self, gptprompt, model='first', just_final=False):
        raise NotImplementedError

    def originality_batch(self, prompts, responses, questions=None, task_types='uses',
                          languages='eng',
                          model='first', raise_errs=False, batch_size=500,
                          debug=False, **kwargs):
        scores = []
        confidences = []
        allflags = []
        responses = [r.strip() for r in responses]
        if (type(task_types) is str) or (task_types is None):
            task_types = [task_types]*len(prompts)
        if (type(languages) is str) or (languages is None):
            languages = [languages]*len(prompts)
        if (type(questions) is str) or (questions is None):
            questions = [questions]*len(prompts)

        assert len(prompts) == len(responses)
        if model == 'first':
            model = self.models[0]

        # ensure that all base col types are forced to be treated as strings
        base_cols = ['prompt', 'response', 'question', 'type', 'language', 'model']

        if (self.cache_path):
            df = pd.DataFrame(list(zip(prompts, responses, questions, task_types, languages)),
                              columns=base_cols[:-1])
            df['model'] = self._models[model]
            df = df.astype({col: 'object' for col in base_cols})

            if len(list(self.cache_path.glob('*.parquet'))) == 0:
                cache_results = pd.DataFrame([], columns=base_cols+['score', 'timestamp'])
                cache_results = df.merge(cache_results, how='left', on=base_cols)
            else:
                # Using IS NOT DISTINCT FROM to handle nulls
                col_match_sql = " AND ".join([f"df.{x} IS NOT DISTINCT FROM cache.{x}" for x in base_cols])
                cache_results = duckdb.query(f"SELECT df.*, cache.score, cache.confidence, cache.flags, cache.timestamp FROM df LEFT JOIN '{self.cache_path}/*.parquet' cache ON {col_match_sql}").to_df()
            cache_results = cache_results.drop_duplicates(base_cols)
            cache_results = cache_results.astype({col: 'object' for col in base_cols})
            # force non-response score to be 1.
            cache_results.loc[cache_results.response.str.strip() == '', 'score'] = 1
            to_score = cache_results[cache_results.score.isna()]

            cache_results = cache_results[~cache_results.score.isna()]
            self.logger.debug(f"To score:{cache_results.score.isna().sum()} / {len(cache_results)}")
            prompts, responses = to_score.prompt.tolist(), to_score.response.tolist()

        nbatches = np.ceil(len(prompts) / batch_size).astype(int)

        for i in tqdm(range(nbatches)):
            targetbatch = prompts[i*batch_size:(i+1)*batch_size]
            responsebatch = responses[i*batch_size:(i+1)*batch_size]

            gptprompts = [self.prompter.craft_prompt(target, response) for target, response in zip(targetbatch, responsebatch)]
            scores_raw = self._score_gpt(gptprompts, model=model, just_final=True)

            for i, score_raw in enumerate(scores_raw):
                score, confidence, flags = None, None, None
                try:
                    response_dict = self.prompter.parse_response(score_raw)
                    score = response_dict['score']
                    confidence = response_dict['confidence']
                    flags = response_dict['flags']
                except:
                    if raise_errs:
                        print(f"GPT prompt: {gptprompts[i].strip()}")
                        print(f"raw response: {score_raw}")
                        raise
                scores.append(score)
                confidences.append(confidence)
                allflags.append(flags)

        if (self.cache_path):
            newly_scored = to_score.copy()
            newly_scored['score'] = scores
            newly_scored['confidence'] = confidences
            newly_scored['flags'] = allflags
            newly_scored['timestamp'] = time.time()
            if not newly_scored.empty:
                newly_scored.to_parquet(self.cache_path / f'results.{time.time()}.parquet')

            right = pd.concat([cache_results, newly_scored])
            self.logger.debug(f"score length: {len(right)}; Merging back to original {len(df)} item frame")
            final_results = df.merge(right, how='left', on=base_cols).replace({np.nan: None})
            # return a list of dicts, with score, confidence, and flags
            return final_results[['score', 'confidence', 'flags']].to_dict('records')
        else:
            final = []
            for s, c, f, r in zip(scores, confidences, allflags, responses):
                # force 1 on blank response
                if r.strip() == '':
                    logging.debug("Blank response detected. Forcing score to 1.")
                    s = 1
                if c is None or np.isnan(c):
                    c = None
                if f is None or np.isnan(f):
                    f = None
                final.append({"score": s,"confidence": c,"flags": f})
            return final

    def originality_df(self, dataframe,
                   model='first',
                   raise_errs=False,
                   batch_size=500,
                   prompt_col='prompt',
                   response_col='response',
                   question_col='question',
                   type_col='type',
                   language_col='language',
                   score_name='score',
                   confidence_name='confidence',
                   flags_name='flags',
                   force_overwrite=False):
        '''Run originality scoring on a dataframe, and append the results to a new dataframe'''
        if question_col in dataframe.columns:
            questions = dataframe[question_col]
        else:
            questions = None
        
        if type_col in dataframe.columns:
            task_types = dataframe[type_col]
        else:
            task_types = 'uses'

        if language_col in dataframe.columns:
            languages = dataframe[language_col]
        else:
            languages = 'eng'

        assert prompt_col in dataframe.columns
        assert response_col in dataframe.columns

        scores = self.originality_batch(dataframe[prompt_col],
                                        dataframe[response_col],
                                        questions=questions,
                                        task_types=task_types,
                                        languages=languages,
                                        model=model,
                                        raise_errs=raise_errs,
                                        batch_size=batch_size)
        scores_df = pd.DataFrame(scores)
        outdf = dataframe.copy()
        newcolnames = [score_name, confidence_name, flags_name]
        # ensure that newcolnames are not already in dataframe
        if not force_overwrite:
            for newcolname in newcolnames:
                if newcolname in outdf.columns:
                    raise Exception(f"Column name {newcolname} already exists in dataframe. Try setting a difference name")
        outdf[newcolnames] = scores_df.values
        return outdf

    @property
    def models(self):
        ''' Return just the names of the models'''
        return list(self._models.keys())

    def fluency(self, **kwargs):
        raise Exception("Fluency is not calculated at the item level. Use `ocs.file.fluency` to calculate it.")

    def elaboration(self, phrase, elabfunc="whitespace"):
        if elabfunc == 'whitespace':
            def elabfunc(x):
                return len(x.split())
        else:
            raise Exception("Only whitespace elaboration calculated by LLM Scoring.")

        try:
            elab = elabfunc(phrase)
        except:
            raise
            elab = None
        return elab
