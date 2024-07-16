import asyncio
from typing import Literal
from ..types import StandardAIResponse, FullScore
from ..prompter import Base_Prompter
import openai
from tqdm.auto import tqdm
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from ..cache import Ocsai_Cache, Ocsai_Parquet_Cache, Ocsai_Redis_Cache
from ..llm_interface import LLM_Base_Interface
import nest_asyncio

class Base_Scorer:

    DEFAULT_PROMPTER = Base_Prompter
    DEFAULT_INTERFACE = LLM_Base_Interface
    max_logprobs = 0
    max_async_processes = 1
    async_client = None

    def __init__(
        self,
        openai_key_path: str | None = None,
        model_dict: dict = {},
        cache: str | Ocsai_Parquet_Cache | Ocsai_Redis_Cache | Path | None = None,
        logger=None,
        prompter=None,
        llm_interface=None,
        max_async_processes: int | None = None,
    ):
        if not logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self._models = model_dict
        if not max_async_processes:
            max_async_processes = self.max_async_processes
        self.async_semaphore = asyncio.Semaphore(max_async_processes)
        nest_asyncio.apply()
        self.async_client = None
        self.client = openai.OpenAI(api_key=openai.api_key)

        self.cache = None
        if cache:
            # if cache is string or Path, initialize parquet cache
            if isinstance(cache, str) or isinstance(cache, Path):
                self.cache = Ocsai_Parquet_Cache(cache_path=Path(cache))
            if isinstance(cache, Ocsai_Cache):
                self.cache = cache

        # ensure that subclasses set the prompter
        self.prompter = prompter if prompter else self.DEFAULT_PROMPTER()
        self.llm_interface = llm_interface if llm_interface else self.DEFAULT_INTERFACE()

    def originality(
        self,
        target: str | None,
        response: str,
        question: str | None = None,
        task_type: str | None = "uses",
        language: str | None = "eng",
        model: str = "first",
        raise_errs=False,
        score_source: Literal["top", "weighted"] = "top",
        **kwargs,
    ) -> float | None:
        """Get originality score"""
        response_dict = self.score(
            target,
            response,
            question=question,
            task_type=task_type,
            language=language,
            model=model,
            raise_errs=raise_errs,
            **kwargs,
        )
        scores = [r["score"] for r in response_dict]
        if len(scores) == 1:
            score = scores[0]
        else:
            score = scores[0] if score_source == "weighted" else scores[1]
        return score

    def _select_model_id(self, model):
        if model == "first":
            model = self.models[0]

        if model in self._models:
            model_id = self._models[model]
        else:
            model_id = model
        return model_id

    def score(
        self,
        target: str,
        response: str,
        question: str | None = None,
        task_type: str | None = "uses",
        language: str | None = "eng",
        model: str = "first",
        top_probs: int = 0,
        raise_errs: bool = False,
        confidence_priority: Literal["content", "probabilities"] = "probabilities",
        progressive_weighted: bool = False,
        prompt_kwargs: dict = {},
        **kwargs,
    ) -> list[FullScore]:
        """
        Get full score information: originality, flags, and confidence

        Args:

        top_probs: int: Number of top probabilities to return.
            If 0, only return the top completion.

        confidence_priority: bool: If getting confidence from multiple sources, which one to keep.

        progressive_weighted: If true, will return a FullScore for each *n*
            in the weighted completion. That is, if log probs were collected for
            10 possibilities, will return a weighted score for the first 1, 2, 3, etc.
            This is primary useful for evaluation of the approach.

        Returns:
        - A full score item, or two items if top_probs > 0 (one weighted, one top)
        """
        model_id = self._select_model_id(model)

        prompt_str = self.prompter.craft_prompt(
            target, response, question=question, task_type=task_type, language=language, **prompt_kwargs
        )

        standard_response_all_choices: list[list[StandardAIResponse]] = self._score_llm(
            prompt_str, model_id=model_id, top_probs=top_probs
        )
        # remove list wrapping - expects only one choice
        standard_response = [y[0] for y in standard_response_all_choices]
        assert len(standard_response) == 1
        return self._parse_standard_response(
            standard_response[0],
            confidence_priority=confidence_priority,
            raise_errs=raise_errs,
            progressive_weighted=progressive_weighted
        )

    async def score_async(
        self,
        target: str,
        response: str,
        question: str | None = None,
        task_type: str | None = "uses",
        language: str | None = "eng",
        model: str = "first",
        top_probs: int = 0,
        raise_errs: bool = False,
        confidence_priority: Literal["content", "probabilities"] = "probabilities",
        progressive_weighted: bool = False,
        prompt_kwargs: dict = {},
        async_if_available: bool = True,
        **kwargs,
    ) -> list[FullScore]:
        model_id = self._select_model_id(model)

        prompt_str = self.prompter.craft_prompt(
            target, response, question=question, task_type=task_type, language=language, **prompt_kwargs
        )

        standard_response_all_choices = await self._score_llm_async(
            prompt_str, model_id=model_id, top_probs=top_probs
        )
        standard_response = [y[0] for y in standard_response_all_choices]
        assert len(standard_response) == 1  # expected only one here
        return self._parse_standard_response(
            standard_response[0],
            confidence_priority=confidence_priority,
            raise_errs=raise_errs,
            progressive_weighted=progressive_weighted
        )

    def _parse_standard_response(
        self,
        standard_response: StandardAIResponse,
        raise_errs: bool = False,
        confidence_priority: Literal["content", "probabilities"] = "probabilities",
        progressive_weighted: bool = False
    ) -> list[FullScore]:
        """Take a single standard response, parse the content and log probs.
        Returns a list of FullScore items - one for the top completion, and one
         (if applicable) for the weighted completion.
        """
        try:
            parsed_content: FullScore = self.prompter.parse_content(
                standard_response["content"], type="top"
            )
        except KeyboardInterrupt:
            raise
        except:  # noqa: E722
            if raise_errs:
                self.logger.exception(
                    f"Problem with content parsing:{standard_response['content']}"
                )
                raise
            parsed_content = {
                "score": None,
                "confidence": None,
                "flags": None,
                "n": None,
                "type": "other",
            }
        if standard_response["logprobs"] is not None:
            weighted_scores: list[FullScore] = []
            if progressive_weighted:
                probs_to_parse = [standard_response["logprobs"][:i]
                                  for i in range(2, len(standard_response["logprobs"]) + 1)]
            else:
                probs_to_parse = [standard_response["logprobs"]]

            for logprobs in probs_to_parse:
                probability_scores = self.prompter.probability_scores(logprobs)

                weighted_score: FullScore = {
                    "score": probability_scores["weighted"],
                    "confidence": probability_scores["weighted_confidence"],
                    "flags": parsed_content["flags"],
                    "n": probability_scores["n"],
                    "type": "weighted",
                }
                weighted_scores.append(weighted_score)
            top_score: FullScore = {
                "score": probability_scores["top"],
                "confidence": probability_scores["top_confidence"],
                "flags": parsed_content["flags"],
                "n": 1,
                "type": "top",
            }
            if (confidence_priority == "content") and parsed_content["confidence"]:
                weighted_score["confidence"] = parsed_content["confidence"]
                top_score["confidence"] = parsed_content["confidence"]

            return [*weighted_scores, top_score]
        else:
            return [parsed_content]

    def add_model(self, name, finetunepath):
        self.models[name] = finetunepath

    def _verify_logprobs(self, top_probs: int):
        if top_probs > 0:
            if top_probs > self.max_logprobs:
                self.logger.warning(
                    f"This API only supports {self.max_logprobs} logprobs at a time. Forcing top_probs={max_probs}."
                )
                top_probs = self.max_logprobs
        return top_probs if top_probs > 0 else None

    def _check_if_running_loop(self):
        # if there is a running loop (e.g. in a jupyter notebook), raise an error
        # weird function - looping for the error
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            self.logger.warn('''Event loop is already running. Usually this means that you're running code in Jupyter,
                             which has runs async. The problem is that async functions cannot be run within synchronous
                             functions when there's already a loop running. If you're developing code to run as a script,
                             this should be fine to ignore - async won't work now, but will in the script.''')
            return True
        return False

    async def _score_llm_async(
        self,
        gptprompt: str | list[str],
        model_id: str,
        top_probs: int = 0,
    ) -> list[list[StandardAIResponse]]:
        gptprompt = [gptprompt] if isinstance(gptprompt, str) else gptprompt
        logprobs = self._verify_logprobs(top_probs)

        assert self.async_client is not None
        assert self.async_semaphore is not None

        async with self.async_semaphore:
            tasks = []
            for prompt in gptprompt:
                tasks.append(
                    self.llm_interface.completion_async(
                        async_client=self.async_client,
                        model=model_id,
                        prompt=prompt,
                        temperature=1e-9,
                        sys_msg_text=self.prompter.sys_msg_text,
                        logprobs=logprobs,
                        stop_char=self.prompter.stop_char,  # STOP on newline only if aiming for score only
                        max_tokens=self.prompter.max_tokens,
                    )
                )
            all_responses = await asyncio.gather(*tasks)

        return all_responses

    def _score_llm(
        self,
        gptprompt: str | list[str],
        model_id: str,
        top_probs: int = 0
    ) -> list[list[StandardAIResponse]]:
        """
        gptprompt is the templated item+response+type+language. Use _craft_gptprompt.

        If a string is provided, returns a single response. If a list is provided,
        returns a list of responses in the same order.

        Returns:
        - A list of lists of StandardAIResponse items. The outer list is for each prompt,
            the inner list is for each choice (usually just one)
        """
        gptprompt = [gptprompt] if isinstance(gptprompt, str) else gptprompt
        logprobs: int | None = self._verify_logprobs(top_probs)

        all_responses = []
        for prompt in (tqdm(gptprompt) if len(gptprompt) > 10 else gptprompt):
            response = self.llm_interface.completion(
                client=self.client,
                model=model_id,
                prompt=prompt,
                temperature=1e-9,
                sys_msg_text=self.prompter.sys_msg_text,
                logprobs=logprobs,
                stop_char=self.prompter.stop_char,  # STOP on newline only if aiming for score only
                max_tokens=self.prompter.max_tokens,
            )
            all_responses.append(response)

        return all_responses

    def originality_batch(
        self,
        prompts: list[str | None] | str | None,
        responses: list[str],
        questions: list[str | None] | str | None = None,
        task_types: list[str | None] | str | None = None,
        languages: list[str | None] | str | None = None,
        model: str = "first",
        raise_errs: bool = False,
        batch_size: int = 20,
        top_probs: int = 0,
        score_source: Literal["top", "weighted"] = "top",
        confidence_priority: Literal["content", "probabilities"] = "probabilities",
        debug: bool = False,
        min_size_to_write_cache: int = 100,
        use_async: bool = False,
        sleep_between_batches: float = 0.1,
        **kwargs,
    ):
        """Get originality in a batch, with optional caching.

        Args:
            score_source: If using top_probs, which source to use for the score.
                'top' uses the top completion, 'weighted' uses the weighted completion.
                Ignored if top_probs=0, forced to 'top` if top_probs == 0.

            progressive_weighted: If true, will return a FullScore for each *n*
                in the weighted completion.
        """
        if top_probs == 0:
            score_source = "top"

        scores = []
        confidences = []
        allflags = []

        assert not (questions is None and prompts is None)

        responses = [r.strip() for r in responses]

        def cast_to_list(x: list | str | pd.Series) -> list[str | None]:
            if type(x) is list and len(x) == 0:
                x = None
            if (type(x) is str) or (x is None):
                x = [x] * len(responses)
            elif isinstance(x, pd.Series):
                x = x.tolist()
            return x

        promptsl: list[str | None] = cast_to_list(prompts)
        questionsl: list[str | None] = cast_to_list(questions)
        task_typesl: list[str | None] = cast_to_list(task_types)
        languagesl: list[str | None] = cast_to_list(languages)

        model_id = self._select_model_id(model)

        if self.cache:
            df = pd.DataFrame(
                list(zip(promptsl, responses, questionsl, task_typesl, languagesl)),
                columns=self.cache.base_cols[:-1],
            )
            df["model"] = model_id
            df = df.astype({col: "object" for col in self.cache.base_cols})
            to_score, cache_results = self.cache.get_cache_scores(df)
            promptsl = to_score.prompt.tolist()
            questionsl = to_score.question.tolist()
            responses = to_score.response.tolist()

        nbatches = np.ceil(len(responses) / batch_size).astype(int)

        for i in tqdm(range(nbatches)):
            start, end = i * batch_size, (i + 1) * batch_size

            gptprompts = []
            for target, response, task_type, question, language in zip(
                promptsl[start:end],
                responses[start:end],
                task_typesl[start:end],
                questionsl[start:end],
                languagesl[start:end]
            ):
                gptprompts.append(
                    self.prompter.craft_prompt(
                        target,
                        response,
                        task_type=task_type,
                        question=question,
                        language=language,
                    )
                )
            if use_async:
                if self._check_if_running_loop():
                    use_async = False

            if use_async:
                standard_responses = asyncio.run(
                        self._score_llm_async(
                            gptprompts, model_id=model_id, top_probs=top_probs
                        )
                    )
            else:
                standard_responses = self._score_llm(
                    gptprompts, model_id=model_id, top_probs=top_probs
                )
                
            for i, standard_response_all_choices in enumerate(standard_responses):
                if len(standard_response_all_choices) > 1:
                    raise Exception("Batching not supported for multiple completion choices")

                standard_response = standard_response_all_choices[0]
                try:
                    fullscores = self._parse_standard_response(
                        standard_response,
                        confidence_priority=confidence_priority,
                        raise_errs=raise_errs,
                        progressive_weighted=False  # as it stands, not used downstream
                    )
                    if top_probs > 0:
                        fullscore = (
                            fullscores[0]
                            if score_source == "weighted"
                            else fullscores[1]
                        )
                    else:
                        fullscore = fullscores[0]
                except KeyboardInterrupt:
                    raise
                except:  # noqa: E722
                    if raise_errs:
                        print(f"GPT prompt: {gptprompts[i].strip()}")
                        print(f"raw response: {standard_response['content']}")
                        raise
                scores.append(fullscore["score"])
                confidences.append(fullscore["confidence"])
                allflags.append(fullscore["flags"])

            if sleep_between_batches:
                time.sleep(sleep_between_batches)

        if self.cache:
            newly_scored = to_score.copy()
            newly_scored["score"] = scores
            newly_scored["confidence"] = confidences
            newly_scored["flags"] = allflags
            newly_scored["timestamp"] = time.time()
            self.cache.write(newly_scored, min_size_to_write_cache)

            right = pd.concat([cache_results, newly_scored])
            self.logger.debug(
                f"score length: {len(right)}; Merging back to original {len(df)} item frame"
            )
            final_results = df.merge(
                right, how="left", on=self.cache.base_cols
            ).replace({np.nan: None})
            # return a list of dicts, with score, confidence, and flags
            return final_results[["score", "confidence", "flags"]].to_dict("records")
        else:
            final = []
            for s, c, f, r in zip(scores, confidences, allflags, responses):
                # force 1 on blank response
                if r.strip() == "":
                    logging.debug("Blank response detected. Forcing score to 1.")
                    s = 1
                if c is not None and np.isnan(c):
                    c = None

                def parse_none_strings(x):
                    if x.lower() in ['none', 'na', 'nan', 'null']:
                        return None
                if type(f) is list:
                    f = [x for x in f if parse_none_strings(x) is not None]

                if f is not None and (len(f) == 0):
                    f = None
                final.append({"score": s, "confidence": c, "flags": f})
            return final

    def originality_df(
        self,
        dataframe,
        model: str = "first",
        raise_errs: bool = False,
        batch_size: int = 20,
        prompt_col: str | None = "prompt",
        response_col: str | None = "response",
        question_col: str | None = "question",
        type_col: str | None = "type",
        language_col: str | None = "language",
        score_name: str = "score",
        confidence_name: str = "confidence",
        flags_name: str = "flags",
        min_size_to_write_cache: int = 100,
        force_overwrite: bool = False,
        use_async: bool = True,
        sleep_between_batches: float = 0.1,
    ):
        """Run originality scoring on a dataframe, and append the results to a new dataframe"""

        if type_col in dataframe.columns:
            task_types = dataframe[type_col]
        else:
            task_types = "uses"

        if language_col in dataframe.columns:
            languages = dataframe[language_col]
        else:
            languages = "eng"

        assert (prompt_col in dataframe.columns) or (question_col in dataframe.columns)
        assert response_col in dataframe.columns

        scores = self.originality_batch(
            prompts=dataframe[prompt_col] if prompt_col in dataframe.columns else None,
            responses=dataframe[response_col],
            questions=dataframe[question_col] if question_col in dataframe.columns else None,
            task_types=task_types,
            languages=languages,
            model=model,
            raise_errs=raise_errs,
            batch_size=batch_size,
            min_size_to_write_cache=min_size_to_write_cache,
            use_async=use_async,
            sleep_between_batches=sleep_between_batches
        )
        scores_df = pd.DataFrame(scores)
        outdf = dataframe.copy()
        newcolnames = [score_name, confidence_name, flags_name]
        # ensure that newcolnames are not already in dataframe
        if not force_overwrite:
            for newcolname in newcolnames:
                if newcolname in outdf.columns:
                    raise Exception(
                        f"Column name {newcolname} already exists in dataframe. "
                        "Try setting a different name"
                    )
        outdf[newcolnames] = scores_df.values
        return outdf

    @property
    def models(self):
        """Return just the names of the models"""
        return list(self._models.keys())

    def fluency(self, **kwargs):
        raise Exception(
            "Fluency is not calculated at the item level. Use `ocs.file.fluency` to calculate it."
        )

    def elaboration(self, phrase, elabfunc: str = "whitespace"):
        if elabfunc == "whitespace":

            def elabfunc_1(x):
                return len(x.split())

        else:
            raise Exception("Only whitespace elaboration calculated by LLM Scoring.")

        try:
            elab = elabfunc_1(phrase)
        except:  # noqa: E722
            raise
            elab = None
        return elab
