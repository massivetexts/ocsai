import logging
from ..types import LogProbPair, ParsedProbPair, ProbPair, ProbScores, ResponseTypes, FullScore
import numpy as np


class Base_Prompter:
    sys_msg_text: str | None = None
    max_tokens: int = 100
    stop_char: str | None = None

    def __init__(self, logger: logging.Logger | None = None):
        if logger:
            self.logger: logging.Logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

    def craft_prompt(
        self,
        item: str | None,
        response: str,
        task_type: str | None = None,
        question: str | None = None,
        language: str | None = None,
    ):
        """Craft a prompt for the language model, given an item, response, and task type"""
        raise NotImplementedError

    def craft_response(self, score, confidence=None, flags=None):
        """Craft a response for the language model, given a score, confidence, and flags"""
        raise NotImplementedError

    def probability_scores(self, score_logprobs: list[LogProbPair]) -> ProbScores:
        """From a standardized list of token log probabilities, return a dictionary of
        weighted and top scores, and their confidences."""

        # convert log probabilities to probabilities
        score_probs: list[ProbPair] = [
            (token, np.exp(log_prob)) for token, log_prob in score_logprobs
        ]

        # Try to parse the scores
        token_probs: list[ParsedProbPair] = []
        for token, prob in score_probs:
            try:
                parsed = self.parse_content(token, type="other")
                score = parsed["score"]
            except ValueError:
                continue
            if not score:
                continue
            token_probs.append((score, prob))

        # Top Choice
        top_choice, confidence = token_probs[0]

        # Weighted Choice
        scores, weights = list(zip(*token_probs))
        weighted_choice = np.average(scores, weights=weights)
        weighted_confidence = sum([w for t, w in token_probs])
        n = len(token_probs)

        return {
            "weighted": weighted_choice,
            "weighted_confidence": weighted_confidence,
            "top": top_choice,
            "top_confidence": confidence,
            "n": n,
        }

    def parse_content(self, content: str, type: ResponseTypes) -> FullScore:
        """Parse the raw text response from the language model into a dict of
        {score, confidence, and flags}

        Sometimes the confidence will come from a different places - that happens
            downstream - this method only parses the text content.

        Args:
            content (str): The text content response from the model
            type (str): What kind of response is this? "top" refers to the top completion,
                "other" refers to another type of completion. This information is kept for
                compatibility with log probability parsing.
        """
        raise NotImplementedError

    def prepare_example(
        self,
        item: str,
        response: str,
        task_type: str | None = None,
        question: str | None = None,
        language: str | None = None,
        target=None,
        confidence=None,
        seed=None,
    ):
        """Prepare an example for training. Not needed for inference."""
        raise NotImplementedError
