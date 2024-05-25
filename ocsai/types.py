from typing import Literal, TypedDict

ProbScores = TypedDict(
    "ProbScores",
    {
        "weighted": float,
        "weighted_confidence": float,
        "top": float,
        "top_confidence": float,
        "n": int,
    },
)

# When parsing a response, what was the source for the data?
# top: The top completion
# weighted: A weighted average of all completions
# other: Something else
ResponseTypes = Literal["weighted", "top", "other"]

LogProbPair = tuple[str, float]
ProbPair = tuple[str, float]
ParsedProbPair = tuple[float, float]

# a format for a standard AI response, for openai or other clients to be cast into
UsageStats = TypedDict(
    "UsageStats",
    {
        "total": int | None,
        "prompt": int | None,
        "completion": int | None,
    },
)

StandardAIResponse = TypedDict(
    "StandardAIResponse",
    {
        "content": str,
        "logprobs": list[LogProbPair] | None,
        "usage": UsageStats | None,
    },
)

FullScore = TypedDict(
    "FullScore",
    {
        "score": float | None,
        "confidence": int | float | None,
        "flags": list[str] | None,
        "n": int | None,
        "type": ResponseTypes,
    },
)
