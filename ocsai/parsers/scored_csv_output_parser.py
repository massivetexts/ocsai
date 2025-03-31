import csv
import re
from io import StringIO


class ScoredCSVOutputParser:
    """Parse out multiple line comma separated lists."""

    def __init__(self, examples: str | None = ["response", "score"]):
        self.examples = examples

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, "
            f"eg: `{','.join(self.examples)}`"
        )

    def parse(self, text: str) -> list[list[str]]:
        """Parse the output of an LLM call."""
        csv_file = StringIO(text)
        cleaned = []
        for row in csv.reader(csv_file):
            if len(row) > 2:
                # fallback on re
                line = ",".join(row)
                row = re.split(r", ?(\d\.\d+)", line)[:-1]
            assert (
                len(row) == 2
            ), f"Currently assumes only two fields: response, score; seeing {len(row)}"
            response, score = row
            response = response.strip("-").strip()
            try:
                score = float(score.strip())
            except ValueError:
                score = score.strip()
            cleaned.append((response, score))
        return cleaned
