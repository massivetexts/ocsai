from typing import Optional, List
import re


class NumberedListOutputParser:
    """Parse out a numbered list."""

    def __init__(self, examples: Optional[List[str]] = ["response", "score"]):
        self.examples = examples

    def get_format_instructions(self) -> str:
        return ""  # no format instructions - it should be apparent from the template

    def parse(self, text: str) -> List[List[str]]:
        """Parse the output of an LLM call."""
        lines = text.strip().split("\n")
        outputs = []
        for line in lines:
            value = re.split("^\d+\. ?", line)[-1]
            outputs.append(value)
        return outputs
