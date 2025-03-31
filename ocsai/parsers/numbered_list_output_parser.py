import re

class NumberedListOutputParser:
    """Parse out a numbered list."""

    def __init__(self, examples: list[str] | None = ["response", "score"]):
        self.examples = examples

    def get_format_instructions(self) -> str:
        return ""  # no format instructions - it should be apparent from the template

    def parse(self, text: str) -> list[list[str]]:
        """Parse the output of an LLM call."""
        lines = text.strip().split("\n")
        outputs = []
        for line in lines:
            value = re.split(r"^\d+\. ?", line)[-1]
            outputs.append(value)
        return outputs
