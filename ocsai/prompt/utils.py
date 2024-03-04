import random


def strip_backticks(s: str) -> str:
    # return the first block in backticks (stripping any markdown codebook format details)
    s = s.split("```")[1].split("```")[0]
    if "\n" in s:
        s = s.split("\n", 1)[1]
    return s.strip()


def hashname():
    return str(random.getrandbits(32))
