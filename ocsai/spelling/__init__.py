import pandas as pd
import textwrap
from ..utils import generic_llm
import anthropic
import openai


def fix_kids_spelling(question: str,
                      response: str,
                      client: anthropic.Anthropic | openai.OpenAI,
                      model: str = 'claude-3-sonnet-20240229',
                      identity_desc: str = "a 2nd grader") -> str | None:
    ''' For balance of performance and pricing, claude-sonnet is a good option here.'''
    if pd.isna(response):
        return None
    response_template = textwrap.dedent('''
    The following response was written by {identify_desc}, in response to the question, `{q}`:

    Response: `{r}`

    Fix any typos in the response, if any, else pre-print the original. In looking for typos, remember that children may have different types of errors (e.g. phonetic spellings, character reversals, omission of consonant blends, overgeneralization of grammatical rules, sight word errors, invented spellings, transpositions, and missing or extra spaces).

    # Action
    Reprint the response (and only the response), making *best-guess* corrections to the writing as necessary, or keeping as-is if it is correct or if there's no reasonable correction.

    ## Rules
    - DO NOT ADD COMMENTARY: JUST WRITE THE RESPONSE.
    - DO NOT include the question or any prefixes: JUST RETURN THE RESPONSE.
    - Do NOT add typos - YOU FIX THEM. Punctuation doesn't need correction.
    ''')

    prompt = response_template.format(q=question,
                                      r=response,
                                      identify_desc=identity_desc)
    print(prompt)
    content = generic_llm(text=prompt,
                          sysmsg="You FIX children's spelling.",
                          client=client,
                          model=model)

    if '`' in content:
        content = content.split('`')[1]

    if content.startswith('"') and not response.startswith('"'):
        content = content.strip('"')

    return content

