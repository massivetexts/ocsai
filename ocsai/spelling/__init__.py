import textwrap

import anthropic
import openai
import pandas as pd

from ..utils import generic_llm


def fix_kids_spelling(question: str,
                      response: str,
                      client: anthropic.Anthropic | openai.OpenAI,
                      model: str = 'claude-3-sonnet-20240229',
                      identity_desc: str = "a 2nd grader") -> str | None:
    ''' For balance of performance and pricing, claude-sonnet is a good option here.'''
    if pd.isna(response):
        return None
    # PROMPT-CACHING MIGHT BE NICE, BUT THATS ONLY FOR >1024 TOKENS (gpt) and >2048 (sonnet)
    sysmsg = textwrap.dedent('''
    You FIX children's spelling.

    Fix any typos in the response, if any, else re-print the original. In looking for typos, remember that children may have different types of errors (e.g. phonetic spellings, character reversals, omission of consonant blends, overgeneralization of grammatical rules, sight word errors, invented spellings, transpositions, and missing or extra spaces).

    # Action
    Reprint the response (and only the response), making *best-guess* corrections to the writing as necessary, or keeping as-is if it is correct or if there's no reasonable correction.

    ## Rules
    - DO NOT ADD COMMENTARY: JUST WRITE THE RESPONSE.
    - DO NOT include the question or any prefixes: JUST RETURN THE RESPONSE. 
    - Do NOT add typos - YOU FIX THEM. Punctuation doesn't need correction.
    - Try your best to fix the spelling. If you can't, then return [NONSENSE] or [NA], as appropriate. 
                             Be conservative andu se these tags as a last resort - remember that these are
      children, so spelling can be far off but decipherable. For example:
       - A salidboil -> A salad bowl
       - Meer -> Mirror
       - Goow --> Goo
       - Gien -> Giant
       - Sord -> Sword
       - Me o -> Meow
       - musoger -> massager
       - marocas -> maracas
       - Bely -> Belly                   
       - Jrumstic -> Drumstick
       - Toopay --> Toupee
       - The ate hippos -> They ate hippos
    ''')

    response_template = textwrap.dedent('''
    The following response was written by {identify_desc}, in response to the question, `{q}`:

    Response: `{r}`
                                        
    Spell-checked response: ''')

    prompt = response_template.format(q=question,
                                      r=response,
                                      identify_desc=identity_desc)
    content = generic_llm(text=prompt,
                          sysmsg=sysmsg,
                          client=client,
                          model=model)

    if '`' in content:
        content = content.split('`')[1]

    if content.startswith('"') and not response.startswith('"'):
        content = content.strip('"')

    return content

