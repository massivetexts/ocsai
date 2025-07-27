import textwrap
from typing import Literal
import anthropic
import openai
import pandas as pd
import asyncio
from tqdm.auto import tqdm
from ..utils import generic_llm, generic_llm_async

def construct_spelling_prompt(question: str, response: str, identity_desc: str,
                              examples: list[tuple[str, str]] | None | Literal["default"] = "default") -> tuple[str, str]:
    
    if examples == "default":
        correction_examples = [
            ("A salidboil", "A salad bowl"),
            ("Meer", "Mirror"),
            ("Goow", "Goo"),
            ("Gien", "Giant"),
            ("Sord", "Sword"),
            ("Me o", "Meow"),
            ("musoger", "massager"),
            ("marocas", "maracas"),
            ("Bely", "Belly"),
            ("Jrumstic", "Drumstick"),
            ("Toopay", "Toupee"),
            ("The ate hippos", "They ate hippos")
        ]
    else:
        correction_examples = examples

    sysmsg = textwrap.dedent('''
        You FIX children's spelling.

        Fix any typos in the response, if any, else re-print the original. In looking for typos, remember that children may have different types of errors (e.g. phonetic spellings, character reversals, omission of consonant blends, overgeneralization of grammatical rules, sight word errors, invented spellings, transpositions, and missing or extra spaces).

        # Action
        Reprint the response (and only the response), making *best-guess* corrections to the writing as necessary, or keeping as-is if it is correct or if there's no reasonable correction.

        ## Rules
        - DO NOT ADD COMMENTARY: JUST WRITE THE RESPONSE.
        - DO NOT include the question or any prefixes: JUST RETURN THE RESPONSE. 
        - Do NOT add typos - YOU FIX THEM.
        - Punctuation doesn't need correction. No need to add (or remove) periods and sentence-casing.
        - Try your best to fix the spelling. If you can't, then return [NONSENSE] or [NA], as appropriate. 
                                Be conservative and use these tags as a last resort - remember that these are
        children, so spelling can be far off but decipherable.
        ''')
    
    if correction_examples is not None:
        sysmsg += "\n\n## Examples\n"
        sysmsg += "\n".join(f"       - {wrong} -> {correct}" for wrong, correct in correction_examples)

    response_template = textwrap.dedent('''
    The following response was written by {identify_desc}, in response to the question, `{q}`:

    Response: `{r}`
                                        
    Spell-checked response: ''')

    user_msg = response_template.format(q=question,
                                      r=response,
                                      identify_desc=identity_desc)
    
    return sysmsg, user_msg

def clean_spelling_response(response: str) -> str:
    if '`' in response:
        response = response.split('`')[1]

    if response.startswith('"') and not response.startswith('"'):
        response = response.strip('"')
    return response

def fix_kids_spelling(question: str,
                      response: str,
                      client: anthropic.Anthropic | openai.OpenAI,
                      model: str = 'claude-3-7-sonnet-latest',
                      identity_desc: str = "a 2nd grader") -> str | None:
    if pd.isna(response):
        return None
    # PROMPT-CACHING MIGHT BE NICE, BUT THATS ONLY FOR >1024 TOKENS (gpt) and >2048 (sonnet)
    sysmsg, user_msg = construct_spelling_prompt(question, response, identity_desc)

    content = generic_llm(text=user_msg,
                          sysmsg=sysmsg,
                          client=client,
                          model=model)

    return clean_spelling_response(content)

async def fix_kids_spelling_async(question: str,
                      response: str,
                      client: anthropic.AsyncAnthropic | openai.AsyncOpenAI,
                      model: str = 'claude-3-7-sonnet-latest',
                      identity_desc: str = "a 2nd grader") -> str | None:
    ''' Async version of fix_kids_spelling.'''
    if pd.isna(response):
        return None

    sysmsg, user_msg = construct_spelling_prompt(question, response, identity_desc)

    content = await generic_llm_async(text=user_msg,
                          sysmsg=sysmsg,
                          client=client,
                          model=model)

    return clean_spelling_response(content)

async def fix_kids_spelling_batch(
    questions: list[str] | str,
    responses: list[str],
    client: anthropic.AsyncAnthropic | openai.AsyncOpenAI,
    model: str = 'claude-3-7-sonnet-latest',
    identity_desc: str = "a 2nd grader",
    batch_size: int = 10
) -> list[str | None]:
    """
    Process a batch of spelling corrections asynchronously.
    
    Args:
        questions: List of questions
        responses: List of responses to correct
        client: Async LLM client
        model: Model to use
        identity_desc: Description of the writer's identity
        batch_size: Number of corrections to process in parallel
        
    Returns:
        List of corrected responses (or None for invalid inputs)
    """
    if len(questions) != len(responses):
        raise ValueError("questions and responses must be the same length")
    
    if isinstance(questions, str):
        questions = [questions] * len(responses)

    # Process in batches to avoid overwhelming the API
    corrected_responses = []
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_questions = questions[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]
        
        # Create tasks for this batch
        tasks = [
            fix_kids_spelling_async(
                question=q,
                response=r,
                client=client,
                model=model,
                identity_desc=identity_desc
            )
            for q, r in zip(batch_questions, batch_responses)
        ]
        
        # Run the batch concurrently
        batch_results = await asyncio.gather(*tasks)
        corrected_responses.extend(batch_results)
        
    return corrected_responses

