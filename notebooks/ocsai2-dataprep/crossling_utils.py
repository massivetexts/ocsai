import pandas as pd
import yaml
import textwrap
import asyncio
from typing import List, Dict, Any, cast
from IPython.display import display
import nest_asyncio
from tqdm.asyncio import tqdm
import re

# Enable nested asyncio for Jupyter
nest_asyncio.apply()

def series_to_yaml(series: pd.Series):
    data_dict = series.where(series.notna(), "[fill-in-the-blank]").to_dict()
    
    # Use safe_dump to avoid any potential YAML parsing issues
    yaml_output = yaml.safe_dump(data_dict, 
                                sort_keys=False, 
                                allow_unicode=True,
                                default_style='"',
                                default_flow_style=False)
    return yaml_output

def create_translate_prompt(row: pd.Series, 
                            translate_cols: list[str],
                            include_cols: list[str],
                            all_langs: list[str],
                            data: pd.DataFrame):
    for lang in all_langs:
        for col in translate_cols:
            col = col.replace('original_', '')
            if lang == row['original_language']:
                row[f'{col}_{lang}'] = row['original_'+col]
            else:
                row[f'{col}_{lang}'] = pd.NA

    row_yaml = series_to_yaml(row)
    examples = make_examples(data, include_cols +translate_cols, all_langs)
    system_prompt = textwrap.dedent(f"""
        You are a translator, translating a YAML object from the `original` language into a set of specified languages.
    """)

    user_prompt = textwrap.dedent("""
        Translate the YAML at the bottom into the specified languages, from the `original` language.
                                    
        Respond with the full YAML object, wrapped in triple backticks. DO NOT include any other text.

        # Example syntax for other data points (not necessarily the same question)
        
        {examples}

        # Data to translate
        
        ```
        {row_yaml}
        ```

        # Format     
                                  
        Return *all* rows, replacing and translating the ['fill-in-the-blank'] values. Some common bugs to avoid:
        - Escape quotation marks inside a value - `\\"` - or switch the outer quotes from double to single when necessary
        - If you start a Polish response with 'ich' (their), don't repeat the word over and over again. A similar bug to avoid with Dutch: 'dozen' shouldn't be repeated endlessly.

    """).format(examples=examples.strip(), row_yaml=row_yaml)
    return system_prompt, user_prompt

def make_examples(data: pd.DataFrame, subset_cols: list[str], all_langs: list[str]):
    '''Sample one example item per language.'''

    string = ""
    for lang in all_langs:
        lang_subset = data.sample(frac=1).query(f'original_language == "{lang}"').head(1)
        # Account for if a dataset was passed without all languages.
        if lang not in lang_subset.original_language.unique():
            continue
        # remove 'original_' from the column names
        lang_subset.columns = lang_subset.columns.str.replace('original_', '')
        string += series_to_yaml(lang_subset[['language'] + [c.replace('original_', '') for c in subset_cols]].iloc[0])
        string += "\n"
    
    return string

def fix_escaping(content: str) -> str:
    """
    Escape unescaped quotation marks in the text.
    This helps handle special characters like when normal quotation marks 
    are used in place of Hebrew Gershayim marks.
    """
    replacements = []
    for line in content.split('\n'):
        if ":" not in line:
            continue
        k, v = line.split(':', 1)
        v = v.strip()
        
        # change double-quotes to single-quotes
        if v.startswith('"') and v.endswith('"') and v.count('"') > 2:
            rep = v[1:-1].replace('"', '\\"').replace("'", "\\'")
            replacements.append((v, f"'{rep}'"))

        # change single-quotes to double-quotes
        elif v.startswith("'") and v.endswith("'") and v.count("'") > 2:
            rep = v[1:-1].replace("'", "\\'").replace('"', '\\"')
            replacements.append((v, f'"{rep}"'))

        ## elif
        # v_stripped = v.strip().strip('"').replace('\\"', '"')
        #if v_stripped.count('"'):
        #    replacements.append((v_stripped, v_stripped.replace('"', '\\"')))

    for v, replacement in replacements:
        content = content.replace(v, replacement)
    return content

def parse_chat_response_to_object(response, translate_cols: list[str], all_langs: list[str]):
    # Extract YAML content from response
    content = response.choices[0].message.content
    yaml_content = content

    # Try to extract content between last set of triple backticks
    backtick_sections = content.split('```')
    if len(backtick_sections) > 1:
        yaml_content = backtick_sections[-2]

    # Escape unescaped quotation marks before parsing
    x = yaml_content
    yaml_content = fix_escaping(yaml_content)

    # Parse YAML and validate expected columns
    try:
        parsed = yaml.safe_load(yaml_content)
        
        # Expected columns based on context
        required_cols = (['type'] + 
                        translate_cols + 
                        [f'{c.replace("original_", "")}_{lang}' 
                        for c in translate_cols 
                            for lang in all_langs])
        
        # Verify all required columns exist
        missing_cols = [col for col in required_cols if col not in parsed]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        print("YAML content that failed to parse:")
        print(yaml_content)
    except ValueError as e:
        print(f"Validation error: {e}")
    return parsed

async def translate_with_llm(async_client, row: pd.Series, 
                             translate_cols: list[str], 
                             all_langs: list[str], 
                             data_for_examples: pd.DataFrame, 
                             include_cols: list[str] = [],
                             retries=3,
                             model: str = 'gpt-4o-mini',
                             temp_backoff=0.5):
    sysprompt, prompt = create_translate_prompt(row, translate_cols, include_cols, all_langs, data_for_examples)

    temperature = 0.0
    for i in range(retries):
        try:
            response = await async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sysprompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            parsed = parse_chat_response_to_object(response, translate_cols, all_langs)
            return parsed
        except (yaml.YAMLError, ValueError) as e:
            if i == retries - 1:
                raise e
            temperature += temp_backoff
    return None

async def translate_batch(async_client,
                         rows: List[pd.Series],
                         translate_cols: List[str], 
                         all_langs: List[str], 
                         data_for_examples: pd.DataFrame, 
                         model: str = 'gpt-4o-mini',
                         max_concurrent: int = 12,
                         include_cols: List[str] = []) -> List[Dict[str, Any]]:
    """
    Translate a batch of rows concurrently.
    
    Args:
        async_client: Async OpenAI client
        rows: List of rows to translate
        translate_cols: Columns to translate
        all_langs: All languages to translate to
        data_for_examples: Full dataset for examples
        max_concurrent: Maximum number of concurrent API calls
        include_cols: Extra columns to include in the examples
    Returns:
        List of translated results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def translate_with_semaphore(row: pd.Series):
        async with semaphore:
            try:
                result = await translate_with_llm(async_client, row, translate_cols, 
                                                  all_langs, data_for_examples, 
                                                  include_cols=include_cols, model=model)
                if result is not None:
                    return result
                else:
                    print(f"Translation failed for row: {row.name if hasattr(row, 'name') else 'unknown'}")
                    return None
            except Exception as e:
                print(f"Error translating row: {e}")
                return None
    
    tasks = [translate_with_semaphore(row) for row in rows]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out any exceptions and None values, then cast to the correct type
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    
    return cast(List[Dict[str, Any]], valid_results)

async def translate_dataframe(async_client, 
                            df: pd.DataFrame, 
                            translate_cols: List[str], 
                            all_langs: List[str], 
                            batch_size: int = 10, 
                            model: str = 'gpt-4o-mini',
                            max_concurrent: int = 12,
                            include_cols: List[str] = []
                            ) -> pd.DataFrame:
    """
    Translate an entire dataframe in batches.
    
    Args:
        async_client: Async OpenAI client
        df: DataFrame to translate
        translate_cols: Columns to translate
        all_langs: All languages to translate to
        batch_size: Number of rows to process in each batch
        max_concurrent: Maximum number of concurrent API calls per batch
        include_cols: Extra columns to include in the examples
    Returns:
        DataFrame with translated results
    """
    results = []
    if batch_size < max_concurrent:
        batch_size = max_concurrent
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    # Process dataframe in batches
    for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Batches"):
        batch = df.iloc[i:i+batch_size]
        
        batch_results = await translate_batch(async_client, 
                                           [row for _, row in batch.iterrows()], 
                                           translate_cols, 
                                           all_langs, 
                                           df, 
                                           model=model,
                                           max_concurrent=max_concurrent,
                                           include_cols=include_cols)
        results.extend(batch_results)
        
        # Optional: Add a small delay between batches to avoid rate limits
        await asyncio.sleep(1)
    
    # Convert results back to DataFrame
    return pd.DataFrame(results)