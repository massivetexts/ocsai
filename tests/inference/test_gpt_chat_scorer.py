from ocsai.inference.gpt_chat_scorer import GPT_Chat_Scorer
from ocsai.train import GPT_Classic_Chat_Prompter, GPT_Ocsai2_Prompter

# Sample data for testing
ocsai1_chat_models = {
    "ocsai-chatgpt": "ft:gpt-3.5-turbo-1106:peter-organisciak::8fEKk0V6"
}
ocsai15_chat_models = {
    "ocsai-1.5": "ft:gpt-3.5-turbo-1106:peter-organisciak:ocsai-full-1-24:8d5RLryO"
}


# Test 1.0 Initialization
def test_gpt_chat_scorer_initialization_classic():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter()
    )
    assert chat_scorer is not None
    assert isinstance(chat_scorer.prompter, GPT_Classic_Chat_Prompter)


# Test 1.5 Initialization
def test_gpt_chat_scorer_initialization_ocsai15():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai15_chat_models, prompter=GPT_Ocsai2_Prompter()
    )
    assert chat_scorer is not None
    assert isinstance(chat_scorer.prompter, GPT_Ocsai2_Prompter)


# Test score method
def test_chat_scorer_score_method_classic():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter()
    )
    result = chat_scorer.score("Pants", "makeshift flag")
    assert result == [{"score": 3.3, "confidence": None, "flags": None, 'n': 1, 'type': 'top'}]


# Test score method
def test_chat_scorer_score_method_ocsai15():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai15_chat_models, prompter=GPT_Ocsai2_Prompter()
    )
    result = chat_scorer.score("Pants", "makeshift flag")
    assert result == [{"score": 3.0, "confidence": 2, "flags": None, 'n': 1, 'type': 'top'}]


# Test originality method
def test_chat_scorer_originality_method_classic():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter()
    )
    score = chat_scorer.originality("Pants", "makeshift flag")
    assert score == 3.3


# Test originality method
def test_chat_scorer_originality_method_ocsai15():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai15_chat_models, prompter=GPT_Ocsai2_Prompter()
    )
    score = chat_scorer.originality("Pants", "makeshift flag")
    assert score == 3.0


# Test originality_batch with questions instead of prompts.
def test_originality_batch_with_questions():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai15_chat_models, prompter=GPT_Ocsai2_Prompter()
    )
    results = chat_scorer.originality_batch(
        prompts=None,
        responses=["makeshift flag", "make a parachute"],
        questions=["What is a surprising use for pants?"] * 2,
        model="ocsai-1.5",
        raise_errs=True,
    )
    expected_results = [
        {"score": 3.0, "confidence": 3, "flags": None},
        {"score": 3.8, "confidence": 3, "flags": None},
    ]
    assert results == expected_results


# Test originality_batch method
def test_chat_scorer_originality_batch_method_classic():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter()
    )
    results = chat_scorer.originality_batch(
        ["Pants", "Pants"],
        ["makeshift flag", "make a parachute"],
        model="ocsai-chatgpt",
    )
    expected_results = [
        {"score": 3.3, "confidence": None, "flags": None},
        {"score": 3.3, "confidence": None, "flags": None},
    ]
    assert results == expected_results


# Test originality_batch method
def test_chat_scorer_originality_batch_method_ocsai15():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai15_chat_models, prompter=GPT_Ocsai2_Prompter()
    )
    results = chat_scorer.originality_batch(
        ["Pants", "Pants"],
        ["makeshift flag", "make a parachute"],
        model="ocsai-1.5",
    )
    expected_results = [
        {"score": 3.0, "confidence": 2, "flags": None},
        {"score": 3.0, "confidence": 2, "flags": None},
    ]
    assert results == expected_results


# Test originality_batch method
def test_originality_df_ocsai15():
    import pandas as pd

    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai15_chat_models, prompter=GPT_Ocsai2_Prompter()
    )
    df = pd.DataFrame(
        [["Pants", "makeshift flag"], ["Pants", "make a parachute"]],
        columns=["prompt_randname", "response_randname"],
    )
    df = chat_scorer.originality_df(
        df, prompt_col="prompt_randname", response_col="response_randname"
    )
    expected_results = {
        'prompt_randname': {0: 'Pants', 1: 'Pants'},
        'response_randname': {0: 'makeshift flag', 1: 'make a parachute'},
        'score': {0: 3.0, 1: 3.0},
        'confidence': {0: 2, 1: 3},
        'flags': {0: None, 1: None}
    }

    df_dict = df.to_dict()
    assert df_dict == expected_results


# Test crafting prompt
def test_chat_scorer_craft_prompt_classic():
    chat_scorer = GPT_Chat_Scorer(
        cache=None, model_dict=ocsai1_chat_models, prompter=GPT_Classic_Chat_Prompter()
    )
    prompt = chat_scorer.prompter.craft_prompt("Pants", "makeshift flag")
    assert prompt == "AUT Prompt:Pants\nResponse:makeshift flag"
