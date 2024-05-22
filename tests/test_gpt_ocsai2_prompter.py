from ocsai.train import GPT_Ocsai2_Prompter
import textwrap

# Test Initialization
def test_GPT_Ocsai2_Prompter_initialization():
    prompter = GPT_Ocsai2_Prompter()
    assert prompter is not None
    assert prompter.sys_msg_text is not None


# Test craft_prompt method
def test_craft_prompt_method():
    prompter = GPT_Ocsai2_Prompter()
    prompt = prompter.craft_prompt("Pants", "makeshift flag")
    ans = """
        ACTION: TAG THE ORIGINALITY OF A RESPONSE TO A CREATIVITY TEST.
        TASK: uses
        PROMPT: Pants
        RESPONSE: `makeshift flag`

        ## Details
        SCALE: 1-5, where 1 is `not original at all` and 5 is `extremely original`
        FORMAT: Return in the format of newline-separated `KEY:value` pairs, with the following fields:
        - `SCORE`: An originality score, 1-5
        - `CONFIDENCE`: A measure of confidence in the score, 1-3, or None.
        - `FLAGS`: A comma-separated list with content flags, such as: 'nonsense', 'violent', 'not practical'
        """
    ans = textwrap.dedent(ans).strip()
    assert prompt == ans


# Test craft_response method
def test_craft_response_method():
    prompter = GPT_Ocsai2_Prompter()
    response = prompter.craft_response(3)
    assert response == "SCORE: 3"


# Test prepare_example method
def test_prepare_example_method():
    prompter = GPT_Ocsai2_Prompter()
    example = prompter.prepare_example(
        "Pants", "makeshift flag", language=None, target=3.3
    )
    expected_output = [
        {
            "role": "system",
            "content": "You are a creativity judge, scoring tests of originality.",
        },
        {
            "role": "user",
            "content": "ACTION: TAG THE ORIGINALITY OF A RESPONSE TO A CREATIVITY TEST.\nTASK: uses\nPROMPT: Pants\nRESPONSE: `makeshift flag`\n\n## Details\nSCALE: 1-5, where 1 is `not original at all` and 5 is `extremely original`\nFORMAT: Return in the format of newline-separated `KEY:value` pairs, with the following fields:\n- `SCORE`: An originality score, 1-5\n- `CONFIDENCE`: A measure of confidence in the score, 1-3, or None.\n- `FLAGS`: A comma-separated list with content flags, such as: 'nonsense', 'violent', 'not practical'",
        },
        {"role": "assistant", "content": "SCORE: 3.3"},
    ]
    assert example == expected_output
