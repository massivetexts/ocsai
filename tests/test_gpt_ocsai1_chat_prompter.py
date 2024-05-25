from ocsai.train import Ocsai1_Chat_Prompter


# Test Initialization
def test_gpt_classic_chat_prompter_initialization():
    prompter = Ocsai1_Chat_Prompter()
    assert prompter is not None
    assert prompter.sys_msg_text is not None


# Test craft_prompt method
def test_craft_prompt_method():
    prompter = Ocsai1_Chat_Prompter()
    prompt = prompter.craft_prompt('Pants', 'makeshift flag')
    assert prompt == 'AUT Prompt:Pants\nResponse:makeshift flag'


# Test craft_response method
def test_craft_response_method():
    prompter = Ocsai1_Chat_Prompter()
    response = prompter.craft_response(3)
    assert response == '30'


# Test prepare_example method
def test_prepare_example_method():
    prompter = Ocsai1_Chat_Prompter()
    example = prompter.prepare_example('Pants', 'makeshift flag', language=None, target=3.3)
    expected_output = [
        {'role': 'system', 'content': 'You score originality in the alternate uses task.'},
        {'role': 'user', 'content': 'AUT Prompt:Pants\nResponse:makeshift flag'},
        {'role': 'assistant', 'content': '33'}
    ]
    assert example == expected_output
