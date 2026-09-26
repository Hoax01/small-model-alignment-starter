ALPACA_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def format_prompt(instruction: str) -> str:
    return ALPACA_PROMPT.format(instruction=instruction.strip())


def format_sft_text(instruction: str, response: str, eos_token: str) -> str:
    return f"{format_prompt(instruction)}{response.strip()}{eos_token}"

