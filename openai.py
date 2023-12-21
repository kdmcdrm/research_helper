from create_summaries import OPENAI_CLIENT, MODEL_NAME


def call_openai(msgs: list[dict]) -> str:
    """
    Calls OpenAI model and provides the top response .
    Args:
        msgs: A list of dictionary in OpenAI message format {"role":<system/user/assistant>, "content":<content>}

    Returns:
        res: Top response
    """
    return OPENAI_CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=msgs
    ).choices[0].message.content
