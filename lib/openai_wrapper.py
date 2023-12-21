import openai


def call_openai(msgs: list[dict],
                client: openai.OPENAI_CLIENT,
                model_name: str) -> str:
    """
    Calls OpenAI model and provides the top response .
    Args:
        msgs: A list of dictionary in OpenAI message format {"role":<system/user/assistant>, "content":<content>}
        client: The OpenAI client object
        model_name: The model name to use

    Returns:
        res: Top response
    """
    return client.chat.completions.create(
        model=model_name,
        messages=msgs
    ).choices[0].message.content

