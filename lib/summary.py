from tqdm import tqdm

from openai_wrapper import call_openai


def summarize_paper_refine(docs) -> str:
    """
    Creates a summary of a single paper using the "refine"
    method similar to langchain.chain.summarize refine option.
    For each document section it is given the option to refine the
    existing summary.
    """
    # Would use LangChains methods but they don't have an obvious way to override the template
    messages = [
        {"role": "system", "content": "You are a scientific research helper. You provide concise and accurate "
                                      "summaries that highlight the most important data"}
    ]
    sum_template = """Write a concise summary of the following section of scientific text. Focus on facts related to 
    what the researchers created and why, not on details of the researchers themselves: \n{text}\n CONCISE SUMMARY:"""

    # Create initial summary
    sum_messages = messages + [{"role": "user", "content": sum_template.format(text=docs[0].page_content)}]
    cur_sum = call_openai(sum_messages)

    # Refine summary
    for doc in tqdm(docs[1:], "Summarizing Pages"):
        refine_template = (
            "Your job is to produce a final summary of a scientific paper\n"
            "We have provided an existing summary up to a certain point: {current_summary}\n"
            "We have the opportunity to refine the existing summary"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary if needed."
            "If the context isn't useful, return the original summary."
            "SUMMARY TEXT:"
        )
        refine_messages = messages + \
            [{"role": "user", "content": refine_template.format(current_summary=cur_sum, text=doc.page_content)}]
        cur_sum = call_openai(refine_messages)

    return cur_sum
