from tqdm import tqdm
import openai
from lib.openai_wrapper import call_openai
from langchain.schema import Document


def summarize_paper(docs: list[Document],
                    client: openai.Client,
                    model_name: str,
                    method: str
                    ) -> str:
    """
    Summarizes a paper uses one of the specified methods.

    Args:
        docs: A set of LangChain documents
        client: An openai Client
        model_name: The name of the model to use.
        method: The method to use to produce summary, "reduce" and "refine" currently supported.

    Returns:
        summary: The text of the summary
    """
    if method == "refine":
        return _summarize_paper_refine(docs, client, model_name)
    elif method == "reduce":
        return _summarize_paper_map_reduce(docs, client, model_name)
    else:
        ValueError(f"Method {method} unrecognized.")


def _summarize_paper_refine(docs: list[Document],
                            client: openai.Client,
                            model_name: str) -> str:
    """
    Creates a summary of a paper by producing an initial summary and then allowing each
    document to potentially refine it.
    Args:
        docs: A set of LangChain documents
        client: An openai Client
        model_name: The name of the model to use.
    Returns:
        summary: The summary text.
    """
    # Would use LangChains methods but they don't have an obvious way to override the template
    # ToDo: Improve openai_caller and remove OpenAI specific code here
    messages = [
        {"role": "system", "content": "You are a scientific research helper. You provide concise and accurate "
                                      "summaries that highlight the most important data"}
    ]
    sum_template = """Write a concise summary of the following section of scientific text. Focus on facts related to 
    what the researchers created and why, not on details of the researchers themselves: \n{text}\n CONCISE SUMMARY:"""

    # Create initial summary
    sum_messages = messages + [{"role": "user", "content": sum_template.format(text=docs[0].page_content)}]
    cur_sum = call_openai(sum_messages, client, model_name)

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
                          [{"role": "user",
                            "content": refine_template.format(current_summary=cur_sum, text=doc.page_content)}]
        cur_sum = call_openai(refine_messages, client, model_name)

    return cur_sum


def _summarize_paper_map_reduce(docs: list[Document],
                                client: openai.Client,
                                model_name: str) -> str:
    """
    Creates a summary of a single paper by summarizing each doc and then creating
    a final summary as a summary of summaries.
    Args:
        docs: A set of LangChain documents
        client: An openai Client
        model_name: The name of the model to use.
    """
    # Would use LangChains methods but they don't have an obvious way to override the template
    messages = [
        {"role": "system", "content": "You are a scientific research helper. You provide concise and accurate "
                                      "summaries that highlight the most important data."}
    ]

    # Page summaries
    page_sum_template = """Write a short summary of the following section of scientific text. Focus on facts 
    related to what the researchers created and why, not on details of the researchers themselves: \n{text}\n CONCISE 
    SUMMARY:"""
    summaries = []
    for doc in tqdm(docs, "Summarizing Pages"):
        sum_message = {"role": "user", "content": page_sum_template.format(text=doc.page_content)}
        summaries.append(call_openai(messages + [sum_message], client, model_name))

    # Create final summary
    final_sum_template = """
        Given the following summaries of scientific text produce a short summary of no more than 3 paragraphs. Focus
        on the novel contributions of the authors. \n {sums} \n CONCISE_SUMMARY:
    """
    sum_message = {"role": "user", "content": final_sum_template.format(sums="\n\n".join(summaries))}
    final_summary = call_openai(messages + [sum_message], client, model_name)

    return final_summary
