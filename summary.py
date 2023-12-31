from tqdm import tqdm
from langchain.schema import Document
from agents import LLMAgent


def summarize_paper(docs: list[Document],
                    agent: LLMAgent,
                    method: str
                    ) -> str:
    """
    Summarizes a paper uses one of the specified methods.

    Args:
        docs: A set of LangChain documents
        agent: An LLM Agent
        method: The method to use to produce summary, "reduce" and "refine" currently supported.

    Returns:
        summary: The text of the summary
    """
    if method == "refine":
        return _summarize_paper_refine(docs, agent)
    elif method == "reduce":
        return _summarize_paper_map_reduce(docs, agent)
    else:
        ValueError(f"Method {method} unrecognized.")


def _summarize_paper_refine(docs: list[Document],
                            agent: LLMAgent) -> str:
    """
    Creates a summary of a paper by producing an initial summary and then allowing each
    document to potentially refine it.
    Args:
        docs: A set of LangChain documents
        agent: An LLM Agent
    Returns:
        summary: The summary text
    """
    # Would use LangChains methods but they don't have an obvious way to override the template
    sum_template = """Write a concise summary of the following section of scientific text. Focus on facts related to 
    what the researchers created and why, not on details of the researchers themselves: \n{text}\n CONCISE SUMMARY:"""
    cur_sum = agent.call_no_history(sum_template.format(text=docs[0].page_content))

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
        cur_sum = agent.call_no_history(refine_template.format(current_summary=cur_sum, text=doc.page_content))

    return cur_sum


def _summarize_paper_map_reduce(docs: list[Document],
                                agent: LLMAgent) -> str:
    """
    Creates a summary of a single paper by summarizing each doc and then creating
    a final summary as a summary of summaries.
    Args:
        docs: A set of LangChain documents
        agent: An LLM Agent
    """
    # Page summaries
    page_sum_template = """Write a short summary of the following section of scientific text. Focus on facts 
    related to what the researchers created and why, not on details of the researchers themselves: \n{text}\n CONCISE 
    SUMMARY:"""
    summaries = []
    for doc in tqdm(docs, "Summarizing Pages"):
        summaries.append(agent.call_no_history(page_sum_template.format(text=doc.page_content)))

    # Create final summary
    final_sum_template = """
        Given the following summaries of scientific text produce a short summary of no more than 3 paragraphs. Focus
        on the novel contributions of the authors. \n {sums} \n CONCISE_SUMMARY:
    """
    final_summary = agent.call_no_history(final_sum_template.format(sums="\n\n".join(summaries)))
    return final_summary
