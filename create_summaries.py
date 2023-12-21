from pathlib import Path
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains.summarize import load_summarize_chain
import openai
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)
_ = load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Environment must define the OPENAI_API_KEY.")
OPENAI_CLIENT = openai.OpenAI()
MODEL_NAME = "gpt-4"


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
        )
        refine_messages = messages + \
            [{"role": "user", "content": refine_template.format(current_summary=cur_sum, text=doc.page_content)}]
        cur_sum = call_openai(refine_messages)

    return cur_sum


def create_paper_summaries(papers_dir: str) -> dict[str,str]:
    """
    Creates a set of summaries of papers in the directory, in a /summaries
    subfolder. Only .pdf is currently supported.
    Args:
        papers_dir: Path to the folder of .pdfs
    Returns:
        summaries: A list of text summaries of the paper content
    """
    logger.info("---- Create Summaries ----")
    papers_dir = Path(papers_dir)
    paper_paths = [str(x) for x in papers_dir.glob("*.pdf")]
    if len(paper_paths) == 0:
        raise ValueError("No .pdfs found in {str(papers_dir)}, please add papers for summarization.")

    logger.info(f"Running summarization for {len(paper_paths)} papers in {str(papers_dir)}")
    sum_dir = papers_dir / "summaries"
    sum_dir.mkdir(exist_ok=True)

    summaries = {}
    for paper_path in tqdm(paper_paths, "Summarizing Papers"):
        paper_name = Path(paper_path).stem
        sum_path = sum_dir / (paper_name + "_summary.md")

        # Only generate summary if it doesn't already exist
        if not sum_path.exists():
            loader = PyMuPDFLoader(str(paper_path))
            summary = summarize_paper_refine(loader.load())
            # Save summary to disk
            logger.debug(f"Saving summary to {sum_path}")
            with open(sum_path, "w", encoding="utf8") as fh:
                fh.write(summary)
        else:
            with open(sum_path, "r") as fh:
                summary = fh.read()

        # Store to dictionary for later reference
        summaries[paper_name] = summary

    return summaries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_paper_summaries("./papers")
