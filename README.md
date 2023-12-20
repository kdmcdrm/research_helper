# research_helper
Language agent for helping with research.

## Plan:

### Process Research

Input -> Directory of files (or maybe Arxiv list if I can get it working)

Outputs:
1. Summaries: Markdown files summarizing paper
    1. Paper title
    2. Paper year and First Author
    3. Paper summary
2. Vectorstore

### Field Summary
Paper Summaries -> High Level Summary markdown file

### Field Expert Bot
Load the High Level summary and vector store and answer questions about
the field.


## Bonus Features:
 - Automatically determine common references and look them up.
 - Automatically load related papers from Connected Papers (needs account + Beautiful Soup?)
