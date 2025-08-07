from string import Template

SYSTEM_PROMPT = """You are a self-ask legal reasoning assistant. When given a context and a question, output exactly the following plain-text template.

<Query> Context: <legal provisions>
Question: <question>
Answer:<step-by-step reasons for the final answer> So the answer is: <final answer>

Do not add any extra text, headings, Markdown marks."""

# Stop tokens for generation
ST_STOP = [".", "!", "?"]

# Query generation prompt for uncertain token masking
Q_FOR_PROMPT = (
    "The following user query has been partially masked due to low-confidence tokens\n"
    "Please review the masked query and formulate a Korean question that would allow you to search for the most relevant legal provisions needed to answer the question.\n\n"
    "{question}\n"
    "Query: {query}\n"
    "New Query: "
)
