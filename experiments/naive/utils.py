import json
import os
import bm25s
from typing import List, Dict, Any


def load_api_key():
    """Load OpenAI API key from environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key


def load_jsonl(file_path: str) -> List[dict]:
    """Load data from a JSONL file."""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list


def save_jsonl(data_list: List[dict], output_path: str) -> None:
    """Save data to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in data_list:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_statute_retriever(path: str = "source/statute.jsonl"):
    """Get statute retriever with BM25 indexing."""
    raw_items = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            raw_items.append(json.loads(line))

    statute = [item['hierarchy'] + item['content'] for item in raw_items]
    corpus_tokens = bm25s.tokenize(statute, stopwords="en", show_progress=False)
    
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    
    return statute, retriever


def normalize(text):
    """Remove leading and trailing whitespaces and replace newlines."""
    return text.strip().replace("\n", "")


def extract_answer_from_completion(completion: str) -> str:
    """Extract answer from completion text."""
    if "Answer:" in completion:
        return completion.split("Answer:")[-1].strip()
    return completion.strip()
