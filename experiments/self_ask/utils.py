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


def get_last_line(text: str) -> str:
    """Get the last line from text."""
    return text.strip().split("\n")[-1]


def extract_after_colon(line: str) -> str:
    """Extract text after colon."""
    return line.split(":", 1)[-1].strip().rstrip('.')


def extract_answer(generated: str) -> str:
    """Extract answer from generated text."""
    return extract_after_colon(get_last_line(generated))


def extract_question(generated: str) -> str:
    """Extract question from generated text."""
    return extract_after_colon(get_last_line(generated))


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


def get_statute_retriever(path: str = "collections3.jsonl"):
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
