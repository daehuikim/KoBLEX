import json
import ast
import re
import os
from typing import List, Dict, Any, Optional, Tuple
import bm25s
import Stemmer


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data_list: List[Dict[str, Any]], output_path: str) -> None:
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in data_list:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def setup_retriever(context_file: str):
    """Setup BM25 retriever with context data."""
    context_list = []
    doc_ids = []
    
    with open(context_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line.strip())
            context_list.append(item['hierarchy'] + item['content'])
            doc_ids.append(i)
    
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(context_list, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    
    return retriever, context_list


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON object from text, handling various formats."""
    # Remove markdown code blocks
    clean = re.sub(r"```(?:json)?\n?|```", "", text)
    
    # Find JSON object
    depth = 0
    json_str = ""
    for ch in clean:
        if ch == "{":
            depth += 1
        if depth > 0:
            json_str += ch
        if ch == "}":
            depth -= 1
            if depth == 0:
                break
    
    # If no proper JSON found, use the whole cleaned text
    if not (json_str.startswith("{") and json_str.endswith("}")):
        json_str = clean
    
    # Try to parse JSON
    try:
        obj = json.loads(json_str)
    except Exception:
        try:
            obj = ast.literal_eval(json_str)
        except Exception:
            obj = {"_raw_output": clean.strip()}
    
    return obj


def avg_logprob(logprobs: List[Any]) -> Optional[float]:
    """Calculate average log probability from logprobs."""
    numeric = []
    for lp in logprobs:
        if isinstance(lp, dict):
            _, v = next(iter(lp.items()))
            numeric.append(v.logprob if hasattr(v, "logprob") else v)
        elif lp is not None:
            numeric.append(lp)
    
    if numeric:
        return sum(numeric) / len(numeric)
    return None


def decide_retrieval(tokens: List[str], logprobs: List[float], token_threshold: float = -0.70) -> Tuple[str, bool]:
    """Decide whether to perform retrieval based on log probabilities."""
    masked = []
    scores = []
    flag = False
    
    for tok, lp in zip(tokens, logprobs):
        scores.append(lp)
        if lp is not None and lp < token_threshold:
            masked.append("_")
            flag = True
        else:
            masked.append(tok)
    
    return "".join(masked), flag


def lp_to_list(lp_info: Any) -> Tuple[List[str], List[float]]:
    """Convert logprob info to lists of tokens and logprobs."""
    if lp_info:
        tokens = [tok.token for tok in lp_info.content]
        logprobs = [tok.logprob for tok in lp_info.content]
        return tokens, logprobs
    return [], []


def _mean(a: List[float]) -> float:
    """Calculate mean of a list."""
    return sum(a) / len(a) if a else 0.0
