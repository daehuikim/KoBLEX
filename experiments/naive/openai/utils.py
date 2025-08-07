import json
import ast
import re
import os
import asyncio
import aiohttp
from typing import List, Optional, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
import bm25s
from sentence_transformers import CrossEncoder
import torch


def load_api_key():
    """Load OpenAI API key from environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key


def normalize(text):
    """Remove leading and trailing whitespaces and replace newlines."""
    return text.strip().replace("\n", "")


def normalize_law(s):
    """Normalize law text by removing special characters."""
    return re.sub(r'[^가-힣A-Za-z0-9]', '', s)


def extract_law_name_q(s):
    """Extract law name from text."""
    base = s.split('(')[0].strip()
    base = normalize_law(base)
    m = LAW_RE.search(base)
    if not m:
        return None
    raw = m.group(1).strip()
    return normalize_law(raw)


def extract_first_list(txt: str) -> Optional[str]:
    """Extract the first list from text that starts with '[' and ends with ']'."""
    start = txt.find('[')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(txt[start:], start):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:             
                return txt[start:i + 1]
    return None


def _fix_common_typos(block: str) -> str:
    """Fix common typos in list blocks."""
    block = block.replace(':]', ']')
    if not block.rstrip().endswith(']'):
        block = block.rstrip(' :;,') + ']'
    if block.count('"') % 2 == 1:
        block = block[:-1] + '"' + block[-1]
    
    return block


def parse_list_block(block: str) -> List[str]:
    """Parse a list block using multiple parsing strategies."""
    block = _fix_common_typos(block)
    for parser in (json.loads, ast.literal_eval):
        try:
            return list(parser(block))
        except Exception:
            pass
    return quoted_items(block)


def escape_quotes(lst: List[str]) -> List[str]:
    """Escape quotes in a list of strings."""
    esc = []
    for s in lst:
        s = s.replace("\\", "\\\\")  
        s = s.replace('"', r'\"')
        s = s.replace("'", r"\'")
        esc.append(s)
    return esc


def quoted_items(text: str) -> List[str]:
    """Extract quoted items from text."""
    items = []
    buf   = []
    in_q  = False        
    qchar = None         
    esc   = False       

    for ch in text:
        if in_q:
            if esc:                 
                buf.append(ch)
                esc = False
            elif ch == '\\':        
                buf.append(ch)
                esc = True
            elif ch == qchar:        
                items.append(''.join(buf))
                buf.clear()
                in_q = False
            else:                   
                buf.append(ch)
        else:
            if ch in ('"', "'"):     
                in_q  = True
                qchar = ch
    return items


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


def get_reranker(model="dragonkue/bge-reranker-v2-m3-ko"):
    """Get a CrossEncoder reranker model."""
    model = CrossEncoder(model, default_activation_function=torch.nn.Sigmoid(), device='cuda')
    return model


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


# Regex patterns
LAW_RE = re.compile(
    r'([가-힣A-Za-z0-9]+?'
    r'(?:법(?:률)?|령|시행령|시행규칙))'
)

_LAW_NAME_RE = re.compile(
    r'^(.+?(?:법(?:률)?|령|규칙|규정|조례|시행령|시행규칙))'
    r'(?=\s*(?:별표|부칙)?\s*(?:제)?\d+조)'
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), before_sleep=print, retry_error_callback=lambda _: None)
async def get_completion(datapoint, model_name, session, semaphore):
    """Get completion from OpenAI API with retry logic."""
    async with semaphore:
        async with session.post("https://api.openai.com/v1/chat/completions", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {load_api_key()}"
        }, json={
            "model": model_name,
            "messages": [{"role": "user", "content": datapoint}],
            "temperature": 0.01,
            "max_tokens": 2000,
            "top_p": 0.9
        }) as resp:

            response_json = await resp.json()

            pred = response_json["choices"][0]['message']["content"]
            usage_container = int(response_json["usage"]["completion_tokens"])
            # Post-processing
            pred = pred.strip()
            return (pred, usage_container)


async def get_completion_list(datapoints, max_parallel_calls, model_name):
    """Get completions for a list of datapoints with parallel processing."""
    semaphore = asyncio.Semaphore(value=max_parallel_calls)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(10)) as session:
        return await asyncio.gather(*[get_completion(datapoint, model_name, session, semaphore) for datapoint in datapoints])
