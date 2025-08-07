import json
import ast
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from sentence_transformers import CrossEncoder
import bm25s


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


def get_statute_retriever(path: Path = Path("source/statute.jsonl")):
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
