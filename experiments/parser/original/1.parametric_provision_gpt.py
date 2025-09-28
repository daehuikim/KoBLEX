import pathlib, os
import logging
import argparse 
from string import Template
import asyncio
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from openai import OpenAI

import json 
import tiktoken
import ast

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def load_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key

def extract_first_list(txt: str) -> str | None:
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
    block = block.replace(':]', ']')
    if not block.rstrip().endswith(']'):
        block = block.rstrip(' :;,') + ']'
    if block.count('"') % 2 == 1:
        block = block[:-1] + '"' + block[-1]
    
    return block

def parse_list_block(block: str) -> list[str]:
    block = _fix_common_typos(block)
    for parser in (json.loads, ast.literal_eval):
        try:
            return list(parser(block))
        except Exception:
            pass
    return quoted_items(block)

def escape_quotes(lst: list[str]) -> list[str]:
    esc = []
    for s in lst:
        s = s.replace("\\", "\\\\")  
        s = s.replace('"', r'\"')
        s = s.replace("'", r"\'")
        esc.append(s)
    return esc

def quoted_items(text: str) -> list[str]:
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

def create_prompts(data_list, prompt_path):
    with open(prompt_path,'r',encoding='utf-8') as f:
        prompt_str = f.read()
    
    gc_prompt = Template(prompt_str)
    prompts = []
    
    for item in data_list:
        background = item['background']
        question = item['question']
        prompt = gc_prompt.substitute(background=background, question=question)
        prompts.append(prompt)
    
    return prompts

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), before_sleep=print, retry_error_callback=lambda _: None)
async def get_completion(datapoint, model_name, session, semaphore, headers):
    async with semaphore:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
            "model": model_name,
            "messages": [{"role": "user", "content": datapoint}],
            "temperature": 0.01,
            "max_tokens": 2000,
            "top_p": 0.9
        }) as resp:

            response_json = await resp.json()

            pred = response_json["choices"][0]['message']["content"]
            pred = pred.strip()
            return pred

async def get_completion_list(datapoints, max_parallel_calls, model_name, headers):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(10)) as session:
        return await tqdm.gather(*[get_completion(datapoint, model_name, session, semaphore, headers) for datapoint in datapoints])

def main(args):
    logger.info(args)
    OPENAI_API_KEY = load_api_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    encoding = tiktoken.get_encoding("cl100k_base")
    
    data_list=[]
    with open(args.input_path,'r',encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    prompts_list = create_prompts(data_list, args.prompt_path)
    
    token_count = 0
    for data in prompts_list:
        token_count += len(encoding.encode(data))

    logger.info(f"Total token count: {token_count}")
    cost= 5 * (token_count / 1e6) + 20 * ((token_count / 1e6) / 6)
    logger.info(f"Expected cost: {cost} USD")

    logger.info(f"Start inference..")
    results = asyncio.run(get_completion_list(prompts_list, args.max_parallel_calls, args.model_name, headers))
    
    with open(args.raw_output_path, "w") as f:
        for r, _ in results:
            f.write(r + "\n")
    
    with open(args.output_path, 'w',encoding='utf-8') as fw:
        for item, (result, tok) in zip(data_list, results):
            # Parse the result string into a list
            block = extract_first_list(result)
            if block is None:
                subs = [item["question"]]  # Use question as fallback
                print(f"No list found in result: {result}")
            else:
                subs = parse_list_block(block)
                subs = escape_quotes(subs)
                if not isinstance(subs, list):
                    subs = [subs]
            
            item["subs"] = subs
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Total LLM Call: {len(prompts_list)}")
    print(f"Wrote {len(data_list)} records with `subs` to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Input file path")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output file path")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Prompt template file path")
    parser.add_argument("--model_name", type=str, 
                        help="Model to be used for generating context", default="gpt-4o")
    parser.add_argument("--raw_output_path", type=str,
                        help="Output path to store raw responses", default="raw_output.txt")
    parser.add_argument("--max_parallel_calls", type=int, default=20, help="Maximum parallel calls for the API.")

    args = parser.parse_args()
    main(args)
