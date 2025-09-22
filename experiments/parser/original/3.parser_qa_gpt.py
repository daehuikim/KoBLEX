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
from tqdm.asyncio import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def load_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key

def create_prompts(data_list):
    with open('prompt-path','r',encoding='utf-8') as f:
        prompt_str = f.read()
    
    qa_prompt = Template(prompt_str)
    prompts = []
    
    for item in data_list:
        question = item['background'] + item['question']
        provisions = item.get('selected_provisions', [])
        prompt = qa_prompt.substitute(
            question=question,
            context_str="\n".join(provisions)
        )
        prompts.append(prompt)
    
    return prompts

def process_completions(completions):
    answers = []
    
    for text in completions:
        answer = text.split("Answer:")[-1].split("</think>")[0].strip()
        answers.append(answer)
    
    return answers

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
            usage_container = int(response_json["usage"]["completion_tokens"])
            pred = pred.strip()
            return (pred, usage_container)

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
    
    input_file = "input-path"
    data_list=[]
    with open(input_file,'r',encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    prompts = create_prompts(data_list)
    
    token_count = 0
    for data in prompts:
        token_count += len(encoding.encode(data))

    logger.info(f"Total token count: {token_count}")
    cost= 5 * (token_count / 1e6) + 20 * ((token_count / 1e6) / 6)
    logger.info(f"Expected cost: {cost} USD")

    logger.info(f"Start inference..")
    outputs = asyncio.run(get_completion_list(prompts, args.max_parallel_calls, args.model_name, headers))
    
    with open(args.raw_output_path, "w") as f:
        for r, _ in outputs:
            f.write(r + "\n")
    
    answers = process_completions(outputs)
    
    output_path = "output-path"
    with open(output_path, 'w', encoding='utf-8') as fw:
        for item, provisions, answer in zip(data_list, [item.get('selected_provisions', []) for item in data_list], answers):
            record = {
                "background": item['background'],
                "question": item['question'],
                "answer": answer,
                "provisions": provisions
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved detailed answers to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, 
                        help="Model to be used for generating context", default="gpt-4o")
    parser.add_argument("--raw_output_path", type=str,
                        help="Output path to store raw responses", default="raw_output.txt")
    parser.add_argument("--max_parallel_calls", type=int, default=20, help="Maximum parallel calls for the API.")

    args = parser.parse_args()
    main(args)
