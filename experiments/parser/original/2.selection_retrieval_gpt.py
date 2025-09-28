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
import bm25s
from tqdm.asyncio import tqdm
import re,torch
from sentence_transformers import CrossEncoder

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def load_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key

def normalize_law(s):
    return re.sub(r'[^가-힣A-Za-z0-9]', '', s)
    
def extract_law_name_q(s):
    base = s.split('(')[0].strip()
    base = normalize_law(base)
    LAW_RE = re.compile(
        r'([가-힣A-Za-z0-9]+?'
        r'(?:법(?:률)?|령|시행령|시행규칙))'
    )
    m = LAW_RE.search(base)
    if not m:
        return None
    raw = m.group(1).strip()
    return normalize_law(raw)

def retrieve_law(statute_path, input_path):
    raw_items = []
    with open(statute_path, encoding='utf-8') as f:
        for line in f:
            raw_items.append(json.loads(line))

    context_list = [item['hierarchy'] + item['content'] for item in raw_items]
    corpus_tokens = bm25s.tokenize(context_list, stopwords="en", show_progress=False)
    
    def extract_law_name(s):
        match = re.search(r'\d+조', s)
        if match:
            return normalize_law(s[:match.start()])
        return None
        
    law2tokens = {}
    law2global = {}
    for idx, item in enumerate(raw_items):
        law = extract_law_name(item['index'])
        law = normalize_law(law)
        if not law:
            continue
        text = item['hierarchy'] + item['content']
        law2tokens.setdefault(law, []).append(text)
        law2global.setdefault(law, []).append(idx)
    
    law2bm25 = {}
    for law, texts in law2tokens.items():
        tokens_list = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        retr = bm25s.BM25()
        retr.index(tokens_list, show_progress=False)
        law2bm25[law] = retr
        
    print(f"총 {len(law2bm25)}개의 법을 찾았습니다:")
    
    global_retriever = bm25s.BM25()
    global_retriever.index(corpus_tokens, show_progress=False)
    
    data_list=[]
    with open(input_path,'r',encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    predictions_top1 = []
    gold_answers=[]
    questions=[]
    candidates=[]
    origins=[]
    backgrounds=[]
    for item in data_list:
        sub_queries=item['subs']
        origins.append(sub_queries)
        question = item['question']
        backgrounds.append(item['background'])
        gold=[]
        for t in item['contexts']:
            gold.append(t['hierarchy']+t['content'])
        temp_candidates = []
        temp_prediction = []
        for query in sub_queries:
            law = extract_law_name_q(query)
            retr = global_retriever
            query_token = bm25s.tokenize([query], show_progress=False)
            local_idxs, _ = retr.retrieve(query_token, k=min(len(law2global.get(law, context_list)), 100), show_progress=False) 
            if law in law2global and retr is not global_retriever:
                gl_idxs = [law2global[law][i] for i in local_idxs[0]]
            else:
                gl_idxs = local_idxs[0]
            temp_candidates.append(gl_idxs)
            temp_prediction.append(context_list[gl_idxs[0]])
            
        predictions_top1.append(temp_prediction)
        gold_answers.append(gold)
        questions.append(question)
        candidates.append(temp_candidates)    
    return predictions_top1, candidates, gold_answers, questions, origins, backgrounds

def rerank2(questions, candidates, statute_path):
    raw_items = []
    with open(statute_path, encoding='utf-8') as f:
        for line in f:
            raw_items.append(json.loads(line))

    context_list = [item['hierarchy'] + item['content'] for item in raw_items]
    
    model = CrossEncoder('dragonkue/bge-reranker-v2-m3-ko', default_activation_function=torch.nn.Sigmoid(), device='cuda')
    reranked_results = []
    for q, sub_query_candidates in zip(questions, candidates):
        reranked_per_sub = []

        for cand_indices in sub_query_candidates:
            pair_batch = [(q, context_list[c]) for c in cand_indices]
            scores = model.predict(pair_batch,
                                   batch_size=50,
                                   show_progress_bar=False)

            scored_with_contexts = sorted(
                zip(scores, (ctx for _, ctx in pair_batch)),
                key=lambda x: x[0],
                reverse=True
            )
            ordered_contexts = [ctx for _, ctx in scored_with_contexts]
            reranked_per_sub.append(ordered_contexts)
        reranked_results.append(reranked_per_sub)

    return reranked_results

def create_prompts(data_list, prompt_path, statute_path):
    with open(prompt_path,'r',encoding='utf-8') as f:
        prompt_str = f.read()
        
    retrieval_only, candidates, gold_answers, questions, origins, backgrounds = retrieve_law(statute_path, "dummy")
    reranked = rerank2([item['question'] for item in data_list], candidates, statute_path)
    
    pick_prompt = Template(prompt_str)
    prompts = []
    counts = []
    
    for item, r in zip(data_list, reranked):
        background = item['background']
        question = item['question']
        for context_list in r:
            top_10 = context_list[:10]
            top_10_str = [f"{i}: {s}" for i, s in enumerate(top_10)]
            cand_str = "\n".join(top_10_str)
            prompt = pick_prompt.substitute(background=background, question=question, candidates=cand_str)
            prompts.append(prompt)
        counts.append(len(r))
    
    return reranked, prompts, counts, gold_answers, questions, origins, backgrounds

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
    
    reranked, prompts_list, counts, gold_answers, questions, origins, backgrounds = create_prompts(data_list, args.prompt_path, args.statute_path)
    
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
    
    top_10_predictions=[]
    idx = 0
    for sub_cnt, re in zip(counts, reranked):
        preds_per_q = []
        for i in range(sub_cnt):
            output = results[idx][0]
            answer = output.split("Answer:")[-1].strip()
            candidates = re[i]
            if answer.isdigit() and len(answer) == 1:
                answer = int(answer)
            else:
                answer = 0

            preds_per_q.append(candidates[answer])
            idx += 1                          
        top_10_predictions.append(preds_per_q)
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for bg, s, golds, question, pp in zip(backgrounds, top_10_predictions, gold_answers, questions, origins):
            output = {
                "background": bg,
                "question": question,
                "parametric_provisions": pp,
                "selected_provisions": s,
                "answers": golds
            }
            json.dump(output, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"Total LLM Call: {len(prompts_list)}")
    print(f"file saved at {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Input file path")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output file path")
    parser.add_argument("--statute_path", type=str, required=True,
                        help="Statute collection file path")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Prompt template file path")
    parser.add_argument("--model_name", type=str, 
                        help="Model to be used for generating context", default="gpt-4o")
    parser.add_argument("--raw_output_path", type=str,
                        help="Output path to store raw responses", default="raw_output.txt")
    parser.add_argument("--max_parallel_calls", type=int, default=20, help="Maximum parallel calls for the API.")

    args = parser.parse_args()
    main(args)
