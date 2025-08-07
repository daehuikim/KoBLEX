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
import ast
from sentence_transformers import CrossEncoder
import re,torch

llm_call=0
tok_gen=0
law_regex = re.compile(r'^(.*?)(?=\s*\d+조)')
LAW_RE = re.compile(
    r'([가-힣A-Za-z0-9]+?'
    r'(?:법(?:률)?|령|시행령|시행규칙))'
    )

_LAW_NAME_RE = re.compile(
    r'^(.+?(?:법(?:률)?|령|규칙|규정|조례|시행령|시행규칙))'
    r'(?=\s*(?:별표|부칙)?\s*(?:제)?\d+조)'
)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

collections_file = "collections3.jsonl"
raw_items = []
with open(collections_file, encoding='utf-8') as f:
    for line in f:
        raw_items.append(json.loads(line))

context_list = [item['hierarchy'] + item['content'] for item in raw_items]

logger = logging.getLogger(__name__)

def load_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key

def normalize(text):
    # remove leading and trailing whitespaces
    # replace "\n" with ""
    return text.strip().replace("\n", "")

def normalize_law(s):
    return re.sub(r'[^가-힣A-Za-z0-9]', '', s)
    
def extract_law_name_q(s):
    base = s.split('(')[0].strip()
    base = normalize_law(base)
    m = LAW_RE.search(base)
    if not m:
        return None
    raw = m.group(1).strip()
    return normalize_law(raw)

def retrieve_law():
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
    
    # load data
    gc_file = "gpt_results/get_context/gpt4o_gc.jsonl"
    data_list=[]
    with open(gc_file,'r',encoding='utf-8') as f:
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
            #retr = law2bm25.get(law, global_retriever)
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
    return predictions_top1, candidates, gold_answers, questions, origins

def rerank2(questions, candidates):
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

def make_qa_prompts2(data,qa_prompt):
    # find root question
    question = data['question']
    prompts= []
    
    #processed = data['prediction_noreranking']
    #processed = data['predictions_top1']
    processed = data['predictions_top10']
    
    context_str = "\n".join(processed)
    prompt = qa_prompt.substitute(question=question, context_str=context_str)
    prompts.append(prompt)
    combs=[processed]
    return prompts, combs

def complete_prompts(data_list,mode):    
    bakground_list=[]
    question_list=[]
    for data in data_list:
        question_list.append(data['question'])
        bakground_list.append(data['background'])
    
    prompts=[]    
    if mode==0:
        with open('prompts/0get_context.txt','r',encoding='utf-8') as f:
            prompt_str = f.read()
        
        gc_prompt = Template(prompt_str)
        
        for bg,q in zip(bakground_list,question_list):
            prompt = gc_prompt.substitute(background=bg,question=q)
            prompts.append(prompt)
        return prompts
    elif mode==1:
        with open('prompts/1pick.txt','r',encoding='utf-8') as f:
            prompt_str = f.read()
            
        retrieval_only, candidates,gold_answers, questions, origins = retrieve_law()
        reranked = rerank2(question_list,candidates)
        
        pick_prompt = Template(prompt_str)
        counts=[]
        for bg,q,r in zip(bakground_list,question_list,reranked):
            for item in r:
                top_10 = item[:10]
                top_10_str = [f"{i}: {s}" for i, s in enumerate(top_10)]
                cand_str = "\n".join(top_10_str)
                prompt = pick_prompt.substitute(background=bg,question=q,candidates=cand_str)
                prompts.append(prompt)
            counts.append(len(r))
        return retrieval_only, reranked, prompts, counts,gold_answers, questions, origins
    elif mode==2:
        with open('prompts/ours_sp.txt','r',encoding='utf-8') as f:
            prompt_str = f.read()
        qa_prompt = Template(prompt_str)
        data_list = []
        with open('gpt_results/pick/gpt4o_wolocal.jsonl','r',encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
                
        all_prompts=[]
        all_combinations=[]
        answer_counts=[]
        for entry in data_list:
            prompts, combos = make_qa_prompts2(entry,qa_prompt)
            answer_counts.append(len(prompts))
            all_prompts.extend(prompts)
            all_combinations.extend(combos)
        return all_prompts, all_combinations
               
        
    
    
    

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

def main(args):
    global llm_call
    global tok_gen
    logger.info(args)
    OPENAI_API_KEY = load_api_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    encoding = tiktoken.get_encoding("cl100k_base")
    
    gold_file="filtered_data.jsonl"
    gold_data_list=[]
    with open(gold_file,'r',encoding='utf-8') as f:
        for line in f:
            gold_data_list.append(json.loads(line))
    mode = 1
    if mode == 0:
        prompts_list = complete_prompts(gold_data_list,0)
    elif mode == 2:
        prompts_list, combination_list = complete_prompts(gold_data_list,2)
    elif mode == 1:
        ret_only,reranked,prompts_list,counts,gold_answers, questions, origins = complete_prompts(gold_data_list,1)
        reranked_top1 = [
            [sub[0] if sub else ""               
            for sub in per_question]            
            for per_question in reranked        
            ]
        
    
    token_count = 0
    for data in prompts_list:
        token_count += len(encoding.encode(data))

    logger.info(f"Total token count: {token_count}")
    cost= 5 * (token_count / 1e6) + 20 * ((token_count / 1e6) / 6)
    logger.info(f"Expected cost: {cost} USD")


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), before_sleep=print, retry_error_callback=lambda _: None)
    async def get_completion(datapoint, model_name, session, semaphore):
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
                # Post-processing
                pred = pred.strip()
                return (pred, usage_container)

    async def get_completion_list(datapoints, max_parallel_calls):
        semaphore = asyncio.Semaphore(value=max_parallel_calls)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(10)) as session:
            return await tqdm.gather(*[get_completion(datapoint, args.model_name, session, semaphore) for datapoint in datapoints])

    logger.info(f"Start inference..")
    results = asyncio.run(get_completion_list(prompts_list, args.max_parallel_calls))
    llm_call += len(prompts_list)
    with open(args.raw_output_path, "w") as f:
        for r, _ in results:
            f.write(r + "\n")
    if mode == 0:
        with open(args.output_path, 'w',encoding='utf-8') as fw:
            for entry,(r,tok),c in zip(gold_data_list,results,combination_list):
                question_text = entry['question']
                answer_list=[{'answer':r.split("Answer:")[-1].strip(),"logp":0.0,"comb":c}]
                record = {
                    "question": question_text,
                    "answers": answer_list
                }
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"Saved detailed answers to {args.output_path}")

    elif mode == 1:
        top_10_predictions=[]
        idx = 0
        for sub_cnt,re in zip(counts,reranked):                    # 각 질문마다
            preds_per_q = []
            for i in range(sub_cnt):              # 그 질문의 sub‑query 수만큼
                output = results[idx][0]
                temp_tok = results[idx][1]
                tok_gen +=temp_tok
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
            for p, preds, preds10, golds, question,o in zip(ret_only,reranked_top1, top_10_predictions,gold_answers,questions,origins):
                output = {
                    "generated": o,
                    "question": question,
                    "prediction_noreranking":p,
                    "predictions_top1": preds,
                    "predictions_top10": preds10,
                    "answers": golds
                }
                json.dump(output, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"Total LLM Call: {llm_call}")
        print(f"Total generated tokens: {tok_gen}")
        print(f"file saved at {args.output_path}")             
    
    elif mode == 2:
        # results_processd=[]
        # for item in results:
        #     block = extract_first_list(item)
        #     if block is None:
        #         results_processd.append([])
        #         continue
        #     subs = parse_list_block(block)
        #     subs = escape_quotes(subs)
        #     if not isinstance(subs, list): 
        #         subs = [subs]
        #     results_processd.append(subs)
        
        # Save the real results        
        with open(args.output_path, "w") as fout:
            for item, (result,tok) in zip(gold_data_list,results):
                item["subs"]=result
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                tok_gen += tok
        
        print(f"Total LLM Call: {llm_call}")
        print(f"Total generated tokens: {tok_gen}")
        print(f"Wrote {len(gold_data_list)} records with `subs` to {args.output_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, 
                        help="Model to be used for generating context", default="gpt-4o")
    parser.add_argument("--output_path", type=str,
                        help="Output path to store synthesized datapoint", default="gpt_results/baselines/call/gpt4o_parser_call.jsonl")
    parser.add_argument("--raw_output_path", type=str,
                        help="Output path to store synthesized datapoint", default="gpt_results/baselines/call/gpt4o_parser_call.txt")
    parser.add_argument("--max_parallel_calls", type=int, default=20, help="Maximum parallel calls for the API.")

    args = parser.parse_args()
    main(args)
