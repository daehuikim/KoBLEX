import copy
import json
import random
import re
import argparse

from pathlib import Path
from tqdm import tqdm

import openai
from qg_utils import load_law_articles

# Constants
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
PROMPT_FILES = {
    "qg_pair_1hop": PROMPTS_DIR / "qg_pair_1hop.txt",
    "qg_pair_mhop": PROMPTS_DIR / "qg_pair_mhop.txt",
    "scenario": PROMPTS_DIR / "qg_scenario.txt",
}


class InvalidAPIresult(Exception):
    pass


def load_prompts() -> dict:
    prompts = {}
    for key, path in PROMPT_FILES.items():
        with open(path, 'r', encoding='utf-8') as f:
            prompts[key] = f.read().strip()
    return prompts



class LawQAGenerator:
    def __init__(self, qg_type: str, law_data: dict, llm: str):
        self.qg_type = qg_type
        self.llm = llm
        self.prompts = load_prompts()
        self.max_token = 1024
        self.temperature = 1.0
        self.top_p = 0.9
        self.qa_data = []

        if qg_type == "random":
            self.law_data = {k: v for k, v in law_data.items() if '호' not in k}
        else:
            self.law_data = law_data

    def api_call(self, prefix: str, prompt: str) -> str:
        response = openai.chat.completions.create(
            model=self.llm,
            messages=[{"role": "system", "content": prefix}, {"role": "user", "content": prompt}],
            max_tokens=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def extract_qa_context(text: str) -> dict:
        return {
            "question": re.search(r"question:\s*(.+?)\n", text, re.DOTALL).group(1).strip() if re.search(r"question:",
                                                                                                         text) else None,
            "answer": re.search(r"answer:\s*(.+?)\n(?=selected_context:|$)", text, re.DOTALL).group(
                1).strip() if re.search(r"answer:", text) else None,
            "context_sent_idx": int(re.findall(r"sent(\d+):", text)[0]) if len(
                re.findall(r"sent(\d+):", text)) == 1 else None,
        }

    @staticmethod
    def extract_qa_with_background(text: str) -> dict:
        return {
            "background_fact": re.search(r"background_fact:\s*(.+?)\n(?=question:)", text, re.DOTALL).group(
                1).strip() if re.search(r"background_fact:", text) else None,
            "question": re.search(r"question:\s*(.+?)\n(?=answer:)", text, re.DOTALL).group(1).strip() if re.search(
                r"question:", text) else None,
            "answer": re.search(r"answer:\s*(.+)", text, re.DOTALL).group(1).strip() if re.search(r"answer:",
                                                                                                  text) else None,
        }

    def pick_random_context(self, window_size: int) -> dict:
        if window_size >= len(self.law_data):
            return self.law_data
        keys = list(self.law_data.keys())
        start = random.randint(0, len(keys) - window_size)
        return {k: self.law_data[k] for k in keys[start:start + window_size]}

    def format_context(self, context: dict) -> str:
        return "".join(f"sent{i+1}: {v['hierarchy']} | {v['content']}\n" for i, v in enumerate(context.values()))

    def qa_pair_generation(self, used_ctx, remain_ctx, prev_pair=None):
        """
        Generate a question-answer pair based on the provided context.
        If used_ctx is empty, it generates a 1-hop question-answer pair.
        If used_ctx is not empty, it generates a multi-hop question-answer pair.
        """
        context_str = self.format_context(remain_ctx)
        if not used_ctx:
            prompt = f"context:\n{context_str}"
            prefix = self.prompts["qg_pair_1hop"]
        else:
            prompt = f"question:{prev_pair['question']}\nanswer:{prev_pair['answer']}\n" \
                     f"current_context:\n{self.format_context(used_ctx)}\n" \
                     f"remain_context:\n{context_str}"
            prefix = self.prompts["qg_pair_mhop"]

        parsed = self.extract_qa_context(self.api_call(prefix, prompt))
        if all(parsed.values()):
            try:
                selected_key = list(remain_ctx.keys())[parsed['context_sent_idx'] - 1]
                used_ctx[selected_key] = remain_ctx[selected_key]
                remain_ctx.pop(selected_key)
                return used_ctx, remain_ctx, parsed["question"], parsed["answer"]
            except Exception:
                raise InvalidAPIresult("Invalid context index")
        raise InvalidAPIresult("QA extraction failed")

    def scenario_generation(self, used_ctx, question, answer) -> dict:
        """
        Generate a background scenario based on the used context, question, and answer.
        The used_ctx should be a dictionary with context IDs as keys and their details as values.
        """
        context_str = self.format_context(used_ctx)
        prompt = f"question: {question}\nanswer: {answer}\ncontext: {context_str}"
        parsed = self.extract_qa_with_background(self.api_call(self.prompts["scenario"], prompt))
        if all(parsed.values()):
            return parsed
        raise InvalidAPIresult("Scenario extraction failed")

    def mhop_qa_generate(self, num_qa_pairs: int):
        """
        Generate multi-hop question-answer pairs based on the law data.
        """
        max_hop = 3
        used_pairs = []
        created = 0

        if self.qg_type == "case":
            filtered_cases = [c for c in self.law_data if len(c['context']) >= 2]
            num_qa_pairs = min(num_qa_pairs, len(filtered_cases))

        for i in tqdm(range(num_qa_pairs)):
            try:
                used_ctx, prev_pair = {}, {}
                remain_ctx = (filtered_cases[i]['context'] if self.qg_type == "case" else self.pick_random_context(30))
                case_cid = filtered_cases[i]['id'] if self.qg_type == "case" else "Random"

                while remain_ctx and len(used_ctx) <= max_hop:
                    used_ctx, remain_ctx, q, a = self.qa_pair_generation(used_ctx, remain_ctx, prev_pair)
                    scenario = self.scenario_generation(used_ctx, q, a)

                    if list(used_ctx.keys()) in used_pairs:
                        break
                    used_pairs.append(list(used_ctx.keys()))
                    self.qa_data.append({
                        "id": f"qa_{i}_{len(used_ctx)}hop_{created}",
                        "question": q,
                        "answer": a,
                        "background_scenario": scenario['background_fact'],
                        "question_scenario": scenario['question'],
                        "answer_scenario": scenario['answer'],
                        "context": copy.deepcopy(used_ctx),
                        "case_cid": case_cid,
                        "mhop": len(used_ctx)
                    })
                    prev_pair = {"question": q, "answer": a}
                    created += 1

            except InvalidAPIresult:
                continue

        print(f"Total {created} QA pairs generated.")

    def save_to_json(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.qa_data, f, ensure_ascii=False, indent=4)
        print(f"Saved QA data to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Multi-hop QA Generation")
    parser.add_argument('--llm', default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument('--case_path')
    parser.add_argument('--collection_path')
    parser.add_argument('--save_path')
    parser.add_argument('--num_qa_pairs', type=int, required=True)
    parser.add_argument('--qg_type', choices=["case", "random"], default="case")
    args = parser.parse_args()

    if args.qg_type == "case":
        assert args.case_path, "case_path is required for case-based generation"
        law_data = json.load(open(args.case_path, encoding='utf-8'))
        default_name = Path(args.case_path).stem
    else:
        assert args.collection_path, "collection_path is required for random generation"
        law_data = load_law_articles(args.collection_path)
        default_name = Path(args.collection_path).stem

    save_path = args.save_path or f"qg_result/{default_name}_qg_draft.json"
    Path("qg_result").mkdir(parents=True, exist_ok=True)

    generator = LawQAGenerator(args.qg_type, law_data, args.llm)
    generator.mhop_qa_generate(args.num_qa_pairs)
    generator.save_to_json(save_path)


if __name__ == "__main__":
    main()
