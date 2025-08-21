import json
import openai
import re
import argparse

from pathlib import Path
from tqdm import tqdm
from itertools import combinations

# Constants
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
PROMPT_FILES = {
    "partial_check": PROMPTS_DIR / "partial_check.txt",
    "full_check": PROMPTS_DIR / "full_check.txt",
}


class InvalidAPIResult(Exception):
    pass


def load_prompts() -> dict:
    prompts = {}
    for key, path in PROMPT_FILES.items():
        with open(path, 'r', encoding='utf-8') as f:
            prompts[key] = f.read().strip()
    return prompts


class LawQAValidator:
    def __init__(self, qg_data: dict, llm: str):
        self.qg_data = qg_data
        self.llm = llm
        self.max_token = 1024
        self.temperature = 0
        self.top_p = 0.9
        self.prompts = load_prompts()

    def api_call(self, prefix: str, prompt: str) -> str:
        response = openai.chat.completions.create(
            model=self.llm,
            messages=[
                {"role": "system", "content": prefix},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content.strip()

    def format_context(self, context: dict) -> str:
        return "".join(
            f"sent{num}: {entity['hierarchy']} | {entity['content']}\n"
            for num, entity in enumerate(context.values(), start=1)
        )

    def validate_all(self):
        """ Validate all question-generation pairs in the dataset."""
        for qg_pair in tqdm(self.qg_data):
            context = qg_pair['context']
            background = qg_pair['background_scenario']
            question = qg_pair['question_scenario']
            answer = qg_pair['answer_scenario']

            # Partial Check
            qg_pair['partial_results'] = self.partial_check_all(context, background, question)
            qg_pair['partial_total_check'] = self.evaluate_partial_validity(context, qg_pair['partial_results'])

            # Full Check
            qg_pair['full_result'] = self.full_check(background, question, answer, context)
            qg_pair['full_total_check'] = all(
                qg_pair['full_result'].get(k) == "Yes"
                for k in ["Scenario Consistent", "Correct", "Derivable"]
            )

    def partial_check_all(self, context: dict, background: str, question: str) -> dict:
        """ Check all combinations of context entities for answerability."""
        keys = list(context.keys())
        results = {}

        for r in range(1, len(keys) + 1):
            for combo in combinations(keys, r):
                subset = {k: context[k] for k in combo}
                combo_key = ','.join(sorted(combo))

                try:
                    result = self.partial_check(background, question, subset)
                    results[combo_key] = result

                    if result.get("Answerable") != "No":
                        return results  # Early stop

                except Exception as e:
                    results[combo_key] = {"error": str(e)}

        return results

    def evaluate_partial_validity(self, full_context: dict, results: dict) -> bool:
        """ Evaluate the validity of partial check results against the full context."""
        full_key = ','.join(sorted(full_context.keys()))
        valid = True

        for key, val in results.items():
            answerable = val.get("answerable") or val.get("Answerable")
            if answerable == "Yes" and key != full_key:
                valid = False
            if answerable == "No" and key == full_key:
                valid = False

        return valid

    def full_check(self, background: str, question: str, answer: str, context: dict) -> dict:
        """ Perform a full check on the question, answer, and context."""
        context_text = self.format_context(context)
        prompt = (
            f"background_scenario: {background}\n"
            f"question_scenario: {question}\n"
            f"answer_scenario: {answer}\n"
            f"context: {context_text}"
        )
        output = self.api_call(self.prompts['full_check'], prompt)
        return self.parse_full_output(output)

    def partial_check(self, background: str, question: str, context: dict) -> dict:
        """ Perform a partial check on the question and context."""
        context_text = self.format_context(context)
        prompt = f"question: {background} {question}\ncontext:\n{context_text}"
        output = self.api_call(self.prompts['partial_check'], prompt)
        return self.parse_partial_output(output)

    @staticmethod
    def parse_partial_output(text: str) -> dict:
        answer = re.search(r"Answerable:\s*(Yes|No)", text, re.IGNORECASE)
        justification = re.search(r"Justification:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        return {
            "Answerable": answer.group(1).strip() if answer else None,
            "Justification": justification.group(1).strip() if justification else None
        }

    @staticmethod
    def parse_full_output(text: str) -> dict:
        keys = {
            "Scenario Consistent": r"Scenario Consistent:\s*(Yes|No)",
            "Scenario Justification": r"Scenario Justification:\s*(.*?)(?=\n[A-Z]|$)",
            "Correct": r"Correct:\s*(Yes|No)",
            "Explanation": r"Explanation:\s*(.*?)(?=\n[A-Z]|$)",
            "Derivable": r"Derivable:\s*(Yes|No)",
            "Justification": r"Justification:\s*(.*?)(?=\n[A-Z]|$)"
        }
        result = {}
        for key, pattern in keys.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
        return result

    def save_to_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.qg_data, f, ensure_ascii=False, indent=4)
        print(f"Saved validated QA to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run validation on multi-hop QA dataset.")
    parser.add_argument('--llm', type=str, default='gpt-4o', help='OpenAI model to use')
    parser.add_argument('--qg_path', type=str, required=True, help='Path to QA JSON file')
    parser.add_argument('--save_path', type=str, help='Where to save the validated results')
    args = parser.parse_args()

    qg_data = json.load(open(args.qg_path, encoding='utf-8'))

    save_path = args.save_path or args.qg_path.replace('.json', '_validation.json')
    validator = LawQAValidator(qg_data, args.llm,)
    validator.validate_all()
    validator.save_to_json(save_path)


if __name__ == '__main__':
    main()
