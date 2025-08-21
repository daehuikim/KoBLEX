import re
import json


def parse_hierarchy(law_dict):
    def extract_law_name(hierarchy):
        """ Extracts the law name from the hierarchy string. """
        match = re.match(r"^(.*?(법|법률)(?: 시행령| 시행규칙)?)", hierarchy)
        return match.group(1) if match else None

    dict_articles = {}
    # Article(조) - Subarticle (항) / (호)
    for key, values in law_dict.items():
        for item in values:
            hierarchy = item['hierarchy']

            # 법 이름 추출
            law_name = extract_law_name(hierarchy)
            if not law_name:
                continue

            # 조 번호 추출 (ex. '10조', '10조의2', '10조의6' 등 모두 포함)
            article_match = re.search(r"(\d+)조(?:\s*(?:의)?\s*(\d+))?", hierarchy)

            if article_match:
                main = article_match.group(1)  # ex: 8
                sub = article_match.group(2)  # ex: 2 (or None)

                if sub:
                    article_number = f"{main}-{sub}조"
                else:
                    article_number = f"{main}조"
            else:
                raise KeyError

            # 항/호 추출
            clause_match = re.search(r"(\d+)(항|호)", hierarchy)
            clause = f"_{clause_match.group(1)}{clause_match.group(2)}" if clause_match else ""

            if article_number:
                dict_key = f"{law_name}_{article_number}{clause}"
                dict_articles[dict_key] = item
            else:
                print(f"조 번호 추출 실패: {hierarchy}")
                continue

    # 호의 내용은 조의 내용에 덧붙이기
    for key in list(dict_articles.keys()):
        if '조' in key and '호' not in key:
            base_content = dict_articles[key]['content']
            for sub_key in sorted(dict_articles.keys()):
                if sub_key.startswith(key) and '호' in sub_key:
                    base_content += f'\n{dict_articles[sub_key]["content"]}'
            dict_articles[key]['content'] = base_content

    return dict_articles


def load_law_articles(filepath):
    """ Load law articles from a JSONL file and parse them into a structured dictionary. """

    def clean_content(content):
        """Remove special characters from content"""
        content = re.sub(r"<[^>]+>", "", content).strip()

        if re.fullmatch(r'^제\d+조(의\d+)?\(.+?\)$', content):
            return ""

        if re.match(r'^제\d+(편|장|인|절)\b', content.split()[0]):
            return ""

        return content

    law_dict = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            index, content = entry["index"], clean_content(entry["content"])

            if content:
                if index not in law_dict:
                    law_dict[index] = []
                law_dict[index].append(entry)

    parsed_dict = parse_hierarchy(law_dict)
    return parsed_dict
