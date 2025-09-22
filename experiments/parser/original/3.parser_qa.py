import json
import os
from string import Template
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

system_qa = """You are a legal assistant AI. Given a user Background and a Question, generate a concise and accurate answer by grounding your response only in the provided legal passages (Context).
-Do not draw on any external knowledge, statutes, precedent, or opinion.
-Keep your answers concise and strictly grounded in the quoted text
-Do not refuse or say insufficient context. Instead, use the provided Context to the fullest and deliver a complete, concise answer grounded solely in those passages.
"""

qa_prompt = Template("""<EXAMPLE>
Question: 갑은 군에서 10년간 복무하다가 부득이하게 퇴역하였습니다. 군인연금법에 따라 갑은 퇴역 후 연금을 받을 수 있는 자격이 없다는 판정을 받았습니다. 갑에게는 가족이 남아 있으며, 갑이 군 복무 중 납부했던 기여금에 대한 반환과 가족의 유족 급여에 관한 문제가 제기되었습니다.
갑이 군 복무 중 납부한 기여금은 어떻게 처리되는가? 또한, 갑의 가족이 유족 급여를 받을 때 우선순위는 어떻게 되는가?
Context: 
군인연금법 4조  기여금의 반환 제4조(기여금의 반환) 군인이었던 사람으로서 이 법에 따른 급여를 받을 권리가 없는 사람 또는 그 유족에 대해서는 그 군인이 복무할 때 낸 기여금에 대통령령으로 정하는 이자를 합한 금액을 반환한다.
군인연금법 10조  유족의 우선순위 제10조(유족의 우선순위) 급여를 받을 유족의 순위는 「민법」에 따른 상속의 순위에 따른다.
Answer: 갑이 군 복무 중 납부한 기여금은 그 기여금에 대통령령으로 정하는 이자를 합한 금액으로 반환된다. 갑의 가족이 유족 급여를 받을 우선순위는 「민법」에 따른 상속의 순위에 따른다.

Question: 갑은 폭행죄로 1심에서 징역 6개월을 선고받았다. 이에 피고인 갑과 검사 모두 항소하였다.
항소심에서 갑에게 1심 판결보다 무거운 형을 선고할 수 있는가?
Context: 
형사소송법 368조  불이익변경의 금지 제368조(불이익변경의 금지) 피고인이 항소한 사건과 피고인을 위하여 항소한 사건에 대해서는 원심판결의 형보다 무거운 형을 선고할 수 없다."]
Answer: 항소심에서 갑에게 1심 판결보다 무거운 형을 선고할 수 없다.

Question: 갑은 을의 집에 몰래 침입하여 금품을 훔치던 중 을에게 발각되었습니다. 을이 금품을 되찾으려 하자, 갑은 을에게 폭력을 행사하여 탈출에 성공했습니다. 며칠 후, 또 다른 사건에서 병은 금품을 강도하는 과정에서 피해자 정에게 심각한 상해를 입혔습니다.
갑이 을에게 폭력을 행사하여 금품 탈취를 시도하고, 병이 정에게 상해를 입힌 경우 각각 어떤 처벌을 받게 되는가?
Context:
형법 337조  강도상해, 치상 제337조(강도상해, 치상) 강도가 사람을 상해하거나 상해에 이르게 한때에는 무기 또는 7년 이상의 징역에 처한다. <개정 1995.12.29>
형법 335조  준강도 제335조(준강도) 절도가 재물의 탈환에 항거하거나 체포를 면탈하거나 범죄의 흔적을 인멸할 목적으로 폭행 또는 협박한 때에는 제333조 및 제334조의 예에 따른다."]
Answer: 병이 사람을 상해하거나 상해에 이르게 한 경우 무기 또는 7년 이상의 징역에 처하고, 갑이 재물의 탈환에 항거하여 폭행한 경우 제333조에 따라 3년 이상의 유기징역에 처한다.

Question: 갑은 서울시 강남구에 위치한 오래된 아파트 단지의 토지 소유자이며, 이 아파트 단지는 도시 및 주거환경정비법에 따른 재건축사업 대상지로 지정되었습니다. 을은 이 재건축사업의 사업시행자로, 재건축 조합을 설립하여 사업을 추진하고 있습니다. 그러나 갑은 재건축 조합 설립 및 사업시행자 지정에 동의하지 않았습니다. 을은 갑에게 해당 아파트와 토지의 소유권 매도를 청구하려고 합니다.
을이 갑에게 아파트와 토지의 소유권 매도를 청구할 수 있는 기간은 언제부터 시작되며, 만약 갑과 협의가 성립되지 않을 경우 을은 어떻게 해야 하는가?
Context: 
도시 및 주거환경정비법 64조  재건축사업에서의 매도청구 4항 ④ 제2항의 기간이 지나면 사업시행자는 그 기간이 만료된 때부터 2개월 이내에 조합설립 또는 사업시행자 지정에 동의하지 아니하겠다는 뜻을 회답한 토지등소유자와 건축물 또는 토지만 소유한 자에게 건축물 또는 토지의 소유권과 그 밖의 권리를 매도할 것을 청구할 수 있다.
도시 및 주거환경정비법 64조  재건축사업에서의 매도청구 1항 ① 재건축사업의 사업시행자는 사업시행계획인가의 고시가 있은 날부터 30일 이내에 다음 각 호의 자에게 조합설립 또는 사업시행자의 지정에 관한 동의 여부를 회답할 것을 서면으로 촉구하여야 한다. 1. 제35조제3항부터 제5항까지에 따른 조합설립에 동의하지 아니한 자 2. 제26조제1항 및 제27조제1항에 따라 시장ㆍ군수등, 토지주택공사등 또는 신탁업자의 사업시행자 지정에 동의하지 아니한 자
도시 및 주거환경정비법 64조  재건축사업에서의 매도청구 2항 ② 제1항의 촉구를 받은 토지등소유자는 촉구를 받은 날부터 2개월 이내에 회답하여야 한다.
도시 및 주거환경정비법 73조  분양신청을 하지 아니한 자 등에 대한 조치 2항 ② 사업시행자는 제1항에 따른 협의가 성립되지 아니하면 그 기간의 만료일 다음 날부터 60일 이내에 수용재결을 신청하거나 매도청구소송을 제기하여야 한다."]
Answer: 을은 제2항의 기간이 만료된 때부터 2개월 이내에 갑에게 아파트와 토지의 소유권 매도를 청구할 수 있으며, 협의가 성립되지 않을 경우 그 기간의 만료일 다음 날부터 60일 이내에 수용재결을 신청하거나 매도청구소송을 제기하여야 한다.

Question: 갑(정부기관)은 새로운 환경 규제를 설정하면서 일부 기업들에게 폐기물 배출량을 줄이라는 명령을 내렸다. 을(한 기업)은 이 명령이 지나치게 엄격하며 자사의 영업을 방해한다고 주장하며, 명령의 기준이 불명확하다고 법원에 소송을 제기했다. 을은 행정청이 기준을 설정하거나 이를 공표하지 않아 명령이 불공정하다고 주장했다.
갑이 을에게 내린 환경 규제가 재량에 속하는 처분이라도 어떤 경우에 법원이 이를 취소할 수 있으며, 갑이 환경 규제의 기준을 설정 및 공표하는 데 어떤 의무가 있는가?
Context: 
행정소송법 27조  재량처분의 취소 제27조(재량처분의 취소) 행정청의 재량에 속하는 처분이라도 재량권의 한계를 넘거나 그 남용이 있는 때에는 법원은 이를 취소할 수 있다.
행정절차법 20조  처분기준의 설정ㆍ공표 1항 ① 행정청은 필요한 처분기준을 해당 처분의 성질에 비추어 되도록 구체적으로 정하여 공표하여야 한다. 처분기준을 변경하는 경우에도 또한 같다."]
Answer: 재량권의 한계를 넘거나 그 남용이 있는 때 법원은 이를 취소할 수 있으며, 갑은 규제의 성질에 비추어 필요한 규제기준을 구체적으로 정하여 공표하여야 한다.

<Query>
Question: $question
Context: 
$context_str
Answer: """)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

class LlmGenerator:
    def __init__(self,
                 model_name: str,
                 dtype: str,
                 trust_remote_code: bool,
                 tensor_parallel_size: int,
                 temperature: float,
                 top_p: float,
                 max_tokens: int):
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=0
        )
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=10000
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"

    def generate(self, prompts):
        batch_inputs = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_qa},
                {"role": "user", "content": prompt}
            ]
            inp = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )
            batch_inputs.append(self.tokenizer.decode(inp[0], skip_special_tokens=False))

        responses = self.llm.generate(batch_inputs, self.sampling_params)
        return [resp.outputs[0].text.strip() for resp in responses]



def create_prompts(data_list):
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

if __name__ == '__main__':
    input_file = "input-path"
    data_list=[]
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    prompts = create_prompts(data_list)
    
    llm_gen = LlmGenerator(
        model_name="model name",
        dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=2,
        temperature=0.0,
        top_p=0.9,
        max_tokens=4000
    )

    outputs = llm_gen.generate(prompts)
    answers = process_completions(outputs)
    
    out_file = "output-path"
    with open(out_file, 'w', encoding='utf-8') as fw:
        for item, provisions, answer in zip(data_list, [item.get('selected_provisions', []) for item in data_list], answers):
            record = {
                "background": item['background'],
                "question": item['question'],
                "answer": answer,
                "provisions": provisions
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved detailed answers to {out_file}")
