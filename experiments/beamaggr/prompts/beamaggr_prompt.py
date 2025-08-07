from string import Template

# Make Tree Prompt - for generating hierarchical question decomposition trees
MAKE_TREE_SYSTEM_PROMPT = """Please generate a hierarchical question decomposition tree (HQDT) with json format for a given question. 
In this tree, the root node is the original complex question, and each non-root node is a sub-question of its parent. 
Each leaf node is an atomic question that cannot be further decomposed and is used for retrieval purposes.
Strictly follow the <EXAMPLE> format: your output must be a single JSON object where
 1) key is parent question, 
2) value is sub-question list 
3)Do not include any explanation or nested object.
4) Only values from parent can be used as key
5)Never use single quotes; always use valid JSON format with double quotes only."""

MAKE_TREE_QA_PROMPT = Template("""Schema Example:
{
  "parent question": ["child1(non-atomic)", "child2(atomic)"],
  "child1": ["grandchild1", "grandchild2"]
}
                       
<EXAMPLE>
Q: 배경:갑은 군에서 10년간 복무하다가 부득이하게 퇴역하였습니다. 군인연금법에 따라 갑은 퇴역 후 연금을 받을 수 있는 자격이 없다는 판정을 받았습니다. 갑에게는 가족이 남아 있으며, 갑이 군 복무 중 납부했던 기여금에 대한 반환과 가족의 유족 급여에 관한 문제가 제기되었습니다.
질문:갑이 군 복무 중 납부한 기여금은 어떻게 처리되는가? 또한, 갑의 가족이 유족 급여를 받을 때 우선순위는 어떻게 되는가?
A: {
    "갑이 군 복무 중 납부한 기여금은 어떻게 처리되는가? 또한, 갑의 가족이 유족 급여를 받을 때 우선순위는 어떻게 되는가?":
        [
            "군 복무 중 납부한 기여금은 어떻게 처리 되나요?", 
            "군인의 유족 급여를 지급할 때 우선 순위는 어떻게 되나요?"
        ]
    }
    
Q: 배경:갑은 폭행죄로 1심에서 징역 6개월을 선고받았다. 이에 피고인 갑과 검사 모두 항소하였다.
질문:항소심에서 갑에게 1심 판결보다 무거운 형을 선고할 수 있는가?
A: {
    "항소심에서 갑에게 1심 판결보다 무거운 형을 선고할 수 있는가?":
        [
            "피고인이 항소한 사건에서 원심보다 무거운 형을 선고할 수 있나요?"
        ]
    }

Q: 배경:갑은 을의 집에 몰래 침입하여 금품을 훔치던 중 을에게 발각되었습니다. 을이 금품을 되찾으려 하자, 갑은 을에게 폭력을 행사하여 탈출에 성공했습니다. 며칠 후, 또 다른 사건에서 병은 금품을 강도하는 과정에서 피해자 정에게 심각한 상해를 입혔습니다.
질문:갑이 을에게 폭력을 행사하여 금품 탈취를 시도하고, 병이 정에게 상해를 입힌 경우 각각 어떤 처벌을 받게 되는가?
A: {
    "갑이 을에게 폭력을 행사하여 금품 탈취를 시도하고, 병이 정에게 상해를 입힌 경우 각각 어떤 처벌을 받게 되는가?":
        [
            "폭력을 행사하거나 금품갈취를 하면 어떤 처벌을 받게 되나요?",
            "상해를 입히면 어떤 처벌을 받게되나요?"
        ],
    "폭력을 행사하거나 금품갈취를 하면 어떤 처벌을 받게 되나요?":
        [
            "폭력을 행사하면 어떤 처벌을 받게 되나요?",
            "금품갈취를 하면 어떤 처벌을 받게 되나요?"
        ]
    }

Q: 배경:갑은 서울시 강남구에 위치한 오래된 아파트 단지의 토지 소유자이며, 이 아파트 단지는 도시 및 주거환경정비법에 따른 재건축사업 대상지로 지정되었습니다. 을은 이 재건축사업의 사업시행자로, 재건축 조합을 설립하여 사업을 추진하고 있습니다. 그러나 갑은 재건축 조합 설립 및 사업시행자 지정에 동의하지 않았습니다. 을은 갑에게 해당 아파트와 토지의 소유권 매도를 청구하려고 합니다.
질문: 을이 갑에게 아파트와 토지의 소유권 매도를 청구할 수 있는 기간은 언제부터 시작되며, 만약 갑과 협의가 성립되지 않을 경우 을은 어떻게 해야 하는가?
A: {
    "을이 갑에게 아파트와 토지의 소유권 매도를 청구할 수 있는 기간은 언제부터 시작되며, 만약 갑과 협의가 성립되지 않을 경우 을은 어떻게 해야 하는가?":
    [
        "재건축사업 시행자는 거주자에게 언제부터 소유권 매도를 청구할 수 있나요?", 
        "재건축사업 시행자와 거주자간 합의가 되지 않으면 어떻게 되나요?"
    ]
  }

Q: 배경:갑(정부기관)은 새로운 환경 규제를 설정하면서 일부 기업들에게 폐기물 배출량을 줄이라는 명령을 내렸다. 을(한 기업)은 이 명령이 지나치게 엄격하며 자사의 영업을 방해한다고 주장하며, 명령의 기준이 불명확하다고 법원에 소송을 제기했다. 을은 행정청이 기준을 설정하거나 이를 공표하지 않아 명령이 불공정하다고 주장했다.
질문: 갑이 을에게 내린 환경 규제가 재량에 속하는 처분이라도 어떤 경우에 법원이 이를 취소할 수 있으며, 갑이 환경 규제의 기준을 설정 및 공표하는 데 어떤 의무가 있는가?
A: {
    "갑이 을에게 내린 환경 규제가 재량에 속하는 처분이라도 어떤 경우에 법원이 이를 취소할 수 있으며, 갑이 환경 규제의 기준을 설정 및 공표하는 데 어떤 의무가 있는가?":
    [
        "행정청이 기업에 내린 환경 규제는 언제 법원이 취소할 수 있나요?",
        "행정청은 환경 규제의 기준을 설정 및 공표하는데 어떤 의무가 있나요?"
    ]
  }

<Query>
Q: 배경: $background
질문: $question
A: """)

# Beam Aggregation Question Answering Prompts
CB_SYSTEM_PROMPT = """You are given legal Q&A examples. For a new legal question, answer briefly and clearly in one or two sentences. Focus only on the core legal rule or principle. Do not use Markdown."""

OB_SYSTEM_PROMPT = """You are given a legal question and its related law texts (context). Read the context carefully and write a concise, plain-text answer (1–2 sentences) that accurately summarizes the legal principle or outcome. Do not use Markdown formatting."""

# Template prompts for different question types
CB_TEMPLATE = Template("""<EXAMPLE>
Question: 갑이 군 복무 중 납부한 기여금은 어떻게 처리되는가?
Answer: 갑이 군 복무 중 납부한 기여금은 그 기여금에 대통령령으로 정하는 이자를 합한 금액으로 반환된다.

Question: 갑의 가족이 유족 급여를 받을 때 우선순위는 어떻게 되는가?
Answer: 갑의 가족이 유족 급여를 받을 우선순위는 「민법」에 따른 상속의 순위에 따른다.

<Query>
Question: $question
Answer: """)

OB_TEMPLATE = Template("""<EXAMPLE>
Question: 갑이 군 복무 중 납부한 기여금은 어떻게 처리되는가?
Context: 군인연금법 4조 기여금의 반환 제4조(기여금의 반환) 군인이었던 사람으로서 이 법에 따른 급여를 받을 권리가 없는 사람 또는 그 유족에 대해서는 그 군인이 복무할 때 낸 기여금에 대통령령으로 정하는 이자를 합한 금액을 반환한다.
Answer: 갑이 군 복무 중 납부한 기여금은 그 기여금에 대통령령으로 정하는 이자를 합한 금액으로 반환된다.

<Query>
Question: $question
Context: $context
Answer: """)