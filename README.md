# KoBLEX: Open Legal Question Answering with Multi-hop Reasoning
* Official Repository for paper **KoBLEX: Open Legal Question Answering with Multi-hop Reasoning**. 
* This work will be presented at [EMNLP 2025](https://2025.emnlp.org/) conference.

For experiments using KoBLEX, see [experiments README](experiments/README.md).  
For data generation details, see [data generation README](data_generation/README.md).

## Benchmark Overview

**KoBLEX** (**Ko**rean **B**enchmark for **L**egal **EX**plainable open-ended QA) is designed to evaluate multi-hop legal reasoning capabilities. 
It comprises 226 multi-hop questions, answers, and their supporting statutory provisions, curated through a hybrid pipeline that combines LLM-based generation with expert revision and evaluation.
It consists of the following files:

| File | Description |
|------|-------------|
| `koblex.jsonl` | Multi-hop QA pairs with background scenarios, questions, answers, and legal contexts. |
| `statute.jsonl` | Korean statutory articles |
| `statute_eng.jsonl` | English translations of statutes for multilingual research purposes. |

---

## Hugging Face Usage

You can load the datasets directly using the `datasets` library:

```python
from datasets import load_dataset

# Load Korean QA dataset
koblex = load_dataset("JihyungL/KoBLEX-koblex")
print(koblex['test'][0])

# Load Korean statute corpus
statute = load_dataset("JihyungL/KoBLEX-statute")
print(statute['corpus'][0])

# Load English statute corpus
statute_eng = load_dataset("JihyungL/KoBLEX-statute-eng")
print(statute_eng['corpus'][0])
```

### English Version of KoBLEX
The English version is provided for multilingual research purposes.  
However, there are several important changes compared to the original Korean version:

1. **Statute corpus alignment**  
   - Some Korean statutes do not have official English translations.  
   - As a result, the English statute corpus `statute_eng.jsonl` cannot be matched **1:1** with the Korean `statute.jsonl`.

2. **Granularity difference**  
   - The Korean statutes are provided at the **paragraph (항)** level.  
   - The English statutes are only available at the **article (조)** level.


## License
KoBLEX is licensed under [the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). 

## Citation
``
To be updated
``

