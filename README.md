# KoBLEX: Open Legal Question Answering with Multi-hop Reasoning
* Official Repository for paper [KoBLEX: Open Legal Question Answering with Multi-hop Reasoning](https://aclanthology.org/2025.emnlp-main.200/). 
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

   
## Experiments Overview

This repository includes comprehensive experiments on **Retrieval-Augmented Reasoning for Legal Multi-hop Open Question Answering**. The experiments evaluate various state-of-the-art RAG methods adapted for legal domain requirements, comparing them against our proposed **Parser** methodology.

### Key Research Focus
- **Multi-hop Legal Reasoning**: Complex legal questions requiring information from multiple statutory provisions
- **Retrieval-Augmented Generation**: Combining document retrieval with generative reasoning
- **Legal Domain Adaptation**: Tailoring general RAG methods for legal text characteristics

### Experimental Methods

**Proposed Method**: 
- **Parser**: Novel 3-stage retrieval pipeline with parametric provision generation for enhanced legal statute search accuracy

**Baseline Methods**:
- **Standard Prompting**: Direct question answering without retrieval augmentation
- **Chain-of-Thought**: Step-by-step reasoning without retrieval
- **Self-Ask**: Question decomposition into sub-questions with step-by-step reasoning
- **IRCoT**: Interleaving retrieval with chain-of-thought reasoning
- **FLARE**: Forward-looking active retrieval for anticipatory information gathering
- **ProbTree**: Hierarchical question decomposition with probabilistic reasoning
- **BeamAggr**: Beam search with multiple strategies for information aggregation

### Evaluation Framework
- **LF-Eval**: We propose a novel Legal Fidelity Evaluation using G-Eval-based assessment
- **Multi-metric Evaluation**: Token-level F1, retrieval precision/recall, and legal accuracy scoring
- **Comprehensive Benchmarking**: Results across multiple model backends (GPT-4o, Qwen, Exaone)

For detailed experimental setup and implementation, see [experiments README](experiments/README.md).

## Results Overview

The `results/` directory contains generated result files from various methods evaluated on KoBLEX.

## Citation
```
@inproceedings{lee-etal-2025-koblex,
    title = "{K}o{BLEX}: Open Legal Question Answering with Multi-hop Reasoning",
    author = "Lee, Jihyung  and
      Kim, Daehui  and
      Hwang, Seonjeong  and
      Kim, Hyounghun  and
      Lee, Gary",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.200/",
    pages = "4019--4053",
    ISBN = "979-8-89176-332-6",
    abstract = "Large Language Models (LLM) have achieved remarkable performances in general domains and are now extending into the expert domain of law. Several benchmarks have been proposed to evaluate LLMs' legal capabilities. However, these benchmarks fail to evaluate open-ended and provision-grounded Question Answering (QA). To address this, we introduce a Korean Benchmark for Legal EXplainable QA (KoBLEX), designed to evaluate provision-grounded, multi-hop legal reasoning. KoBLEX includes 226 scenario-based QA instances and their supporting provisions, created using a hybrid LLM{--}human expert pipeline. We also propose a method called Parametric provision-guided Selection Retrieval (ParSeR), which uses LLM-generated parametric provisions to guide legally grounded and reliable answers. ParSeR facilitates multi-hop reasoning on complex legal questions by generating parametric provisions and employing a three-stage sequential retrieval process. Furthermore, to better evaluate the legal fidelity of the generated answers, we propose Legal Fidelity Evaluation (LF-Eval). LF-Eval is an automatic metric that jointly considers the question, answer, and supporting provisions and shows a high correlation with human judgments. Experimental results show that ParSeR consistently outperforms strong baselines, achieving the best results across multiple LLMs. Notably, compared to standard retrieval with GPT-4o, ParSeR achieves +37.91 higher F1 and +30.81 higher LF-Eval. Further analyses reveal that ParSeR efficiently delivers consistent performance across reasoning depths, with ablations confirming the effectiveness of ParSeR."
}
```

