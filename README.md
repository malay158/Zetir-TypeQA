# TypeQA — A novice–expert query benchmark for type-informed RAG

This work presents TypeQA, a paired question-answering dataset developed for evaluating query-implied information requirements in retrieval-augmented generation. Conventional RAG systems retrieve evidence mainly from query semantics without considering whether the question requires a simple explanation or specialised information. TypeQA provides 1,590 novice–expert query pairs across medical (1,000) and educational (590) domains. The benchmark supports the evaluation of type-score estimation, evidence selection, and novice–expert answer adaptation in ZeTIR.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install datasets pyarrow
python restore_original_mmlu_answers.py --help
```

Run the commands from the dataset package directory. This package includes the dataset audit and MMLU answer-recovery script, **not** the complete ZeTIR training and evaluation implementation.

## Data format

The main file is `typeqa_1590_original_preserved.csv`, containing one novice–expert pair per row. The columns are `pair_id`, `domain`, `topic`, `match_score`, `novice_query`, `novice_answer`, `expert_query`, `expert_answer`, `tau_novice`, and `tau_expert`.

## Main pipeline

Recover missing MMLU multiple-choice answer keys from the original public dataset:

```bash
python restore_original_mmlu_answers.py --hf --input typeqa_1590_original_preserved.csv --output typeqa_1590_mmlu_answers_restored.csv
```

If the original MMLU test CSVs are already available locally, use:

```bash
python restore_original_mmlu_answers.py --mmlu-csv-dir /path/to/mmlu/data/test --input typeqa_1590_original_preserved.csv --output typeqa_1590_mmlu_answers_restored.csv
```

The script writes `mmlu_answer_recovery_log.csv` and fills answers only when the source question has a unique matching answer key. Unmatched items remain empty. The recovered option text is not a detailed expert explanation.

## Data and provenance

Medical pairs originate from [MedQuAD](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) and [PubMedQA](https://pubmedqa.github.io/). Educational pairs originate from [ARC](https://allenai.org/data/arc) and [MMLU](https://huggingface.co/datasets/cais/mmlu). The reconstructed collection preserves all 660 rows of the earlier TypeQA file and adds 930 candidate pairs. The recovered collection cannot yet be verified as the exact dataset used for the manuscript's original numerical results.
