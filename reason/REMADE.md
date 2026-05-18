# Stage 2: Reasoning

## Table of Contents

* [Installation](#installation)
* [Prepare Knowledge Graph Data](#prepare-knowledge-graph-data)
* [Inference with LLMs](#inference-with-llms)
* [Evaluation](#evaluation)

## Installation

```bash
pip install -r requirements.txt
```
Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```
Prepare Knowledge Graph Data
Generate KG files from Hugging Face datasets:

```bash
python build_id_map.py --dataset webqsp --kg-base-dir data/kg_final
```
This creates the following files:

data/kg_final/webqsp/kg_entities.json
data/kg_final/webqsp/kg_relations.json
data/kg_final/webqsp/kg_triples.json
data/kg_final/webqsp/id_to_name_map.json
Inference with LLMs
Using Retrieval Results
To run reasoning with retrieval results from Stage 1:

```bash
python main.py -d webqsp --pre-retrieved-results-path P
```
where P is the path to the retrieval results obtained from retrieval inference, e.g., ../retrieve/webqsp/retrieval_result.pth.
Note: We provide pre-processed retrieval results in retrieval_result

Config
Our default config for each dataset:


# WebQSP
```bash
python main.py -d webqsp --pre-retrieved-results-path P --max-hops 2 --top-paths 128
```
# CWQ
```bash
python main.py -d cwq --pre-retrieved-results-path P --max-hops 3 --top-paths 128
```
where P is the path to retrieval results.


Evaluation
Calculate Aggregate Metrics
After running inference, calculate aggregate metrics from the log file:

```bash
python eval.py --log-file output/cwq/cwq_3_128_gpt_4o_mini.jsonl
```
Note: We provide results in optput