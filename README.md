# SGPR: Semantic-Gated Path Retrieval for Knowledge Graph-Grounded RAG

This is the PyTorch implementation for the paper.


## Requirements

python>=3.8
torch>=2.0.0
torch_geometric>=2.5.0
openai>=1.0.0
datasets>=2.0.0
tqdm>=4.60.0
tenacity>=8.0.0



## Datasets

We support two built-in multi-hop knowledge graph question answering (KGQA) datasets:

- [WebQSP](https://huggingface.co/datasets/rmanluo/RoG-webqsp): WebQuestions Semantic Parses dataset for complex question answering over Freebase.
- [CWQ](https://huggingface.co/datasets/rmanluo/RoG-cwq): ComplexWebQuestions dataset for multi-hop reasoning over knowledge graphs.

## Usage

This is a retrieval-and-reasoning pipeline for knowledge-graph-based retrieval-augmented generation.

1. For the retrieval stage, see [the retrieve folder](./retrieve/).
2. For the reasoning stage, see [the reason folder](./reason/).
