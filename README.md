
# RAG Model Pipeline

This repository is built untop of the [RAGMEUP](https://github.com/FutureClubNL/RAGMeUp) repo, it contains the implementation of a modular, production-grade Retrieval-Augmented Generation (RAG) pipeline. The system integrates document ingestion, hybrid retrieval, reranking, context-grounded response generation, and evaluation into a cohesive framework, with support for real-world deployments including SharePoint integration and advanced provenance tracking.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Modular RAG architecture with separate components for ingestion, retrieval, reranking, generation, and evaluation.
- **Document ingestion** from local files and SharePoint, including automated extraction of provenance links from .docx files.
- **Hybrid retrieval** combining sparse (BM25) and dense (embedding-based) retrieval using Milvus and GIST-small embeddings.
- **Reranking** of retrieved documents using cross-encoders like MiniLM-L-12-v2, implemented in `RAGHelper.py`.
- **Context-grounded generation** with customizable prompts for multi-turn conversations, using `llm_loader.py` to load and switch between LLM backends.
- **Evaluation** pipeline with RAGAS metrics, Krippendorff’s alpha, and structured TAM analysis, including custom metrics (Faccor) and thematic analysis of open-ended feedback.
- **Deployment-ready**: Backend implemented using FastAPI (`server.py`), with options for local or containerized deployment.
- **Advanced provenance tracking** linking chatbot answers to specific documents and SharePoint links (`provenance.py`).

## System Architecture

1. **Ingestion**: SharePoint and local document ingestion with metadata enrichment (e.g., provenance links), using `RAGHelper_local.py` and `sharepoint_regex.py`.
2. **Retrieval**: Hybrid retriever combining BM25 and GIST-small embeddings via `RAGHelper.py`, with support for Milvus vector store.
3. **Reranking**: Cross-encoder reranking to refine top-K documents using `RAGHelper.py` and `MiniLM-L-12-v2`.
4. **Generation**: LLM response generation with customizable prompts. Supports Hugging Face models, quantized LLaMA models (`llama_cpp_wrapper.py`), and modular backend loading via `llm_loader.py`.
5. **Evaluation**: Automated RAG performance evaluation (`Ragas_eval.py`), custom Faccor metrics (`faccor.py`), provenance analysis (`provenance.py`), and thematic analysis of open feedback (`plot_generator.py`).
6. **API Service**: FastAPI-based server with endpoints for interaction, implemented in `server.py`.

## Prerequisites

- Python 3.8+
- Hugging Face Transformers
- LangChain
- Pingouin
- Krippendorff alpha library
- Milvus (for vector storage)
- FastAPI
- Docker (optional)

## Installation

```bash
git clone https://github.com/NBillar/ragmodel_thesis_nufail_billar_JADS.git
cd ragmodel_thesis_nufail_billar_JADS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

- **Environment Variables**: `.env` files configure prompt templates, model paths, and SharePoint directories.
- **Model Loading**: Configurable via `llm_loader.py`, supporting modular selection of models (e.g., `Llama-3.2-3B`, `DeepSeek`, `Qwen`).
- **SharePoint Ingestion**: Path setup via `.env` or directly in `RAGHelper_local.py`.

## Usage

- **Run API Server**:
```bash
uvicorn server:app --reload
```
- **Document Ingestion**: Automatically ingests documents from configured SharePoint path.
- **Interactive Queries**: Send questions to the API and receive context-grounded answers with provenance tracking.
- **Evaluation**: Run `Ragas_eval.py` for automated benchmark metrics.

## Evaluation

- **Quantitative Metrics**: RAGAS, Faccor (custom), Krippendorff’s alpha for interrater agreement.
- **Qualitative Analysis**: Open-ended feedback processed into frequency tables and word clouds using `plot_generator.py`.
- **Thematic Analysis**: Themes like Trust, Usefulness, Clarity are extracted and coded.
- **Visualization**: Automated plots for TAM scores, Krippendorff’s alpha, open-ended themes.

## Directory Structure

```
ragmodel_thesis_nufail_billar_JADS/server
├── RAGHelper.py                # Core RAG functionality (retrieval, reranking)
├── RAGHelper_local.py          # Local ingestion with SharePoint integration
├── llm_loader.py               # Modular LLM loading
├── llama_cpp_wrapper.py        # Quantized LLaMA model support
├── provenance.py               # Provenance scoring and tracking
├── Ragas_eval.py               # Evaluation pipeline with RAGAS and custom metrics
├── faccor.py                   # Custom Faccor metric calculation
├── server.py                   # FastAPI-based backend
├── sharepoint_regex.py         # SharePoint link extraction from .docx
├── plot_generator.py           # Open-ended feedback analysis and plotting
└── /data, /models, /config     # Data, model files, and configuration
```

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.


