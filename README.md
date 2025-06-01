# RAG Model Pipeline

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) pipeline. The system integrates a modular architecture for ingesting documents, performing hybrid retrieval (dense and sparse), reranking with a cross-encoder, and generating context-grounded responses using Large Language Models (LLMs). The RAG pipeline is designed for enterprise-grade performance, supporting real-time retrieval from distributed sources and integrating seamlessly with SharePoint-based document ingestion.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Open-Ended Feedback Analysis](#open-ended-feedback-analysis)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Modular RAG architecture supporting hybrid retrieval (BM25 and dense embeddings).
- Configurable LLM backend (supports Hugging Face, quantized LLaMA models, DeepSeek, Qwen, etc.).
- Real-time ingestion of documents from local files and SharePoint with provenance tracking.
- Integration with Krippendorff's alpha for interrater agreement evaluation.
- Comprehensive UAT (User Acceptance Testing) framework.
- Customizable prompt templates for context and memory-aware conversation.
- Robust logging, error handling, and performance monitoring.

## System Architecture

1. **Document Ingestion**: Supports .txt, .docx, and SharePoint-synced files. Extracts SharePoint links for provenance.
2. **Retrieval**: Implements hybrid retrieval using BM25 and dense embeddings (e.g., GIST-small).
3. **Reranking**: Uses cross-encoder models (e.g., MiniLM-L-12-v2) to rerank retrieved documents.
4. **Response Generation**: Leverages a modular LLM backend, supporting various models with customizable prompts.
5. **Evaluation**: Includes RAGAS metrics, Krippendorff’s alpha, TAM analysis, and open-ended feedback analysis.

## Prerequisites

- Python 3.8+
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Pingouin](https://pingouin-stats.org/)
- [Krippendorff](https://github.com/pln-fing-udelar/krippendorff-alpha)
- [Milvus](https://milvus.io/) for vector storage (optional)
- [FastAPI](https://fastapi.tiangolo.com/) for backend services
- Docker (optional for containerized deployment)

