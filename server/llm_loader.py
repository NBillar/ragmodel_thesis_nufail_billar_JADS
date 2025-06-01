# llm_loader.py

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

def load_llm(logger=None):
    model_type = os.getenv("llm_type", "llama_hf")  # default: HF
    start = time.time()
    logger.info("Loading LLM...")
    if model_type == "llama_cpp_quant":
        from llama_cpp_wrapper import LlamaCppLLMWrapper
        model_path = os.getenv("llama_cpp_model_path")
        duration = time.time() - start
        logger.info(f"LLM loaded in {duration:.2f} seconds.")
        return LlamaCppLLMWrapper(model_path), None  # No tokenizer

    elif model_type == "llama_hf":
        llm_model = os.getenv("llm_model")
        tokenizer = AutoTokenizer.from_pretrained(llm_model, use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(llm_model,
            device_map="auto",
            torch_dtype=torch.float16)
        text_gen = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=float(os.getenv("temperature")),
            max_new_tokens=int(os.getenv("max_new_tokens")),
            return_full_text=True,
        )
        duration = time.time() - start
        logger.info(f"LLM loaded in {duration:.2f} seconds.")
        return HuggingFacePipeline(pipeline=text_gen), tokenizer
    
    
    elif model_type == "deepseek":
        model_name = os.getenv("llm_model")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

        text_gen = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=float(os.getenv("temperature", "0.6")),  # recommended by DeepSeek
            max_new_tokens=int(os.getenv("max_new_tokens", "1024")),
            return_full_text=True,
        )

        duration = time.time() - start
        logger.info(f"DeepSeek model loaded in {duration:.2f} seconds.")
        return HuggingFacePipeline(pipeline=text_gen), tokenizer
    
    elif model_type == "falcon":
        model_name = os.getenv("llm_model", "tiiuae/Falcon3-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        text_gen = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=float(os.getenv("temperature", "0.7")),
            max_new_tokens=int(os.getenv("max_new_tokens", "1024")),
            return_full_text=True,
        )
        duration = time.time() - start
        logger.info(f"Falcon model loaded in {duration:.2f} seconds.")
        return HuggingFacePipeline(pipeline=text_gen), tokenizer
    
    
    elif model_type == "qwen3":
        model_name = os.getenv("llm_model", "Qwen/Qwen3-8B")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        text_gen = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=float("0.6"),  # recommended for Qwen in thinking mode
            top_p=float(os.getenv("top_p", "0.95")),
            top_k=int(os.getenv("top_k", "20")),
            max_new_tokens=int(os.getenv("max_new_tokens", "1024")),
            return_full_text=True,
        )
        duration = time.time() - start
        logger.info(f"Qwen model loaded in {duration:.2f} seconds.")
        return HuggingFacePipeline(pipeline=text_gen), tokenizer
    
    elif model_type == "qwen2_5":
        model_name = os.getenv("llm_model", "Qwen/Qwen2.5-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        text_gen = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=float("0.6"),  # typical for Qwen
            top_p=float(os.getenv("top_p", "0.95")),
            top_k=int(os.getenv("top_k", "20")),
            max_new_tokens=int(os.getenv("max_new_tokens", "1024")),
            return_full_text=True,
        )
        duration = time.time() - start
        logger.info(f"Qwen2.5 model loaded in {duration:.2f} seconds.")
        return HuggingFacePipeline(pipeline=text_gen), tokenizer
    
    
    elif model_type == "llama3_3b":
        model_name = os.getenv("llm_model", "meta-llama/Llama-3.2-3B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        text_gen = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=float(os.getenv("temperature", "0.7")),
            max_new_tokens=int(os.getenv("max_new_tokens", "1024")),
            return_full_text=True,
        )
        duration = time.time() - start
        logger.info(f"LLaMA 3.2 3B model loaded in {duration:.2f} seconds.")
        return HuggingFacePipeline(pipeline=text_gen), tokenizer