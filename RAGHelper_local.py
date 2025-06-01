import json
import os
import re
import numpy as np
import torch
from provenance import (
    compute_attention,
    compute_rerank_provenance,
    compute_llm_provenance,
    DocumentSimilarityAttribution
)
from RAGHelper import RAGHelper, escape_curly_braces
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
# from langchain.chains.llm import LLMChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from collections import defaultdict
from llm_loader import load_llm

import logging

logger = logging.getLogger(__name__)



def _safe_json(obj):
    """Convert non-serializable types (e.g., numpy types) to JSON-safe values."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)  # fallback
    
class RAGHelperLocal(RAGHelper):
    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger
        self.llm, self.tokenizer = load_llm(self.logger)
        self.embeddings = self._initialize_embeddings()
        logger.debug(f"Tokenizer loaded: {self.tokenizer}")
        logger.debug(f"Model loaded: {self.llm}")
        # Load the data
        self.load_data()
        # Create RAG chains
        self.rag_fetch_new_chain = self._create_rag_chain()
        self.rewrite_ask_chain, self.rewrite_chain = self._initialize_rewrite_chains()
        # Initialize provenance method
        self.attributor = DocumentSimilarityAttribution() if os.getenv("provenance_method") == "similarity" else None
    
    def _get_prompt_key(self, base: str) -> str:
        llm_type = os.getenv("llm_type", "llama_hf")
        if llm_type == "deepseek":
            return f"{base}_deepseek"
        return base
    
    @staticmethod
    def _get_bnb_config():
        """Get BitsAndBytes configuration for model quantization."""
        use_4bit = True
        bnb_4bit_compute_dtype = "float16"
        bnb_4bit_quant_type = "nf4"
        use_nested_quant = False

        return BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=use_nested_quant,
        )
        
    def _render_chat_template(self, messages: list[dict]) -> str:
        llm_type = os.getenv("llm_type", "llama_hf")

        if llm_type == "deepseek":
            prompt_parts = []
            for msg in messages:
                prompt_parts.append(msg["content"]) 
            return "<think>\n" + "\n".join(prompt_parts) + "<|eot_id|>assistant\n\n"
    
        if llm_type == "falcon":
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"<|system|>\n{msg['content']}")
                elif msg["role"] == "user":
                    prompt_parts.append(f"<|user|>\n{msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"<|assistant|>\n{msg['content']}")
            prompt_parts.append("<|assistant|>\n") 
            return "\n".join(prompt_parts)
        
        
        if llm_type == "llama3_3b":
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return rendered
            except Exception as e:
                self.logger.warning(f"LLaMA 3.2 chat template error: {e}")
                return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])    
            
                
        if llm_type == "qwen3":
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  
                )
                return rendered  
            except Exception as e:
                self.logger.warning(f"Qwen3 chat template error: {e}")
                return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
            
        if llm_type == "qwen2_5":
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return rendered
            except Exception as e:
                self.logger.warning(f"Qwen2.5 chat template error: {e}")
                return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
            


        # fallback: use tokenizer's chat template if available
        try:
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                return self.tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            self.logger.warning(f"Chat template rendering failed: {e} — using fallback format.")

        # fallback manual formatting
        return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    

    @staticmethod
    def _initialize_embeddings():
        """Initialize and return embeddings for vector storage."""
        model_kwargs = {
            'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if os.getenv(
                'force_cpu') != "True" else 'cpu'
        }
        return HuggingFaceEmbeddings(
            model_name=os.getenv('embedding_model'),
            model_kwargs=model_kwargs
        )

    def _create_rag_chain(self):
        instruction_key = self._get_prompt_key("rag_fetch_new_instruction")
        question_key = self._get_prompt_key("rag_fetch_new_question")

        rag_thread = [
            {'role': 'system', 'content': os.getenv(instruction_key)},
            {'role': 'user', 'content': os.getenv(question_key)}
        ]
        rag_prompt_template = self._render_chat_template(rag_thread)
        rag_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=rag_prompt_template,
        )
        return {"question": RunnablePassthrough()} | rag_prompt | self.llm

    def _initialize_rewrite_chains(self):
        """Initialize and return rewrite ask and rewrite chains if required."""
        rewrite_ask_chain = None
        rewrite_chain = None

        if os.getenv("use_rewrite_loop") == "True":
            rewrite_ask_chain = self._create_rewrite_ask_chain()
            rewrite_chain = self._create_rewrite_chain()

        return rewrite_ask_chain, rewrite_chain

    def _create_rewrite_ask_chain(self):
        """Create and return the chain to ask if rewriting is needed."""
        rewrite_ask_thread = [
            {'role': 'system', 'content': os.getenv('rewrite_query_instruction')},
            {'role': 'user', 'content': os.getenv('rewrite_query_question')}
        ]
        rewrite_ask_prompt_template = self._render_chat_template(rewrite_ask_thread)
        rewrite_ask_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=rewrite_ask_prompt_template,
        )
        

        context_retriever = self.rerank_retriever if self.rerank else self.ensemble_retriever
        return {
            "context": context_retriever | RAGHelper.format_documents,
            "question": RunnablePassthrough()
        } | rewrite_ask_prompt | self.llm

    def _create_rewrite_chain(self):
        """Create and return the rewrite chain."""
        rewrite_thread = [{'role': 'user', 'content': os.getenv('rewrite_query_prompt')}]
        rewrite_prompt_template = self._render_chat_template(rewrite_thread)
        rewrite_prompt = PromptTemplate(
            input_variables=["question"],
            template=rewrite_prompt_template,
        )
        

        return {"question": RunnablePassthrough()} | rewrite_prompt | self.llm

    def handle_rewrite(self, user_query: str) -> str:
        """Handle the rewriting of the user query if necessary."""
        if os.getenv("use_rewrite_loop") == "True":
            response = self.rewrite_ask_chain.invoke(user_query)
            llm_type = os.getenv("llm_type", "llama_hf")
            end_string = os.getenv("llm_assistant_token", "assistant\n\n")
            if llm_type == "falcon":
                end_string = "<|assistant|>"
            if isinstance(response, str):
                text = response
            elif isinstance(response, dict):
                text = response.get("text", "")
            else:
                raise ValueError(f"Unexpected type for rewrite_ask_chain response: {type(response)}")

            logger.debug(f"Full LLM response text: {text}")

            try:
                index = text.rindex(end_string)
                reply = text[index + len(end_string):]
            except ValueError:
                logger.debug("Marker not found in response; using full output.")
                reply = text

            reply = re.sub(r'\W+ ', '', reply)

            if reply.lower().startswith('no'):
                response = self.rewrite_chain.invoke(user_query)

                if isinstance(response, str):
                    text = response
                elif isinstance(response, dict):
                    text = response.get("text", "")
                else:
                    raise ValueError(f"Unexpected type for rewrite_chain response: {type(response)}")

                try:
                    index = text.rindex(end_string)
                    reply = text[index + len(end_string):]
                except ValueError:
                    reply = text

                return reply.strip()
            else:
                return user_query
        else:
            return user_query

    def handle_user_interaction(self, user_query, history):
        # Step 1: Rewriting
        user_query = self.handle_rewrite(user_query)
        if os.getenv("use_re2") == "True":
            user_query = f'{user_query}\n{os.getenv("re2_prompt")}{user_query}'

        # Create full_history = history + user message (pure function style)
        full_history = history.copy()

        if len(full_history) == 0:
            full_history.append({
                "role": "system",
                "content": os.getenv(self._get_prompt_key("rag_instruction")).format(context="")
            })

        full_history.append({"role": "user", "content": user_query})

        # Step 2: Check if I should fetch new docs
        fetch_new_documents = self._should_fetch_new_documents(user_query, full_history)

        # Step 3: Retrieve docs + build prompt
        retrieved_docs = self.ensemble_retriever.invoke(user_query)
        for i, doc in enumerate(retrieved_docs):
            self.logger.info(f"[RETRIEVED DOC {i}] Source: {doc.metadata.get('source')} | SharePoint: {doc.metadata.get('sharepoint_url')} | ID: {doc.metadata.get('id')}")
            self.logger.debug(f"[RETRIEVED DOC {i}] Snippet: {doc.page_content[:150].replace(chr(10), ' ')}")
        context = RAGHelper.format_documents(retrieved_docs)

        # Step 4: Prepare prompt and run chain
        thread = self._prepare_conversation_thread(full_history, fetch_new_documents, user_query, context)
        input_variables = self._determine_input_variables(fetch_new_documents)
        prompt = self._create_prompt_template(thread, input_variables, context, user_query, full_history)
        llm_chain = self._create_llm_chain(fetch_new_documents, prompt)

        # Step 5: Generate answer
        query_input = {"question": user_query}
        reply = self._invoke_rag_chain(query_input, llm_chain, history=full_history)

        # Can NOT mutate history. The assistant message will be appended by server.py.
        logger.info("Current History:")
        for h in full_history:
            logger.info(f"- {h['role']}: {h['content'][:80]}")

        # Step 6: Optional provenance tracking
        if fetch_new_documents:
            reply["docs"] = retrieved_docs
            self._track_provenance(user_query, reply, thread)

        return thread, reply

    def _should_fetch_new_documents(self, user_query, history):
        """Determine whether to fetch new documents based on the user's query and conversation history."""
        if len(history) == 0:
            return True

        response = self.rag_fetch_new_chain.invoke(user_query)
        reply = self._extract_reply(response)
        return reply.lower().startswith('yes')

  
    def _prepare_conversation_thread(self, history, fetch_new_documents, user_query, context=""):
        thread = []

        instruction_key = self._get_prompt_key("rag_instruction")
        initial_key = self._get_prompt_key("rag_question_initial")
        followup_key = self._get_prompt_key("rag_question_followup")
        llm_type = os.getenv("llm_type", "llama_hf")
        thread.append({'role': 'user', 'content': os.getenv(instruction_key).format(context=context)})

        if not history or fetch_new_documents:
            prompt = os.getenv(initial_key).format(question=user_query)
        else:
            formatted_history = RAGHelper.format_history_for_prompt(history)
            prompt = os.getenv(followup_key).format(
                context=context,
                history=formatted_history,
                question=user_query
            )

        thread.append({'role': 'user', 'content': prompt})

        for i, msg in enumerate(thread):
            logger.info(f"[THREAD {i}] {msg['role'].upper()}: {msg['content']}")
        return thread


    def _determine_input_variables(self, fetch_new_documents):
        """Determine which inputs to pass to the LLM chain based on prompt usage."""
        variables = ["context", "question"]
        if "{history}" in os.getenv(self._get_prompt_key("rag_question_initial"), "") or "{history}" in os.getenv(self._get_prompt_key("rag_question_followup"), ""):
            variables.append("history")
        return variables

    def _create_prompt_template(self, thread, input_variables, context, user_query, history):
        rendered_prompt = self._render_chat_template(thread)

        if "{context}" in rendered_prompt:
            context = escape_curly_braces(context)
            rendered_prompt = rendered_prompt.replace("{context}", context)

        if "{question}" in rendered_prompt:
            rendered_prompt = rendered_prompt.replace("{question}", user_query)

        if "{history}" in rendered_prompt and history:
            history_text = RAGHelper.format_history_for_prompt(history)
            logger.info(f"history context: {history_text}")
            rendered_prompt = rendered_prompt.replace("{history}", history_text)

        logger.info(f"rendered_prompt: \n{rendered_prompt}...")
        assert "{" not in rendered_prompt or "{{" in rendered_prompt, "Unescaped braces in final prompt!"
        assert "{{{{" not in rendered_prompt, "Over-escaped SharePoint link!"

        self.logger.debug(f"[PROMPT TEMPLATE] Final prompt:\n{rendered_prompt}")
        return PromptTemplate(input_variables=[], template=rendered_prompt)

    def _create_llm_chain(self, fetch_new_documents, prompt):
        """Create the LLM chain for invoking the RAG pipeline."""
        return prompt | self.llm

   
    def _invoke_rag_chain(self, user_query, llm_chain, history=None):
        """Invoke the RAG pipeline using a manually rendered prompt template — no input variables needed."""
        
        # Extract plain string query
        if isinstance(user_query, str):
            query_str = user_query
        elif isinstance(user_query, dict):
            query_str = user_query.get("question", "").strip()
        else:
            raise ValueError(f"user_query must be str or dict, got {type(user_query)}")

        self.logger.warning(f"[RAG_CHAIN] user_query: {user_query}")
        self.logger.warning(f"[RAG_CHAIN] query_str: {query_str}")

        try:
            context_docs = self.ensemble_retriever.invoke(query_str)
        except Exception as e:
            self.logger.error(f"Failed to get documents from retriever: {e}")
            raise

        context = RAGHelper.format_documents(context_docs)
        assert "{" not in context or "{{" in context, "Unescaped braces in context!"

        if history:
            self.logger.debug(f"[RAG_CHAIN] Raw history:\n{history}")
        else:
            self.logger.info("[RAG_CHAIN] No history passed.")

        self.logger.info("[RAG_CHAIN] Final inputs to LLMChain: {} (none needed)")
        llm_response = llm_chain.invoke({})

        # Wrap plain string output into dict if needed
        if isinstance(llm_response, str):
            llm_response = {"text": llm_response}

        llm_response["docs"] = context_docs
        return llm_response

    @staticmethod
    def _extract_reply(response):
        llm_type = os.getenv("llm_type", "llama_hf")
        if llm_type == "falcon":
            end_string = "<|assistant|>"
        else:
            end_string = os.getenv("llm_assistant_token", "<|eot_id|>assistant\n\n")
        if isinstance(response, str):
            text = response
        elif isinstance(response, dict):
            text = response.get("text", "")
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
        logger.info(f"💬 Raw LLM response: {text}")

            # Falcon-style extraction
        if llm_type == "falcon":
            # Find last <|assistant|> and extract until end or next <|user|>
            pattern = r"<\|assistant\|>\s*(.*?)\s*(?=<\|user\|>|$)"
            matches = re.findall(pattern, text, flags=re.DOTALL)
            if matches:
                reply = matches[-1].strip()
                logger.debug(f"[FALCON EXTRACT] Extracted reply: {reply[:300]}")
                return reply

            logger.warning("[FALCON EXTRACT] No <|assistant|> match found — using full output")
            return text.strip()


        if llm_type in {"qwen3", "qwen2_5"}:
            pattern = r"<\|im_start\|>assistant\s*(.*)"
            match = re.search(pattern, text, flags=re.DOTALL)
            if match:
                reply = match.group(1).strip()
                logger.debug(f"[QWEN EXTRACT] Final reply extracted: {reply[:300]}")
                return reply
            logger.warning("[QWEN EXTRACT] No '<|im_start|>assistant' found — using full response")
            return text.strip()
        
        if llm_type == "llama3_3b":
            # LLaMA 3.2 format: Look for <|eot_id|>assistant and extract everything after
            pattern = r"<\|eot_id\|>assistant\s*(.*)"
            match = re.search(pattern, text, flags=re.DOTALL)
            if match:
                reply = match.group(1).strip()
                logger.debug(f"[LLAMA3.2 EXTRACT] Found <|eot_id|>assistant: {reply[:300]}")
                return reply
            logger.warning("[LLAMA3.2 EXTRACT] No '<|eot_id|>assistant' found — using full response.")
            return text.strip()
    
        if not end_string or end_string not in text:
            logger.info("[WARN] Assistant token not found in response. Using full output.")
            return text.strip()

        try:
            index = text.rindex(end_string)
            reply = text[index + len(end_string):]
            if not reply.strip():
                raise ValueError("Empty reply after marker.")
        except ValueError:
            logger.debug("[WARN] Assistant token not found or empty reply. Returning full response.")
            reply = text  

        return reply.strip()
    
    def _track_provenance(self, user_query, reply, thread):
        """Track the provenance of the LLM response and annotate documents with provenance scores."""
        provenance_method = os.getenv("provenance_method")
        logger.info(f"[DEBUG] provenance_method = {provenance_method}")
        self.logger.info("Raw reply before provenance: %s", reply.get("text", "NO TEXT FIELD"))
        if provenance_method in ['rerank', 'attention', 'similarity', 'llm']:
            logger.info(f"Before extract_reply: {reply.get('text', '')[:300]}")
            answer = self._extract_reply(reply)
            logger.info(f"After extract_reply: {answer[:300]}")
            
            logger.info(f"Final LLM answer before provenance: {repr(answer)}")
            logger.warning(f"[DEBUG] Answer length: {len(answer)}")
        if len(answer.strip()) < 10:
            logger.error("Very short or empty answer — reranker input might be broken.")
            if not answer:
                self.logger.warning("Empty answer extracted — skipping provenance scoring.")
                return
        # Only pass in the keys i actually use in the prompt templates
        context = escape_curly_braces(RAGHelper.format_documents(reply['docs']))
        safe_vars = {
            "context": context,
            "question": user_query
        }

        new_history = []
        for msg in thread:
            try:
                msg_text = msg["content"]
                formatted_content = msg_text.format_map(safe_vars) 
            except KeyError as e:
                logger.warning(f"Missing key in prompt template: {e}. Using raw content.")
                formatted_content = msg["content"]
            new_history.append({"role": msg["role"], "content": formatted_content})

        new_history.append({"role": "assistant", "content": answer})
        context_chunks = context.split("\n\n<NEWDOC>\n\n")

        provenance_scores = self._compute_provenance(
            provenance_method, user_query, reply, context_chunks, answer, new_history
        )

        self._aggregate_provenance_scores(reply, provenance_scores)
                
    def _aggregate_provenance_scores(self, reply, provenance_scores):
        doc_scores = defaultdict(list)
        doc_examples = {}
        logger.info("Input provenance_scores: %s", provenance_scores)
        logger.info("Raw document metadata before aggregation:")
        for doc in reply['docs']:
            logger.info("- %s", doc.metadata)

        # Group scores by document key
        for doc, score in zip(reply['docs'], provenance_scores):
            sharepoint_url = doc.metadata.get("sharepoint_url", "").strip()
            source = doc.metadata.get("source", "").strip()
            doc_id = doc.metadata.get("id")
            key = sharepoint_url if sharepoint_url else (source if source else doc_id)

            doc_scores[key].append(score)

            if key not in doc_examples:
                doc_examples[key] = doc
        
        logger.debug("Raw provenance scores per document:")
        for key, scores in doc_scores.items():
            logger.debug(f"- {key}: scores = {[round(s, 4) for s in scores]}")

        # Compute normalization if needed
        all_scores = [s for scores in doc_scores.values() for s in scores]
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 1
        score_range = max_score - min_score if max_score != min_score else 1
        
        # Debug: normalization bounds
        logger.debug(f"Normalization: min={min_score:.4f}, max={max_score:.4f}, range={score_range:.4f}")

        aggregated = []
        for key, scores in doc_scores.items():
            avg_score = sum(scores) / len(scores)

            # [DEBUG] Log before normalization
            self.logger.info(
                f"[DEBUG] Aggregating doc: {key} | raw scores={scores} | avg={avg_score:.4f} | min={min_score:.4f} | max={max_score:.4f}"
            )

            if self.normalize_provenance:
                normalized_score = (avg_score - min_score) / score_range
                # [DEBUG] Log normalized score
                self.logger.info(f"[DEBUG] Normalized score for {key}: {normalized_score:.4f}")
            else:
                normalized_score = avg_score

            doc = doc_examples[key]
            doc.metadata['provenance'] = normalized_score
            aggregated.append(doc)

        # Sort by provenance descending
        aggregated = sorted(aggregated, key=lambda d: d.metadata.get('provenance', 0), reverse=True)

        logger.info("Aggregated Provenance:")
        for doc in aggregated:
            key = doc.metadata.get("sharepoint_url") or doc.metadata.get("source") or doc.metadata.get("id")
            logger.info(f"- {key}: provenance={doc.metadata['provenance']:.2f}")

        reply['docs'] = aggregated
        

    def _compute_provenance(self, provenance_method, user_query, reply, context, answer, new_history):
        """Compute provenance scores based on the selected method."""
        if provenance_method == "rerank":
            return self._compute_rerank_provenance(user_query, reply, answer)
        if provenance_method == "attention":
            return compute_attention(self.model, self.tokenizer,
                                     self._render_chat_template(new_history), user_query,
                                     context, answer)
        if provenance_method == "similarity":
            return self.attributor.compute_similarity(user_query, context, answer)
        if provenance_method == "llm":
            return compute_llm_provenance(self.tokenizer, self.model, user_query, context, answer)
        return []

    def _compute_rerank_provenance(self, user_query, reply, answer):
        """Compute rerank-based provenance for the documents."""
        logger.debug("[CHECK] Entered _compute_rerank_provenance")
        if not os.getenv("rerank") == "True":
            raise ValueError(
                "Provenance attribution is set to rerank but reranking is not enabled. Please choose another provenance method or enable reranking.")
        logger.info(f"[DEBUG] Calling reranker with {len(reply['docs'])} docs")
        logger.warning(f"[RERANK DEBUG] self.compressor = {self.compressor}")
        reranked_docs = compute_rerank_provenance(self.compressor, user_query, reply['docs'], answer)
        logger.info(f"[DEBUG] Reranker returned: {reranked_docs}")
        if not reranked_docs or any([getattr(doc, "score", None) is None for doc in reranked_docs]):
            logger.info("[RERANK ERROR] No valid scores returned!")
        for i, doc in enumerate(reranked_docs):
            content_snippet = doc.page_content[:150].replace("\n", " ").strip()
            metadata = doc.metadata
            score = metadata.get("relevance_score", "N/A")
            logger.info(f"[RERANK DEBUG] Doc {i} - Score: {score}, ID: {metadata.get('id')}")

            self.logger.info(f"[RERANKED DOC {i}] Score: {float(score):.4f} | Snippet: '{content_snippet}'")

            try:
                safe_metadata = {k: _safe_json(v) for k, v in metadata.items()}
                logger.info(f"[RERANKED DOC {i}] Metadata: {json.dumps(safe_metadata, indent=2)}")
            except Exception as e:
                logger.warning(f"[RERANKED DOC {i}] Could not serialize metadata due to: {e}")
                logger.debug(f"[RERANKED DOC {i}] Raw metadata: {metadata}")

        return [d.metadata['relevance_score'] for d in reranked_docs if
                d.page_content in [doc.page_content for doc in reply['docs']]]
