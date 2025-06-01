import os
import random
import logging
import pandas as pd
from random import sample
from dotenv import load_dotenv

from RAGHelper_local import RAGHelperLocal
from RAGHelper import RAGHelper

from datasets import Dataset
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.run_config import RunConfig

from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms.base import BaseRagasLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import LLMResult, Generation, PromptValue
import re
import time
from cost_estimator import estimate_cost_per_model
from faccor import get_faccor_metrics

logger = logging.getLogger(__name__)

class GeminiRagasLLM(BaseRagasLLM):
    def __init__(self, model_name: str, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def generate_text(self, prompt: PromptValue, **kwargs) -> LLMResult:
        prompt_str = prompt.to_string()
        logger.info(" [Gemini] Generating judgment for prompt:\n%s", prompt_str[:500])
        text = self.llm.invoke(prompt_str).content
        cleaned = self._strip_code_fence(text)
        logger.info("[Gemini] Output:\n%s", cleaned.strip())
        return LLMResult(generations=[[Generation(text=cleaned)]])

    async def agenerate_text(self, prompt: PromptValue, **kwargs) -> LLMResult:
        prompt_str = prompt.to_string()
        logger.info(" [Gemini] (Async) Generating judgment for prompt:\n%s", prompt_str[:500])
        text = (await self.llm.ainvoke(prompt_str)).content
        cleaned = self._strip_code_fence(text)
        logger.info("[Gemini] (Async) Output:\n%s", cleaned.strip())
        return LLMResult(generations=[[Generation(text=cleaned)]])

    def _strip_code_fence(self, text: str) -> str:
        return re.sub(r"^```(?:json)?\n(.*?)\n```$", r"\1", text.strip(), flags=re.DOTALL)

#################### Setup ####################
load_dotenv()
os.environ["use_rewrite_loop"] = "False"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

raghelper = RAGHelperLocal(logger)
raghelper.load_data()

#################### Sampling ####################
ragas_sample_size = int(os.getenv("ragas_sample_size"))
documents = raghelper.chunked_documents
document_sample = sample(documents, min(ragas_sample_size, len(documents)))

ragas_use_n_documents = int(os.getenv("rerank_k") if os.getenv("rerank") == "True" else os.getenv("vector_store_k"))
ragas_qa_pairs = int(os.getenv("ragas_qa_pairs"))
end_string = os.getenv("llm_assistant_token")

#################### QA generation via Gemini ####################
qa_generator = GeminiRagasLLM(model_name="gemini-1.5-pro", api_key=os.getenv("GOOGLE_API_KEY"))

question_template = os.getenv("ragas_question_instruction") + "\n" + os.getenv("ragas_question_query")
answer_template = os.getenv("ragas_answer_instruction") + "\n" + os.getenv("ragas_answer_query")

qa_pairs = []
seen_questions = set()

# Generate more than needed in case of duplicates
for idx in range(ragas_qa_pairs * 2):
    if len(qa_pairs) >= ragas_qa_pairs:
        break

    logger.info(f"Attempting QA pair {idx + 1}")
    selected_docs = random.sample(document_sample, ragas_use_n_documents)
    formatted_docs = RAGHelper.format_documents(selected_docs)

    question_prompt_str = question_template.format(context=formatted_docs)
    question_resp = qa_generator.llm.invoke(question_prompt_str).content.strip()

    if question_resp in seen_questions:
        logger.warning("Duplicate question detected, skipping: %s", question_resp)
        continue

    seen_questions.add(question_resp)
    logger.info("🤖 Synthetic Question: %s", question_resp)

    answer_prompt_str = answer_template.format(context=formatted_docs, question=question_resp)
    answer_resp = qa_generator.llm.invoke(answer_prompt_str).content.strip()
    logger.info("Ground Truth Answer: %s", answer_resp)

    qa_pairs.append({
        "question": question_resp,
        "ground_truth": answer_resp,
        "used_docs": [doc.metadata.get("source", "unknown") for doc in selected_docs]
    })

#################### Get actual RAG answers ####################
new_qa_pairs = []
rag_start_time = time.perf_counter()  # Start latency timer
for idx, qa_pair in enumerate(qa_pairs):
    logger.info(f"Running RAG for QA pair {idx + 1}/{len(qa_pairs)} — Q: {qa_pair['question'].strip()}")
    _, response = raghelper.handle_user_interaction(qa_pair["question"], [])
    rag_output = response['text']
    if end_string in rag_output:
        answer = rag_output[rag_output.rindex(end_string) + len(end_string):].strip()
    else:
        logger.warning("End string '%s' not found in output. Returning full output.", end_string)
        answer = rag_output.strip()
    logger.info("RAG Answer: %s", answer.strip())

    new_qa_pairs.append({
        "question": qa_pair["question"],
        "answer": answer,
        "contexts": [doc.page_content for doc in response["docs"]],
        "ground_truth": qa_pair["ground_truth"],
        "used_docs": qa_pair.get("used_docs", [])
    })
    
rag_end_time = time.perf_counter()  # End latency timer
total_latency = rag_end_time - rag_start_time
average_latency = total_latency / len(qa_pairs)
#################### Build Dataset ####################
dataset = Dataset.from_list(new_qa_pairs)
# dataset.save_to_disk(os.getenv("ragas_dataset"))

#################### Embedding wrapper ####################
gist_embedding = HuggingFaceEmbeddings(model_name="avsolatorio/GIST-small-Embedding-v0")
embedding_wrapper = LangchainEmbeddingsWrapper(embeddings=gist_embedding)

#################### Judge LLM using Gemini ####################
judge_llm = GeminiRagasLLM(model_name="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
logger.info("Using Gemini as judge LLM.")

#################### Run RAGAS ####################
results = evaluate(
    dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    embeddings=embedding_wrapper,
    llm=judge_llm,
    run_config=RunConfig(timeout=12000)
)

per_question_df = results.to_pandas()
print("RAGAS Evaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.3f}")

#################### Compute FacCor + AnsCor Metrics ###################################
fac_cor_scores, ans_cor_scores, ans_sim_scores = [], [], []
tp_counts, fp_counts, fn_counts = [], [], []

for row in new_qa_pairs:
    faccor, anscor, anssim, tp, fp, fn = get_faccor_metrics(
        question=row["question"],
        ground_truth=row["ground_truth"],
        answer=row["answer"],
        gemini_llm=judge_llm
    )
    fac_cor_scores.append(faccor)
    ans_cor_scores.append(anscor)
    ans_sim_scores.append(anssim)
    tp_counts.append(tp)
    fp_counts.append(fp)
    fn_counts.append(fn)

# Add to DataFrame
per_question_df["factual_correctness"] = fac_cor_scores
per_question_df["answer_correctness"] = ans_cor_scores
per_question_df["answer_similarity"] = ans_sim_scores
per_question_df["true_positives"] = tp_counts
per_question_df["false_positives"] = fp_counts
per_question_df["false_negatives"] = fn_counts

llm_type = os.getenv("llm_type")
llm_model = os.getenv("llm_model")

# If using quantized model, override with llm_cpp_quant
if llm_type == "llama_cpp_quant":
    llm_model = os.getenv("llm_cpp_quant")
elif llm_type == "deepseek":
    llm_model = os.getenv("llm_deepseek")
elif llm_type == "falcon":
    llm_model = os.getenv("llm_falcon")
elif llm_type == "qwen3":
    llm_model = os.getenv("llm_qwen3")
elif llm_type == "qwen2_5":
    llm_model = os.getenv("llm_qwen2_5")
elif llm_type == "llama3_3b":
    llm_model = os.getenv("llama32_3b")
    
    
estimated_cost = estimate_cost_per_model(llm_model)

log_path = "ragas_eval_log_fake.csv"
entry = {
    "llm_model": os.getenv("llm_model"),
    "embedding_model": os.getenv("embedding_model"),
    "rerank_model": os.getenv("rerank_model"),
    "provenance_method": os.getenv("provenance_method"),
    "vector_store_k": os.getenv("vector_store_k"),
    "rerank_k": os.getenv("rerank_k"),
    "sample_size": ragas_sample_size,
    "qa_pairs": ragas_qa_pairs,
    "estimated_monthly_cost": estimated_cost,
    "total_latency_seconds": round(total_latency, 2),
    "avg_latency_per_answer": round(average_latency, 2),
    "avg_factual_correctness": round(sum(fac_cor_scores) / len(fac_cor_scores), 3),
    "avg_answer_correctness": round(sum(ans_cor_scores) / len(ans_cor_scores), 3),
    **{k: float(v) for k, v in results.items()}
}

df = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
df.to_csv(log_path, index=False)

print("\n Sample per-question evaluation:")
for i, row in per_question_df.iterrows():
    print(f"\n--- QA Pair {i+1} ---")
    print(f"Q: {row['question']}")
    print(f"A: {row['answer']}")
    print(f"Precision: {row['context_precision']:.3f} | Recall: {row['context_recall']:.3f} | Faithfulness: {row['faithfulness']:.3f} | Relevancy: {row['answer_relevancy']:.3f}")
per_question_df.to_csv("ragas_per_question_fake.csv", index=False)
print("Saved per-question results to ragas_per_question.csv")
print(f"Logged results to {log_path}")