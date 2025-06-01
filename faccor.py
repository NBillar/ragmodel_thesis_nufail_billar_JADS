import json
import time
import re
from sentence_transformers import SentenceTransformer, util
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

embedding_model_name = os.getenv("embedding_model", "avsolatorio/GIST-small-Embedding-v0")
semantic_model = SentenceTransformer(embedding_model_name)

faccor_prompt_template = os.getenv("faccor_prompt", '''You are evaluating factual overlap between a ground truth and a generated answer. 
Classify each sentence in the generated answer as follows:

- TP: Correct and present in the ground truth
- FP: Incorrect or not present in the ground truth
- FN: Present in the ground truth but missing in the generated answer

Return a JSON with three lists:
{{
  "true_positives": [...],
  "false_positives": [...],
  "false_negatives": [...]
}}

QUESTION: {question}
GROUND TRUTH: {ground_truth}
ANSWER: {answer}''')

def get_faccor_metrics(question, ground_truth, answer, gemini_llm, max_retries=3):
    prompt = faccor_prompt_template.format(question=question, ground_truth=ground_truth, answer=answer)

    for attempt in range(max_retries):
        response = gemini_llm.llm.invoke(prompt).content.strip()

        # Remove markdown-style code fences if present
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\n(.*?)\n```$", r"\1", response, flags=re.DOTALL)

        try:
            parsed = json.loads(response)

            # Validate expected format
            if isinstance(parsed, dict) and all(k in parsed for k in ["true_positives", "false_positives", "false_negatives"]):
                tp = len(parsed["true_positives"])
                fp = len(parsed["false_positives"])
                fn = len(parsed["false_negatives"])
                break  
            else:
                raise ValueError("Gemini response JSON is not in expected TP/FP/FN format.")

        except Exception as e:
            logger.warning(f"Failed to parse Gemini response on attempt {attempt + 1}: {e}\nRaw:\n{response}")
            time.sleep(1)
            if attempt == max_retries - 1:
                logger.error("All retries exhausted. Returning zeros.")
                return 0, 0, 0, 0, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0

    # === Compute FacCor and AnsCor ===
    faccor = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    ans_emb = semantic_model.encode(answer, convert_to_tensor=True)
    gt_emb = semantic_model.encode(ground_truth, convert_to_tensor=True)
    ans_sim = util.cos_sim(ans_emb, gt_emb).item()
    anscor = 0.7 * faccor + 0.3 * ans_sim

    return faccor, anscor, ans_sim, tp, fp, fn