# llama_cpp_wrapper.py

from langchain_core.runnables import Runnable
from typing import Dict, Any, Union, Optional
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompt_values import StringPromptValue

class LlamaCppLLMWrapper(Runnable):
    def __init__(self, model_path: str, temperature: float = 0.7, max_tokens: int = 1000):
        from llama_cpp import Llama
        self.model = Llama(model_path=model_path, n_ctx=32768)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(
        self,
        input: Union[str, Dict[str, Any], list],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Dict[str, str]:
        from langchain_core.prompt_values import StringPromptValue

        if isinstance(input, list):
            if len(input) != 1:
                raise ValueError("LlamaCppLLMWrapper only supports batch size 1.")
            input = input[0]

        if isinstance(input, StringPromptValue):
            input = input.text

        if isinstance(input, str):
            prompt = input
        elif isinstance(input, dict):
            if "messages" in input:
                messages = input["messages"]
                result = self.model.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return {"text": result['choices'][0]['message']['content']}
            elif "question" in input:
                prompt = input["question"]
            elif "input" in input:
                prompt = input["input"]
            elif input == {}:
                raise ValueError("Empty dict passed but no prompt provided")
            else:
                raise ValueError("Unsupported input dict structure for LlamaCppLLMWrapper.")
        else:
            raise ValueError(f"Expected string, dict, or StringPromptValue, got: {type(input)}")

        # Mode detection: chat-style vs instruction-style
        mode = config.get("mode") if config else None

        # Heuristic fallback: detect "Instruction:" prefix
        is_instruction = mode == "instruction" or prompt.strip().lower().startswith("instruction:")

        if is_instruction:
            # Raw completion mode
            result = self.model(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return {"text": result["choices"][0]["text"].strip()}
        else:
            # Default chat-style message mode
            messages = [{"role": "user", "content": prompt}]
            result = self.model.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return {"text": result['choices'][0]['message']['content']}
