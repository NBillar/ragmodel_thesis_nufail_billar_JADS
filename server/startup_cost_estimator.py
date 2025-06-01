from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

start = time.time()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)

end = time.time()

# Size estimation in GB (assumes float16, so 2 bytes per param)
model_size_gb = sum(p.numel() for p in model.parameters()) * 2 / 1e9

print(f"✅ Model loaded in {end - start:.2f} seconds")
print(f"📦 Estimated model size: {model_size_gb:.2f} GB")