# https://huggingface.co/openlm-research/open_llama_3b?text=My+name+is+Thomas+and+my+main
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

device = 'cuda'

model_path = 'openlm-research/open_llama_3b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))