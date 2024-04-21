# samples single run output from HuggingFace model (works with any model with chat template interface)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"
# mistralai/Mistral-7B-Instruct-v0.2, meta-llama/Llama-2-7b-chat-hf, google/gemma-2b-it
model_id = "EleutherAI/llemma_7b_muinstruct_camelmath"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

answer_context = [
        #     {"role": "user", "content": "What is the result of 26+1*6+7-23*14?"},
            {"role": "user", "content": "Calculate the result of 26+1*6+7-23*14. Show your work step by step."},
                    ]

model_inputs = tokenizer.apply_chat_template(answer_context, return_tensors="pt", padding=True, return_attention_mask=True, truncation=True)
outputs = model.generate(model_inputs.to(device), max_length=512, do_sample=True, pad_token_id=model.config.eos_token_id) 
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

# model_inputs = encodeds.to(device)
# model.to(device)

# generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
# decoded = tokenizer.batch_decode(generated_ids)
# # print(decoded[0])
answer = outputs[0].split("[/INST]")[-1].strip()

print(answer)