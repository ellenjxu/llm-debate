from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").half().to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

answer_context = [
            {"role": "user", "content": "What is the result of 3+4*5+6-1*2?"},
                    ]

model_inputs = tokenizer.apply_chat_template(answer_context, return_tensors="pt")
outputs = model.generate(model_inputs.to(device), max_new_tokens=1000, do_sample=True)
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

# model_inputs = encodeds.to(device)
# model.to(device)

# generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
# decoded = tokenizer.batch_decode(generated_ids)
# # print(decoded[0])
answer = outputs.split("[/INST]")[-1].strip()
print(answer)
