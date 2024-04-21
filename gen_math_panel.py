"""Generate outputs from diverse set of HuggingFace models"""

# import openai
import os
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

model_names = ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf", "google/gemma-2b-it"]

models = {}
tokenizers = {}

for model_name in model_names:
    print(f"Loading model and tokenizer for {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set for models that need it
    
    models[model_name] = model
    tokenizers[model_name] = tokenizer

print("All models and tokenizers loaded.")

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def generate_answer(answer_context, model_name):
    model = models[model_name]
    tokenizer = tokenizers[model_name]

    model_inputs = tokenizer.apply_chat_template(answer_context, return_tensors="pt", padding=True, return_attention_mask=True, truncation=True)
    
    outputs = model.generate(model_inputs.to(device), max_length=8192, do_sample=True, pad_token_id=model.config.eos_token_id) 
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    answer = outputs[0].split("[/INST]")[-1].strip()
    return answer

def construct_message(agents, question, idx):

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    # content = completion["choices"][0]["message"]["content"]
    content = completion
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


if __name__ == "__main__":
    # answer = parse_answer("My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--evals", type=int, default=25) # 100
    parser.add_argument("--models", nargs='+', type=int, help="List of model indices for each agent")


    args = parser.parse_args()

    agents = len(args.models)
    rounds = args.rounds
    agent_model_choices = [model_names[idx] for idx in args.models]

    np.random.seed(0)

    evaluation_round = args.evals
    scores = []

    generated_description = {}
    performance = []

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_dir = "output/{}".format(timestamp)

    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """Calculate the result of {}+{}*{}+{}-{}*{}. Show your work step by step. Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)] # reinitialize every round
        # agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)] # reinitialize every round

        # content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt, 2*round - 1)
                    agent_context.append(message)

                    # print("message: ", message)

                model_name = agent_model_choices[i]
                completion = generate_answer(agent_context, model_name)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                # print(completion)

        text_answers = []

        for agent_context in agent_contexts:
            text_answer = agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        generated_description[f"{a}+{b}*{c}+{d}-{e}*{f}"] = (agent_contexts, answer)

        try:
            text_answer = most_frequent(text_answers)
            if int(text_answer) == int(answer):
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

        print(f"agent answer: {text_answer} answer: {answer}")
        print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))

        performance.append(f"{a}+{b}*{c}+{d}-{e}*{f},{answer},{text_answer}")
        performance.append(f"{np.mean(scores)},{np.std(scores) / (len(scores) ** 0.5)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    indices = "_".join([str(idx) for idx in args.models])
    with open(os.path.join(output_dir, "math_agents{}_rounds{}_models{}.txt".format(agents, rounds, indices)), "w") as f:
        for p in performance:
            f.write("{}\n".format(p))

    pickle.dump(generated_description, open(os.path.join(output_dir, "math_agents{}_rounds{}_models{}.p".format(agents, rounds, indices)), "wb"))
