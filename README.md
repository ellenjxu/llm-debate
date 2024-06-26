# llm-debate

Runs multi-agent debate with open-source HuggingFace models on the Arithmetic problem.

`gen_math.py` runs debates with HuggingFace model (e.g. mistralai/Mistral-7B-Instruct-v0.2)

To reproduce the figure in original paper (scaling with rounds and agents):

1. set agents/rounds and run `./gen_math.sh` training script
2. generate figures in `outputs.ipynb`

`gen_math_panel.py` and `./gen_math.sh` are modified to run a panel experiment with multiple different HuggingFace models. Models are specified from a list of available options by passing in indices as command line arguments.

## Scaling agents and rounds

![image](https://github.com/ellenjxu/llm-debate/assets/56745453/bc784d3e-49d9-4c37-bc3e-e2bc99f5b345)

Reproduced figures

![image](https://github.com/ellenjxu/llm-debate/assets/56745453/cb7763e6-7827-496f-8e7c-959a2a57392a)

## Panel experiment

Testing with diverse panel of HuggingFace open source models
![image](https://github.com/ellenjxu/llm-debate/assets/56745453/3d48d666-0059-4ff8-9d2b-6feb704bf580)

| model            | Arithmetic (%) | std |
| ---------------- | ------ | --------- |
| Single Agent (Mistral) | 16 | 7.3    |
| Single Agent Panel (Mistral)           | 28 | 8.9    |
| Multi Agent Panel           | 24 | 8.5    |

---

[Original paper](https://arxiv.org/abs/2305.14325)
[Github](https://github.com/composable-models/llm_multiagent_debate/tree/main)
