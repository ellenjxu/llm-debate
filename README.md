# llm-debate

Runs multi-agent debate with open-source HuggingFace models on the Arithmetic problem.

`gen_math.py` has code for running debates with HuggingFace model (e.g. mistralai/Mistral-7B-Instruct-v0.2)

To reproduce the figure in original paper (scaling with rounds and agents):
1. set agents/rounds and run `./gen_math.sh` training script
2. generate figures in `outputs.ipynb`

## Original

![image](https://github.com/ellenjxu/llm-debate/assets/56745453/bc784d3e-49d9-4c37-bc3e-e2bc99f5b345)


## Reproduced figures

![image](https://github.com/ellenjxu/llm-debate/assets/56745453/cb7763e6-7827-496f-8e7c-959a2a57392a)

---

[Original paper](https://arxiv.org/abs/2305.14325)
[Github](https://github.com/composable-models/llm_multiagent_debate/tree/main)
