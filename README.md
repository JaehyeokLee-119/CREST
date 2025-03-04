# CREST
This repository contains the code for **Consistency-driven Rationale Evaluation for Self-Training (CREST)**.<br>
**CREST** evaluates model-generated Chain-of-Thought rationales by checking whether they lead to consistent answers in follow-up questions. It then uses the evaluation results to train the model during supervised fine-tuning and preference learning.
This process improves both the model's reasoning performance and its rationale generation ability.
<br>
My paper [*Self-Training Meets Consistency: Improving LLMs’ Reasoning with Consistency-Driven Rationale Evaluation*](https://arxiv.org/abs/2411.06387) is accepted to **NAACL 2025 main conference**.

## Getting Started
Python version: 3.12.2 <br>
To set up your environment, you’ll need Python 3.12.2. Use the following commands to create a virtual environment and install the required libraries. Once set up, follow the order of scripts in the 'scripts' directory to run CREST.
```bash
conda create -n <ENV_NAME> python=3.12.2
conda activate <ENV_NAME>
pip install -r requirements.txt
home=$(pwd)
```
## Running CREST
Prepare datasets. Sample formats are in the `resources/data` directory.

You can run CREST using Bash scripts located in the `scripts` directory.
- `crest_stage_1.sh`: rationale generation and evaluation
- `crest_stage_2.sh`: supervised fine-tuning with rationale filtering
- `crest_stage_3.sh`: preference learning with DPO
- `direct_fine-tune_stage_2.sh`: codes for the experiments of direct fine-tuning approaches

Initial settings are configured with `data=ReClor` and `base_model=Llama 3 8B`. Make sure to update the home directory path like `home=$(pwd)`, as well as other variables (e.g., dataset, hyperparameters) as needed.

## Directory Structure
The main directories include:
- scripts: Contains bash scripts to run the code
- resources: Holds pre-processed datasets, templates, and few-shot examples used in CREST
- src: Source code directory
    - `1_rationale_generation`: Code for rationale generation and evaluation
    - `2_supervised_fine_tuning`: Code for rationale filtering and supervised fine-tuning
    - `3_preference_learning`: Code for preference learning
    - `analysis`: FLASK code used for rationale evaluation

After running CREST, an `outputs` directory will be created to store generated rationales and evaluation results. Following stages 2 and 3, a `models` directory will be generated, containing trained models from each stage.

## Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2024-00509258, Global AI Frontier Lab).

## Citation

```bibtex
@misc{lee2025selftrainingmeetsconsistencyimproving,
      title={Self-Training Meets Consistency: Improving LLMs' Reasoning with Consistency-Driven Rationale Evaluation}, 
      author={Jaehyeok Lee and Keisuke Sakaguchi and JinYeong Bak},
      year={2025},
      eprint={2411.06387},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.06387}, 
}
```
