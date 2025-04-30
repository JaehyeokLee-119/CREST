# CREST
This repository contains the code for **Consistency-driven Rationale Evaluation for Self-Training (CREST)**.<br>
**CREST** evaluates model-generated Chain-of-Thought rationales by checking whether they lead to consistent answers in follow-up questions. It then uses the evaluation results to train the model during supervised fine-tuning and preference learning.
This process improves both the model's reasoning performance and its rationale generation ability.
<br>
My paper [*Self-Training Meets Consistency: Improving LLMs’ Reasoning with Consistency-Driven Rationale Evaluation*](https://aclanthology.org/2025.naacl-long.528) is accepted to **NAACL 2025 main conference**.

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
@inproceedings{lee-etal-2025-self,
    title = "Self-Training Meets Consistency: Improving {LLM}s' Reasoning with Consistency-Driven Rationale Evaluation",
    author = "Lee, Jaehyeok  and
      Sakaguchi, Keisuke  and
      Bak, JinYeong",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.528/",
    pages = "10519--10539",
    ISBN = "979-8-89176-189-6",
    abstract = "Self-training approach for large language models (LLMs) improves reasoning abilities by training the models on their self-generated rationales. Previous approaches have labeled rationales that produce correct answers for a given question as appropriate for training. However, a single measure risks misjudging rationale quality, leading the models to learn flawed reasoning patterns. To address this issue, we propose CREST (Consistency-driven Rationale Evaluation for Self-Training), a self-training framework that further evaluates each rationale through follow-up questions and leverages this evaluation to guide its training. Specifically, we introduce two methods: (1) filtering out rationales that frequently result in incorrect answers on follow-up questions and (2) preference learning based on mixed preferences from rationale evaluation results of both original and follow-up questions. Experiments on three question-answering datasets using open LLMs show that CREST not only improves the logical robustness and correctness of rationales but also improves reasoning abilities compared to previous self-training approaches."
}
```
