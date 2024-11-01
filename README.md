# CREST
Each directory has:
- scripts: Bash scripts for running codes
- resources: Pre-processed datasets, templates, and few-shot examples we used
- src: source codes
    - 1_rationale_generation: Source code for rationale generation and evaluation in CREST
    - 2_supervised_fine_tuning: Source code for rationale filtering and supervised fine-tuning
    - 3_preference_learning: Source code for preference learning
    - analysis: FLASK rationale evaluation codes we used

### How to run
Python version: 3.12.2 <br>
Ensure you have all necessary dependencies by installing the required packages listed in `requirements.txt`:

```bash
conda create -n <ENV_NAME> python=3.12.2
conda activate <ENV_NAME>
pip install -r requirements.txt

home=$(pwd)
```

You can run CREST using Bash scripts located in the `/scripts` directory.
- stage 1: rationale generation and evaluation
- stage 2: supervised fine-tuning
- stage 3: preference learning with DPO

The initial settings are configured as follows: data=ReClor and base_model=Llama 3 8B_model. 
Please adjust the home directory path in each script file and modify the variables as needed for specific settings, such as the dataset or hyperparameters.

Then, `outputs` directory will be generated, which contains generated rationales and their evaluation results.
As the results of stage 2 and 3, `models` directory will be generated, which contains trained model from each stage.
