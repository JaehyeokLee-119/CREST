import time
import os 
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import pickle
import json 
load_dotenv()
from request import ask_chatgpt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import fire 

# get/set openai API KEY
API_KEY = os.getenv("API_KEY")
SEED = 42

class GPT_run:
    def __init__(self, model_name, file_name, outputfile_name, num_threads=15):
        self.model_name = model_name
        self.client = OpenAI(api_key=API_KEY)
        self.num_threads = num_threads
        self.filename = file_name
        self.outputfile = outputfile_name
        
        self.data_df = self.read_file()
        
        self.data_df['input_string'] = np.nan
        self.data_df['inference'] = np.nan
        
    def read_file(self):
        data_df = pd.read_json(self.filename)
        return data_df
    
    def do_gpt(self, model_name, prompt):
        prompt = prompt
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        params = {
            "model": model_name,
            "messages": PROMPT_MESSAGES,
        }
        response = ask_chatgpt(params, self.client)
        return response
    
    def inference(self, inputs):
        response = self.do_gpt(self.model_name, inputs)
        return response
    
    def run(self):
        inference_template = """
We would like to request your feedback on the performance of the response of the assistant to the user instruction displayed below. In the feedback, I want you to rate the quality of the response in these 3 categories according to each score rubric:

[Skill 1. Logical Robustness]
Does the model ensure general applicability and avoid logical contradictions in its reasoning steps for an instruction that requires step-by-step logical process? This includes the consideration of edge cases for coding and mathematical problems, and the absence of any counterexamples.

Score 1: The logic of the model's response is completely incoherent.
Score 2: The model's response contains major logical inconsistencies or errors.
Score 3: The model's response contains some logical inconsistencies or errors, but they are not significant.
Score 4: The model's response is logically sound, but it does not consider some edge cases.
Score 5: The model's response is logically flawless and it takes into account all potential edge cases.


[Skill 2. Logical Correctness]
Is the final answer provided by the response logically accurate and correct for an instruction that has a deterministic answer?

Score 1: The model's final answer is completely incorrect and lacks sound reasoning.
Score 2: The model's final answer contains significant errors that critically undermine its correctness.
Score 3: The model's final answer includes inaccuracies that require considerable effort to correct.
Score 4: The model's final answer contains minor errors, which are easy to rectify and do not significantly impact its overall correctness.
Score 5: The model's final answer is completely accurate and sound.


[Skill 3. Logical Efficiency]
Is the response logically efficient? The logic behind the response should have no redundant step, remaining simple and efficient. For tasks involving coding, the proposed solution should also consider time complexity.

Score 1: The logic behind the response is significantly inefficient and redundant, necessitating a complete reorganization of logic for clarity and efficiency.
Score 2: The logic of the response lacks efficiency and conciseness, requiring a substantial reorganization for better optimization.
Score 3: The logic of the response is not efficient enough, necessitating major edits for improved optimization.
Score 4: The logic of the response is largely efficient, but it still has some redundant steps. It could be handled from minor edits for better optimization.
Score 5: The logic of the response is optimally efficient, requiring no further optimization.

[Instruction]
{question}

[Ground truth Answer]
{ground_truth_answer}

[Assistant's Response]
{answer}

[The End of Assistant's Response]

Please give feedback on the assistant's responses. Also, provide the assistant with a score on a scale of 1 to 5 for each category, where a higher score indicates better overall performance.

Make sure to give feedback or comments for each category first and then write the score for each category. Only write the feedback corresponding to the score rubric for each category. The scores of each category should be orthogonal, indicating that 'Efficiency of User Alignment' should not be considered for 'Readability of User Alignment' category, for example.

Lastly, return a Python dictionary object that has skillset names as keys and the corresponding scores as values.
""".strip()

        data_df = self.read_file()
        
        result_dict = json.load(open(self.filename))
        
        def get_instruction(row):
    
            def output_postprocessing(output):
                output1 = output.split('[Answer]')[0].strip()
                if len(output.split('[Answer]')) == 1:
                    output = output1
                else:
                    output2 = output.split('[Answer]')[1].strip()
                    output = output1 + ' ' +output2
                output.replace('[Answer]', ' ')
                return output.strip()
            
            instruction_template = "Read the question below carefully, write the intermediate reasoning steps about the problem and then choose an answer to the question."
            instruction_template += row['q_i']
            answer = output_postprocessing(row['output'])
            label = row['label']
            
            return inference_template.format(question=instruction_template, ground_truth_answer=label, answer=answer)

        inputs = []
        for idx, row in enumerate(result_dict):
            inputs.append(get_instruction(row))
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            outputs = list(tqdm(executor.map(self.inference, inputs), total=len(inputs)))
        
        data_df['input_string'] = inputs
        data_df['inference'] = outputs
        
        data_df.to_json(self.outputfile, orient='records', lines=True)
            
def set_request_params_simple(question, model_name):
    prompt = f"{question}"        
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    print(f"prompt: {prompt}")
    params = {
        "model": model_name, 
        "messages": PROMPT_MESSAGES,
        "seed": SEED,
    }
    return params
    
def main_run(
    result_dir = '',
    file = '',
    key = '',
    model_name = '',
    num_threads = 15,
):
    os.makedirs(result_dir, exist_ok=True)
    
    runner = GPT_run(model_name, 
        file, 
        f'{result_dir}/{key}_{model_name}.jsonl', 
        num_threads=num_threads)
    runner.run()
        

if __name__ == '__main__':
    fire.Fire(main_run)