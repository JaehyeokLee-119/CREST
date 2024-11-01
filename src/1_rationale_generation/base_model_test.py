import fire
import os
import json
from tqdm import tqdm 
import utils.data_processing as dp
from utils.generation import Generator_vllm
from utils.utils import (
    construct_fewshot
)

def main(
    base_model_name: str = "meta-llama/Meta-Llama-3-8B",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_gen_len: int = 512,
    batch_size: int = 16,
    data: str = None, # train_data path
    correct_n: int = 16,
    incorrect_n: int = 0,
    output_directory: str = None,
    fewshot_samples_path = None,
    fewshot_count: int = 20, #3,
    template_file_path: str = None,
    CoT=True,
):
    os.makedirs(output_directory, exist_ok=True)
    
    # Load: original data
    with open(data) as f:
        jsonl_data = [json.loads(l) for l in f.readlines()]
    
    template_dict = json.load(open(template_file_path))
    
    if fewshot_samples_path is not None:
        Instruction = template_dict['instruction']
        fewshot = construct_fewshot(oneshot_input=template_dict['oneshot_input'], oneshot_output=template_dict['oneshot_output'], fewshot_path=fewshot_samples_path, CoT=CoT, fewshot_count=fewshot_count)
        Template_rationale_generation = template_dict['first_input'] + '\n\n' + '{fewshots}' + '\n\n' + \
            template_dict['oneshot_input'] + template_dict['final_output']
    else:        
        fewshot = None
        Instruction = template_dict['instruction']
        fewshot=''
        Template_rationale_generation = template_dict['first_input'] + '\n\n' + \
            template_dict['oneshot_input'] + template_dict['final_output']
    
    diverse_questions = []
    for i, row in enumerate(jsonl_data):
        diverse_questions += dp.diverse_question(template=Template_rationale_generation, instruction=Instruction, row=row, correct_n=correct_n, incorrect_n=incorrect_n, fewshots=fewshot)

    # Generate rationale
    input_prompts = [diverse_questions[i]['input_string'] for i in range(len(diverse_questions))]
    
    output_tmp= f'{output_directory}/1-rationale_tmp.json'
    results = []
    diverse_questions = [{k: v for k, v in diverse_questions[i].items() if k in ['id_string', 'given_answer']} for i in range(len(diverse_questions))]
    
    count = 0
    if os.path.exists(output_tmp):
        with open(output_tmp) as f:
            results = json.load(f)
        count = len(results)
        start_index = count
        print(f"Already processed until {count}th. Resuming from {count+1}th.")
    else:
        start_index = 0
    
    if count == len(diverse_questions):
        print("Already finished!")
    else:
        generator = Generator_vllm(model=base_model_name, 
                                stop_string="[", 
                                max_gen_length=max_gen_len, 
                                temperature=temperature, 
                                top_p=top_p)
        print("Generator built!")
        
        for i in tqdm(range(start_index, len(input_prompts), batch_size), desc="Rationale generation"):
            results += generator.generating(input_prompts[i:i+batch_size])
            diverse_questions_2 = [dict(diverse_questions[j], **results[j]) for j in range(len(results))]
            json.dump(diverse_questions_2, open(output_tmp, 'w'), indent=4) 
    
    diverse_questions = [dict(diverse_questions[i], **results[i]) for i in range(len(diverse_questions))]
    
    result_dicts = []
    for row in jsonl_data:
        result_dict = {
            "id_string": row['id_string'],
            "passage": row['context'],
            "question": row['question'],
            "answers": row['answers'],
            "label": row['label'],
            "rationales_for_given_answer": {}
        }
        answer_labels = ['A', 'B', 'C', 'D', 'E'][:len(row['answers'])]
        for label in answer_labels:
            rationale_list = [diverse_questions[i]['generated_text'] for i 
                              in range(len(diverse_questions)) 
                              if diverse_questions[i]['id_string'] == row['id_string'] and diverse_questions[i]['given_answer'] == label]
            result_dict["rationales_for_given_answer"][label] = rationale_list
            
        result_dicts.append(result_dict)
        
    # saves
    # templates
    output_metadata = {
        "model": base_model_name,
        "dataset_name": data,
        'Template': Template_rationale_generation,
        "Instruction": Instruction,       
        "max_gen_len": max_gen_len,
        "batch_size": batch_size,
        "temperature": temperature,
        "top_p": top_p,
        "fewshots":fewshot,
        "template_fewshot_oneshot":fewshot,
        "fewshot_counts": fewshot_count,
    }
    json.dump(output_metadata, open(f'{output_directory}/1-rationale_metadata.json', 'w'), indent=4)
    json.dump(result_dicts, open(f'{output_directory}/1-rationales.json', 'w'), indent=4)
    
    prediction_path = f'{output_directory}/1-rationales.json'
    prediction_data = json.load(open(prediction_path))
    
    result = []
    for data in tqdm(prediction_data):
        for given_answer in data["rationales_for_given_answer"]:
            data["rationales_for_given_answer"][given_answer] = [dp.rationale_refining(rationale) for rationale in data["rationales_for_given_answer"][given_answer]]
        result.append(data)
        
    with open(prediction_path, 'w') as f:
        json.dump(result, f, indent=4)

    # file removal (1-rationale_tmp.json)
    # os.remove(output_tmp)

if __name__ == "__main__":
    fire.Fire(main)
