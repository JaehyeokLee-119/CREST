import os
from typing import Dict, Optional
from torch.utils.data import DataLoader

from utils.utils import combine_query, result_evaluation

import json
import torch
from datasets import Dataset, load_dataset
from peft import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from tqdm import tqdm
import fire 


def get_paired_dataset(
    data_dir: str = None,
    sanity_check: bool = False,
    max_datas: Optional[int] = None,
    shuffle: Optional[bool] = False,
    num_proc=24,
    seed=42,
) -> Dataset:
    
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "id_string": samples["id_string"],
            "prompt": samples["query"],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    dataset = load_dataset(
        "json",
        data_files=data_dir,
        split="train",
    )
    original_columns = dataset.column_names
    
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))
    else:
        if max_datas:
            dataset = dataset.select(range(min(len(dataset), max_datas)))
            
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def main(
    # Model and Data
    base_model_name: str = None,
    model_dir: str = None,
    test_data: str = None,
    output_dir = None,
    test_batch_size = 8,
    test_template_path = None,
    test_max_new_tokens = 560,
):
    template_dict = json.load(open(test_template_path))
    test_instruction = template_dict['instruction']
    test_template = template_dict['first_input'] + '\n\n' + template_dict['oneshot_input'] + template_dict['final_output']
    
    # Saving Training arguments at Working directory
    os.makedirs(model_dir, exist_ok=True)
    script_args = locals().copy()
    with open(os.path.join(model_dir, "setting.json"), "w") as f:
        json.dump(script_args, f, indent=2)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # WANDB_PROJECT
    if output_dir == None:
        output_dir = os.path.join(model_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    #### Evaluate on Test Dataset
    def collate_fn_nolabel(inputs):
        inputs = [x.strip() for x in inputs]
        input_dict = tokenizer(inputs, padding=True, return_tensors="pt")
        return input_dict
    
    def collate_fn_nolabel_for_repredict(inputs):
        inputs = [x['input'].strip() for x in inputs]
        input_dict = tokenizer(inputs, padding=True, return_tensors="pt")
        return input_dict

    def inference(dataloader, model, tokenizer, max_new_tokens=560):
        outputs = []
        for batch in tqdm(dataloader, desc="Generating"):
            with torch.no_grad():
                if hasattr(model, 'current_device'):
                    input_ids = batch['input_ids'].to(model.current_device) # current_device
                    attention_mask = batch['attention_mask'].to(model.current_device) # current_device
                else:
                    input_ids = batch['input_ids'].to(model.device) # current_device
                    attention_mask = batch['attention_mask'].to(model.device) # current_device
            
                output = model.generate(input_ids=input_ids, 
                                    attention_mask=attention_mask, 
                                    max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id,
                                    early_stopping=True,
                                    do_sample=False,
                                    )
                input_decoded  = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
                
                output_decoded = [x.replace(input_decoded[ix], '').strip() for ix, x in enumerate(output_decoded)]
                outputs.extend(output_decoded)
        return outputs
    
    device = 'auto'
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map=device)
    model = PeftModelForCausalLM.from_pretrained(model, os.path.join(model_dir, "dpo_train"), device_map=device)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    
    ''' 2) Prediction '''
    ## 1. First Prediction
    inference_output_path = os.path.join(output_dir, "inference_result_data.jsonl")
    repredicted_output_path = os.path.join(output_dir, f'inference_result_data-repredicted.jsonl')
    test_dataset = load_dataset("json", data_files=test_data, split="train")
    if '.jsonl' in test_data:
        test_dataset = test_dataset.rename_column('context', 'passage')
        
    # input_strings = [combine_query(ex, template=test_template, instruction=test_instruction) for ex in test_dataset]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    input_strings = [combine_query(ex, template=test_template, instruction=test_instruction, 
                                    tokenizer=tokenizer) for ex in test_dataset]
    
    dataloader = DataLoader(input_strings, batch_size=test_batch_size, collate_fn=collate_fn_nolabel)
    
    result_dict_list = []
    outputs = inference(dataloader, model, tokenizer, max_new_tokens=test_max_new_tokens)
    for i in range(len(outputs)):
        current_result = {
            "id_string": test_dataset['id_string'][i],
            "input": input_strings[i],
            "label": test_dataset['label'][i],
            "output": outputs[i]
        }
        result_dict_list.append(current_result)
            
    json.dump(result_dict_list, open(inference_output_path, 'w'), indent=2)
    
    ''' 3) Re-Prediction '''
    ## 2. Re-prediction
    predicted_test_dataset_dict = json.load(open(inference_output_path))
    repredict_test_dataset = load_dataset("json", data_files=inference_output_path, split="train")
    
    previous_outputs = repredict_test_dataset['output']
    index_no_answer = [i for i, ex in enumerate(previous_outputs) if '[Answer]' not in ex]
    input_strings_no_answer = [repredict_test_dataset[i] for i in index_no_answer]
    
    for i in range(len(input_strings_no_answer)):
        input_strings_no_answer[i]['input'] += input_strings_no_answer[i]['output'] + '\n\n[Answer]\nThe correct answer is '
    
    dataloader_repredict = DataLoader(input_strings_no_answer, batch_size=test_batch_size, collate_fn=collate_fn_nolabel_for_repredict)
    outputs_repredict = inference(dataloader_repredict, model, tokenizer, max_new_tokens=test_max_new_tokens)
    
    print(f"Repredicted {len(outputs_repredict)} answers")
    
    for ix, i in enumerate(index_no_answer):
        predicted_test_dataset_dict[i]['input'] = input_strings_no_answer[ix]['input']
        predicted_test_dataset_dict[i]['output'] = f'[Answer] The correct answer is {outputs_repredict[ix]}'
    
    json.dump(predicted_test_dataset_dict, open(repredicted_output_path, 'w'), indent=2)
    
    r_total_correct, r_extracted_ratio, r_extracted_correct = result_evaluation(repredicted_output_path)
    total_correct, extracted_ratio, extracted_correct = result_evaluation(inference_output_path)
    output_statis_dict = {
        "test/total_correct": total_correct,
        "test/extracted_ratio": extracted_ratio,
        "test/extracted_correct": extracted_correct,
        "test/repredicted_total_correct": r_total_correct,
        "test/repredicted_extracted_ratio": r_extracted_ratio,
        "test/repredicted_extracted_correct": r_extracted_correct,
    }
    
    json.dump(output_statis_dict, open(os.path.join(output_dir, "test_result.json"), 'w'), indent=4)
    
    model = None
    del model
    torch.cuda.empty_cache()
        

if __name__ == "__main__":
    fire.Fire(main)