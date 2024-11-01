import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import fire
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import wandb

from utils.utils import generate_option_string, generate_prompt_test

class ListDataset(Dataset):
     def __init__(self, original_list):
        self.original_list = original_list
     def __len__(self):
        return len(self.original_list)

     def __getitem__(self, i):
        return self.original_list[i]
    
def is_reprediction_target(generated: str):
    generated = generated.strip()
    if '[Answer]' in generated:
        generated = generated.split('[Answer]')[1].strip()
    
    if '[Question]' in generated:
        generated = generated.split('[Question]')[0].strip()
        
    labels = ['A.', 'B.', 'C.', 'D.', 'E.']
    for label in labels:
        if label in generated:
            return False
    return True


def extract_label_from_output(generated: str):
    generated = generated.strip()
    if '[Answer]' in generated:
        generated = generated.split('[Answer]')[1].strip()
    
    if '[Question]' in generated:
        generated = generated.split('[Question]')[0].strip()
        
    labels = ['A', 'B', 'C', 'D', 'E']
    for i in generated:
        if i in labels:
            return i
    return 'X'

def construct_fewshot(oneshot_input, oneshot_output, fewshot_path, CoT):
    fewshots = []
    fewshot_data = json.load(open(fewshot_path, "r"))
    
    for oneshot in fewshot_data:
        question = oneshot['question'].strip() if oneshot['question'] != None else ''
        option_string = generate_option_string(oneshot['answers']).strip()
        rationale = oneshot['rationale'].strip() if oneshot['rationale'] != None else ''
        given_answer = oneshot['label'].strip() if oneshot['label'] != None else ''
        passage = oneshot['context'].strip() if oneshot['context'] != None else ''
        
        input_string = oneshot_input.format(
            question=question,
            passage=passage,
            option_string=option_string
        )
        
        if CoT:
            fewshot = input_string + '\n' + oneshot_output.format(
                rationale=rationale,
                given_answer=given_answer
            )
        else:
            fewshot = input_string + "\n\n" + oneshot_output.format(
                given_answer=given_answer
            )
            
        fewshots.append(fewshot)
    
    return "\n\n".join(fewshots)

def main(
    base_model: str = "",
    eval_data_path: str = None,
    model_dir = None, 
    output_dir = None,
    CoT = False,
    max_new_tokens = 15,
    batch_size=2,
    template_dict_path = None,
    fewshot_path = None,
):
    # visible devices 
    print(f"visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    if output_dir == None:
        output_dir = model_dir
        assert model_dir != None, "model_dir should be given if output_dir is None"
    
    if type(CoT) == str:
        CoT = CoT.lower() == 'true'
    print(f"Use CoT: {CoT}")
        
    eval_data = load_dataset("json", data_files=eval_data_path, split='train')
    
    os.makedirs(output_dir, exist_ok=True)
    
    template_dict = json.load(open(template_dict_path, "r"))
    
    if fewshot_path == None:
        instruction = template_dict['instruction']
        template = '\n\n'.join([template_dict['first_input'], template_dict['oneshot_input'], template_dict['final_output']]).strip()
    else:
        instruction = template_dict['instruction']
        fewshot = construct_fewshot(oneshot_input=template_dict['oneshot_input'], oneshot_output=template_dict['oneshot_output'], fewshot_path=fewshot_path, CoT=CoT)
        template = '\n\n'.join([template_dict['first_input'], fewshot, template_dict['oneshot_input'], template_dict['final_output']]).strip()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    prompt_list = generate_prompt_test(eval_data, tokenizer, instruction=instruction, template=template)
    
    prompt_list = ListDataset(prompt_list)
    
    if model_dir != None:  
        finetune_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map={"":0})
    else:
        finetune_model = AutoModelForCausalLM.from_pretrained(base_model, device_map={"":0})
    
    pipe_finetuned = pipeline("text-generation", 
                              model=finetune_model, 
                              tokenizer=tokenizer, 
                              max_new_tokens=max_new_tokens,
                              do_sample=False)
    
    ### First Prediction ### 
    outputs = []
    pipe_finetuned.tokenizer.pad_token_id = finetune_model.config.eos_token_id
    for out in tqdm(pipe_finetuned(prompt_list, batch_size=batch_size, return_full_text=False), desc="First Prediction"):
        outputs.append(out)
    
    first_result_dicts = []
    for i in range(len(outputs)):
        generated = outputs[i][0]['generated_text']
        result_dict = {
            'id_string': eval_data[i]['id_string'],
            'label': eval_data[i]['label'],
            'input_string': prompt_list[i],
            'generated_text': generated,
            'prediction': extract_label_from_output(generated) 
        }
        first_result_dicts.append(result_dict)
    
    total = len(first_result_dicts)
    correct = 0
    unextracted = 0

    for result in first_result_dicts:
        if result['label'] == result['prediction']:
            correct += 1
        if result['prediction'] == 'X':
            unextracted += 1    
    
    eval_result_dict = {
        'accuracy': (correct / total) if total > 0 else 0,
        'extracted_ratio': ((total - unextracted) / total) if total > 0 else 0,
        'extracted_accuracy': (correct / (total - unextracted)) if total - unextracted > 0 else 0,
        'total': total,
        'correct': correct,
        'extracted': total - unextracted,
    }
            
    json.dump(eval_result_dict, open(os.path.join(output_dir, "eval_result.json"), "w"), indent=4)
    json.dump(first_result_dicts, open(os.path.join(output_dir, "eval_result_dicts.json"), "w"), indent=4)
    print("Evaluation result saved at", f"{output_dir}/eval_result.json")
    
    if CoT == True:
        ### Second Prediction (reprediction) Target ###
        reprediction_target_idx = []
        for i in range(len(first_result_dicts)):
            if is_reprediction_target(first_result_dicts[i]['generated_text']):
                reprediction_target_idx.append(i)

        reprediction_prompt_list = [prompt_list[i] for i in reprediction_target_idx]
        for idx, i in enumerate(reprediction_target_idx):
            reprediction_prompt_list[idx] += first_result_dicts[i]['generated_text'] \
                + "[Answer]\nTherefore, the correct answer is"
        
        reprediction_prompt_list = ListDataset(reprediction_prompt_list)
        reprediction_outputs = []
        for out in tqdm(pipe_finetuned(reprediction_prompt_list, 
                                       batch_size=batch_size, 
                                       truncation="only_first",
                                       return_full_text=False
                        ), desc="Second Prediction"):
            reprediction_outputs.append(out)

        second_result_dicts = first_result_dicts.copy()
        for idx, i in enumerate(reprediction_target_idx):
            generated = reprediction_outputs[idx][0]['generated_text']
            second_result_dicts[i]['generated_text'] = generated
            second_result_dicts[i]['prediction'] = extract_label_from_output(generated)
        
        total = len(second_result_dicts)
        correct = 0
        unextracted = 0

        for result in second_result_dicts:
            if result['label'] == result['prediction']:
                correct += 1
            if result['prediction'] == 'X':
                unextracted += 1    
        
        second_eval_result_dicts = {
            'accuracy': correct / total,
            'extracted_ratio': (total - unextracted) / total,
            'extracted_accuracy': correct / (total - unextracted),
            'total': total,
            'correct': correct,
            'extracted': total - unextracted,
        }
        json.dump(second_result_dicts, open(os.path.join(output_dir, "eval_second_resul.json"), "w"), indent=4)
        json.dump(second_eval_result_dicts, open(os.path.join(output_dir, "eval_second_result_dicts.json"), "w"), indent=4)
        
    
if __name__ == "__main__":  
    import os 
    fire.Fire(main)