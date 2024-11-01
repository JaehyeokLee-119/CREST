import fire
import os
from typing import List
import json
from tqdm import tqdm 
import utils.data_processing as dp
from utils.generation import Generator_vllm

def main(
    base_model_name: str = None,
    max_gen_len: int = 10,
    batch_size: int = 48,
    data: str = None,
    rationale_directory: str = None,
    template_file_path: str = None, 
    save_frequency = 50
):
    rationale_path = f'{rationale_directory}/3-high_quality_rationales.json'
    
    with open(rationale_path) as f:
        rationale_data = json.load(f)
        
    template_dict = json.load(open(template_file_path))
    Instruction = template_dict['Instruction_consistency_prediction']
    Template_prediction = template_dict['first_input'] + \
        '\n\n' + \
        template_dict['input_consistency']
        
    new_dict_list = []
    num = 0
    for i in tqdm(range(len(rationale_data)), desc="Rationale unfolding"):
        target_options = ['A', 'B', 'C', 'D', 'E'][:len(rationale_data[i]['answers'])]
        for target_option in target_options:
            new_dict = {
                "id_string": rationale_data[i]['id_string'],
                "passage": rationale_data[i]['passage'],
                "question": rationale_data[i]['question'],
                "answers": rationale_data[i]['answers'],
                "given_answer": rationale_data[i]['given_answer'],
                "label": rationale_data[i]['label'],
                "target_option": target_option,
                "rationale": rationale_data[i]['rationale'],
                'rationale_num': num 
            }
            new_dict_list.append(new_dict)
        num += 1
        
    for i in tqdm(range(len(new_dict_list)), desc="Input string generation"):
        new_dict_list[i]['input_string'] = dp.templating_for_predict_binary_answer(Template_prediction, Instruction,
                                                                            new_dict_list[i]['passage'], new_dict_list[i]['question'],
                                                                            new_dict_list[i]['answers'], new_dict_list[i]['rationale'],
                                                                            new_dict_list[i]['target_option'])
    
    # Generate rationale
    input_prompts = [new_dict_list[i]['input_string'] for i in range(len(new_dict_list))]
    
    output_metadata = {
        "dataset_name": data,
        'Template_prediction': Template_prediction,
        "Instruction": Instruction,     
        "max_gen_len": max_gen_len,
        "batch_size": batch_size,
    }
    json.dump(output_metadata, open(f'{rationale_directory}/4-consistency_prediction-metadata.json', 'w'), indent=4)
    
    results = []
    output_fname= f'{rationale_directory}/4-consistency-predictions.json'
    if os.path.exists(output_fname):
        with open(output_fname) as f:
            results = json.load(f)
            
        start_index = len(results)
        
        print(f"Already processed until {len(results)}th rationale")
    else:
        start_index = 0
    
    generator = Generator_vllm(model=base_model_name, 
                               stop_string="[", 
                               max_gen_length=max_gen_len, 
                               temperature=0, 
                               top_p=1)
    
    for i in tqdm(range(start_index, len(input_prompts), batch_size), desc="Consistency predictions"):
        results += generator.generating(input_prompts[i:i+batch_size])
        new_dict_list_2 = [dict(new_dict_list[i], **results[i]) for i in range(len(results))]
        
        if i % save_frequency == 0:
            json.dump(new_dict_list_2, open(output_fname, 'w'), indent=4)   
    
    json.dump(new_dict_list_2, open(output_fname, 'w'), indent=4)   
    
    consistency_scores = []
    for i in range(num):
        
        current_rows = []
        for j in range(len(new_dict_list_2)):
            if new_dict_list_2[j]['rationale_num'] == i:
                current_rows.append(new_dict_list_2[j])
        
        consistency_score = 0
        for row in current_rows:
            label = row['label']
            target_option = row['target_option']
            
            if (label == target_option): 
                tmp_label = 'correct'
            else:
                tmp_label = 'incorrect'
            
            if 'not' in row['generated_text'] or 'incorrect' in row['generated_text']:
                tmp_prediction = 'incorrect'
            else:
                tmp_prediction = 'correct'
            
            if tmp_label == tmp_prediction:
                consistency_score += 1
                
        consistency_scores += [consistency_score]
    
    fname = '4-consistency-evaluation.json'
    eval_result = {
        'consistency_scores': {
            '0': len([score for score in consistency_scores if score == 0]),
            '1': len([score for score in consistency_scores if score == 1]),
            '2': len([score for score in consistency_scores if score == 2]),
            '3': len([score for score in consistency_scores if score == 3]),
            '4': len([score for score in consistency_scores if score == 4]),
            '5': len([score for score in consistency_scores if score == 5]),
            '6>=': len([score for score in consistency_scores if score > 5])
        }
    }
    
    with open(f'{rationale_directory}/{fname}', 'w') as f:
        json.dump(eval_result, f, indent=4)
    
    
if __name__ == "__main__":
    fire.Fire(main)
