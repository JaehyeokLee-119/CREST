import fire
import json
from tqdm import tqdm 
import utils.data_processing as dp
from utils.generation import Generator_vllm

def main(
    base_model_name: str = None,
    max_gen_len: int = 20,
    batch_size: int = 32,
    data: str = None,
    rationale_directory: str = None,
    template_file_path: str = None, 
): 
    rationale_path = f'{rationale_directory}/1-rationales.json'
    
    ### Load and combine Template ### 
    template_dict = json.load(open(template_file_path))
    Instruction = template_dict['instruction']
    Template_prediction = template_dict['first_input'] + '\n\n' + \
            template_dict['oneshot_input'] + template_dict['prediction_input']
    
    with open(rationale_path) as f:
        rationale_data = json.load(f)
    
    input_data_list_with_rationale = []
    
    for i in range(len(rationale_data)):
        given_answers = rationale_data[i]['rationales_for_given_answer'].keys()
        for given_answer in given_answers:
            for rationale in rationale_data[i]['rationales_for_given_answer'][given_answer]:
                new_dict = {
                    "id_string": rationale_data[i]['id_string'],
                    "passage": rationale_data[i]['passage'],
                    "question": rationale_data[i]['question'],
                    "answers": rationale_data[i]['answers'],
                    "given_answer": given_answer,
                    "label": rationale_data[i]['label'],
                    "rationale": rationale,
                }
                input_data_list_with_rationale.append(new_dict)
    
    for i in range(len(input_data_list_with_rationale)):
        input_data_list_with_rationale[i]['input_string'] = dp.templating_for_predict_answer(Template_prediction, Instruction,
                                                                            input_data_list_with_rationale[i]['passage'], input_data_list_with_rationale[i]['question'],
                                                                            input_data_list_with_rationale[i]['answers'], input_data_list_with_rationale[i]['rationale'],)
    
    # Generate rationale
    input_prompts = [input_data_list_with_rationale[i]['input_string'] for i in range(len(input_data_list_with_rationale))]
    
    generator = Generator_vllm(model=base_model_name, 
                               stop_string="[", 
                               max_gen_length=max_gen_len, 
                               temperature=0,
                               top_p=1.0
                )
    
    print("Generator built!")
    tokenizer = generator.tokenizer
    print("tokenizer built!")
    
    longest_inputs, max_seq_len = dp.longest_inputs_and_max_seq_len(input_prompts, batch_size, tokenizer)
    longest_inputs = sorted(input_prompts, key=lambda x: len(x), reverse=True)[:batch_size]
    _ = generator.generating(longest_inputs)
    
    results = []
    for i in tqdm(range(0, len(input_prompts), batch_size), desc="Generating predictions"):
        results += generator.generating(input_prompts[i:i+batch_size])
    
    input_data_list_with_rationale = [dict(input_data_list_with_rationale[i], **results[i]) for i in range(len(input_data_list_with_rationale))]
           
    # saves
    # templates
    output_metadata = {
        "dataset_name": data,
        'Template_prediction': Template_prediction,
        "Instruction": Instruction,       
        "max_seq_len": max_seq_len,
        "max_gen_len": max_gen_len,
        "batch_size": batch_size,
    }
    json.dump(output_metadata, open(f'{rationale_directory}/2-prediction-metadata.json', 'w'), indent=4)
    json.dump(input_data_list_with_rationale, open(f'{rationale_directory}/2-predictions.json', 'w'), indent=4)   
    
if __name__ == "__main__":
    fire.Fire(main)
