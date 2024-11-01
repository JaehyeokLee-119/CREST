import json 
import pandas as pd 
from tqdm import tqdm 
import fire
import os 
import random
    
def dataset_collate_fn(examples, instruction, template_input):
    def generate_option_string(answers:list):
        option_string = ''
        for i in range(len(answers)):
            option_string += f'{chr(65+i)}. {answers[i]}\n'
        return option_string
    
    question = examples['question']
    answers = examples['answers']
    option_string = generate_option_string(answers)
    
    if 'passage' in examples:
        passage = examples['passage']
    else: 
        passage = ''
    collated_query = template_input.format(instruction=instruction, passage=passage, question=question, option_string=option_string)
    return collated_query

def make_output_j(example, template_output):
    predicted_label = example['prediction_j']
    predicted_label = ''.join(filter(str.isalpha, predicted_label))
    
    output = template_output.format(rationale=example['rationale_j'], given_answer=predicted_label)
    return output

def make_output_k(example, template_output):
    predicted_label = example['prediction_k']
    predicted_label = ''.join(filter(str.isalpha, predicted_label))
    
    output = template_output.format(rationale=example['rationale_k'], given_answer=predicted_label)
    return output


def sort_key(x):
    key = ''.join([i for i in x if i.isdigit()])
    return int(key)

def binary_set_pairing(hq_data_df, lq_data_df):
    hq_id_strings = list(set(hq_data_df['id_string'].tolist()))
    lq_id_strings = list(set(lq_data_df['id_string'].tolist()))
    available_id_strings = list(set(hq_id_strings) & set(lq_id_strings)) 
    available_id_strings.sort(key=sort_key)
    paired_data = []

    pid = 0
    for id_ in tqdm(available_id_strings, desc='Z pairing'):
        j_data = hq_data_df[hq_data_df['id_string'] == id_]
        k_data = lq_data_df[lq_data_df['id_string'] == id_]
        
        for j in range(len(j_data)):
            for k in range(len(k_data)):
                paired_data.append({
                    'pair_id': pid,
                    'id_string': id_,
                    'passage': j_data.iloc[j]['passage'],
                    'question': j_data.iloc[j]['question'],
                    'answers': j_data.iloc[j]['answers'],
                    'label': j_data.iloc[j]['label'],
                    'rationale_j': j_data.iloc[j]['rationale'],
                    'rationale_k': k_data.iloc[k]['rationale'],
                    'prediction_j': j_data.iloc[j]['prediction'],
                    'prediction_k': k_data.iloc[k]['prediction'],
                })
                pid += 1

    random.shuffle(available_id_strings)
    
    train_id_strings = available_id_strings
    dev_id_strings = []
    test_id_strings = []
    
    train_data = [i for i in paired_data if i['id_string'] in train_id_strings]
    dev_data = [i for i in paired_data if i['id_string'] in dev_id_strings]
    test_data = [i for i in paired_data if i['id_string'] in test_id_strings]
    
    return train_data, dev_data, test_data

def generate(
    stage_1_result = '', # stage_1 result directory path
    output_dir = '',
    template_file_path = '',
):
    # os.makedirs(stage_3_processed_dir, exist_ok=True)
    fname_cs = '4-hq_rationale_with_consistency_score.json'
    fname_hq = '3-high_quality_rationales.json'
    fname_lq = '3-low_quality_rationales.json'
    
    hq_data = json.load(open(os.path.join(stage_1_result, fname_hq)))
    hq_data_df = pd.DataFrame(hq_data)
    hq_data_df['prediction'] = hq_data_df['prediction'].apply(lambda x: x.upper())

    lq_data = json.load(open(os.path.join(stage_1_result, fname_lq)))
    lq_data_df = pd.DataFrame(lq_data)
    lq_data_df['prediction'] = lq_data_df['prediction'].apply(lambda x: x.upper())

    cs_data = json.load(open(os.path.join(stage_1_result, fname_cs)))
    cs_data_df = pd.DataFrame(cs_data)
    
    train_data, dev_data, test_data = binary_set_pairing(hq_data_df, lq_data_df)
    z_pair_df = pd.DataFrame(train_data)
    
    # json.dump(train_data, open(os.path.join(stage_3_processed_dir, 'train_data.json'), 'w'), indent=2)
    
    cs_data_df_cs_list = []
    for i in range(0,6):
        cs_data_df_cs_list.append(cs_data_df[cs_data_df['consistency_score'] == i])

    cs_paired_data = {}
    
    pid = 0
    for j in range(5, -1, -1):
        pair_n_to_under = []
        j_data = cs_data_df_cs_list[j] 
        j_id_strings = list(set(j_data['id_string']))
        for k in range(j-1, -1, -1):
            pair_n_to_under.append(len(cs_data_df_cs_list[k]))
            k_data = cs_data_df_cs_list[k] 
            
            for j_id_string in tqdm(j_id_strings, desc=f'j={j}, k={k}'):
                j_data_ = j_data[j_data['id_string'] == j_id_string]
                k_data_ = k_data[k_data['id_string'] == j_id_string]
                for j_i in range(len(j_data_)):
                    for k_i in range(len(k_data_)):
                        if str(j) not in cs_paired_data.keys():
                            cs_paired_data[str(j)] = []
                        cs_paired_data[str(j)].append({
                            'pair_id': int(pid),
                            'id_string': j_id_string,
                            'passage': j_data_.iloc[j_i]['passage'],
                            'question': j_data_.iloc[j_i]['question'],
                            'answers': j_data_.iloc[j_i]['answers'],
                            'label': j_data_.iloc[j_i]['label'],
                            'rationale_j': j_data_.iloc[j_i]['rationale'],
                            'rationale_k': k_data_.iloc[k_i]['rationale'],
                            'prediction_j': j_data_.iloc[j_i]['prediction'],
                            'prediction_k': k_data_.iloc[k_i]['prediction'],
                            'consistency_score_j': int(j_data_.iloc[j_i]['consistency_score']),
                            'consistency_score_k': int(k_data_.iloc[k_i]['consistency_score']),
                        })
                        pid += 1
    cs_paired_data_total = []
    for i in cs_paired_data.keys():
        cs_paired_data_total += cs_paired_data[i]
        
    cs_id_string_list = list(set([i['id_string'] for i in cs_paired_data_total]))
    cs_id_string_list.sort(key=sort_key)
    random.shuffle(cs_id_string_list)

    cs_train_id_strings = cs_id_string_list
    cs_train_data = [i for i in cs_paired_data_total if i['id_string'] in cs_train_id_strings]

    tz_pair_df = pd.DataFrame(cs_train_data)
    
    dfs = [tz_pair_df, z_pair_df]
    output_folders = ['only_tz', 'only_z']
    
    for train_df, output_folder in zip(dfs, output_folders):
        output_dir_path = os.path.join(output_dir, output_folder)
        
        template_dict = json.load(open(template_file_path, "r"))
        instruction = template_dict['instruction']
        Template_input = template_dict['first_input'] + '\n\n' + template_dict['oneshot_input']
        Template_output = template_dict['oneshot_output']
        train_queries = []
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f'{output_folders}'):
            query = dataset_collate_fn(row, Template_input, instruction)
            train_queries.append(query)
            
        train_df['query'] = train_queries
        train_df['response_j'] = train_df.apply(lambda x: make_output_j(x, Template_output), axis=1)
        train_df['response_k'] = train_df.apply(lambda x: make_output_k(x, Template_output), axis=1)
        
        train_df.drop(columns=['passage', 'question', 'answers', 'rationale_j', 'rationale_k', 'prediction_j', 'prediction_k'], inplace=True)
        
        if 'consistency_score_j' and 'consistency_score_k' in train_df.columns:
            train_df.drop(columns=['consistency_score_j', 'consistency_score_k'], inplace=True)
            
        os.makedirs(output_dir_path, exist_ok=True)
        
        train_df['pair_id'] = [i for i in range(len(train_df))]
        train_df.to_json(f'{output_dir_path}/qjk_train_data.json', orient='records', lines=True)
        
        
if __name__ == '__main__':
    random.seed(42)
    fire.Fire(generate)