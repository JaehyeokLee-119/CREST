import fire
import utils.data_processing as dp
import json 
from tqdm import tqdm 

def main(
    prediction_directory: str = None
):
    prediction_path = f'{prediction_directory}/1-rationales.json'
    prediction_data = json.load(open(prediction_path))
    
    result = []
    for data in tqdm(prediction_data):
        for given_answer in data["rationales_for_given_answer"]:
            data["rationales_for_given_answer"][given_answer] = [dp.rationale_refining(rationale) for rationale in data["rationales_for_given_answer"][given_answer]]
        result.append(data)
    
    fname = '1-rationales_old.json'
    
    original_data = json.load(open(prediction_path))
    with open(prediction_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    with open(f'{prediction_directory}/{fname}', 'w') as f:
        json.dump(original_data, f, indent=4)
    
if __name__ == '__main__':
    fire.Fire(main)