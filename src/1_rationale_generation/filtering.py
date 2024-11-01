import fire
import utils.data_processing as dp
import json 
from tqdm import tqdm 

def main(
    prediction_directory: str = None,
    test = False,
):
    if type(test) == str:
        test = True if test.lower() == 'true' else False
    prediction_path = f'{prediction_directory}/2-predictions.json'
    prediction_data = json.load(open(prediction_path))
    
    for i in tqdm(range(len(prediction_data))):
        id_string = prediction_data[i]['id_string']
        generated = prediction_data[i]['generated_text']
        answers = prediction_data[i]['answers']
        choices = ['A', 'B', 'C', 'D', 'E'][:len(answers)]
        prediction = dp.prediction_extraction(generated, answers=choices)
        prediction_data[i]['prediction'] = prediction   
        
    if test == True:
        high_quality_rationales = []
        for data in prediction_data:
            high_quality_rationales.append(data)
        low_quality_rationales = []
    else:
        low_quality_rationales = []
        for data in prediction_data:
            if data['prediction'] != data['label']: 
                low_quality_rationales.append(data)
                
        high_quality_rationales = []
        for data in prediction_data:
            if data['prediction'] == data['label']:
                high_quality_rationales.append(data)
        
        
    high_quality_rationales_path = f'{prediction_directory}/3-high_quality_rationales.json'
    low_quality_rationales_path = f'{prediction_directory}/3-low_quality_rationales.json'
    
    with open(high_quality_rationales_path, 'w') as f:
        json.dump(high_quality_rationales, f, indent=4)
    
    with open(low_quality_rationales_path, 'w') as f:
        json.dump(low_quality_rationales, f, indent=4)
    
if __name__ == '__main__':
    fire.Fire(main)