import fire
from typing import List
import json
from tqdm import tqdm 

def main(
    rationale_directory: str = None,
):
    rationale_path = f'{rationale_directory}/4-consistency-predictions.json'
    high_quality_rationales_path = f'{rationale_directory}/3-high_quality_rationales.json'
        
    
    with open(rationale_path) as f:
        cs_rated_rationale_data = json.load(f)
    
    with open(high_quality_rationales_path) as f:
        high_quality_rationales = json.load(f)
    
    id_question_count = max([cs_rated_rationale_data[i]['rationale_num'] for i in range(len(cs_rated_rationale_data))]) + 1
    consistency_scores = []
    consistency_dict = []
    for i in tqdm(range(id_question_count)):
        current_rows = []
        for j in range(len(cs_rated_rationale_data)):
            if cs_rated_rationale_data[j]['rationale_num'] == i:
                current_rows.append(cs_rated_rationale_data[j])
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
        consistency_dict += [{
            'consistency_score': consistency_score,
            'rationale_num': i,
            'rationale': current_rows[0]['rationale']
        }]
    
    consistency_scores_distribution_fname = '4-consistency-evaluation.json'
    eval_result = {
        'consistency_scores_distribution': {
            '0': len([score for score in consistency_scores if score == 0]),
            '1': len([score for score in consistency_scores if score == 1]),
            '2': len([score for score in consistency_scores if score == 2]),
            '3': len([score for score in consistency_scores if score == 3]),
            '4': len([score for score in consistency_scores if score == 4]),
            '5': len([score for score in consistency_scores if score == 5]),
            '6>=': len([score for score in consistency_scores if score > 5])
        }
    }
    
    for i in range(len(consistency_scores)):
        high_quality_rationales[i]['consistency_score'] = consistency_scores[i]
        
    
    with open(f'{rationale_directory}/{consistency_scores_distribution_fname}', 'w') as f:
        json.dump(eval_result, f, indent=4)
    
    with open(f'{rationale_directory}/4-hq_rationale_with_consistency_score.json', 'w') as f:
        json.dump(high_quality_rationales, f, indent=4)
    
if __name__ == "__main__":
    fire.Fire(main)
