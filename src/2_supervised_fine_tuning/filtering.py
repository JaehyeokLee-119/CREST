import json
import os
import pandas as pd
import fire 
from tqdm import tqdm 

# Split the rationales based on tilde(z) (>= consistency_score)
def main(
    stage1_output_dir = '',
):
    # Load stage1 output
    filename = '4-hq_rationale_with_consistency_score.json'
    
    df = pd.read_json(os.path.join(stage1_output_dir, filename))
    
    df['consistency_score'] = df['consistency_score'].fillna(-1)
    consistency_score_list = df.consistency_score.unique()
    
    for consistency_score in tqdm(consistency_score_list, desc=f'Consistency score filtering'):
        consistency_score = int(consistency_score)
        if consistency_score == -1:
            continue
        
        df_filtered = df[df.consistency_score >= consistency_score]
        print(f"Consistency score: {consistency_score}, Number of data: {len(df_filtered)}/{len(df)}")
        output_fname = os.path.join(stage1_output_dir, f'4-hq_rationale_with_consistency_score_{consistency_score}.json')
        df_filtered.to_json(output_fname, orient='records', lines=True)
        
    print(len(df))

if __name__ == '__main__':
    fire.Fire(main)