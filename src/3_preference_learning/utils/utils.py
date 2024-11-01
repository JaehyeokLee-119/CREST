import json 
import pandas as pd
import os 

def generate_option_string(answers:list):
    option_string = ''
    for i in range(len(answers)):
        option_string += f'{chr(65+i)}. {answers[i]}\n'
    return option_string


def combine_query(examples, template=None, instruction=None, tokenizer=None):
    passage = examples['passage']
    question = examples['question']
    answers = examples['answers']
    option_string = generate_option_string(answers)
    
    if instruction == None:
        instruction = "Read the question below carefully, write the intermediate reasoning steps about the problem and then choose an answer. You must analyze the question and the choices, and provide a logically organized reasoning steps that can be an explanation of why your answer is the most appropriate.\n"
    if template == None:
        template = "[Instruction and Question]\n{instruction}\n<Passage> {passage}\n<Question> {question}\n{option_string}\n\n[Rationale]\nLet's think step by step.\n\n"
    
    query = template.format(instruction=instruction, passage=passage, question=question, option_string=option_string)
        
    if tokenizer != None:
        start_token_string = tokenizer.bos_token if tokenizer.bos_token else ""
        eos_token_string = tokenizer.eos_token if tokenizer.eos_token else ""
    
        query = start_token_string + query

    
    return query

# def extract_label(output: str):
#     labels = ['A', 'B', 'C', 'D', 'E']
    
#     output = output.replace('<', '[')
#     output = output.replace('>', ']')
    
#     keyword_list = ['[Answer]', 'correct answer']
#     for keyword in keyword_list:
#         answer_index = output.find(keyword)
#         if answer_index != -1: 
#             output_part = output[answer_index+len(keyword):].strip()
            
#             if '\n\n' in output_part:
#                 output_part = output_part.split('\n\n')[0]
            
#             for char in output_part:
#                 if char in labels:
#                     return char
#             return 'X'
        
#     return 'X'


# def extract_label(output: str): # FOR LABEL MODELS
#     labels = ['A', 'B', 'C', 'D', 'E']
    
#     output = output.replace('<', '[')
#     output = output.replace('>', ']')
    
#     output_part = output.strip()
    
#     if '\n\n' in output_part:
#         output_part = output_part.split('\n\n')[0]
    
#     for char in output_part:
#         if char in labels:
#             return char
#     return 'X'

def extract_label(output: str):
    labels = ['A', 'B', 'C', 'D', 'E']
    question_index = output.find('[Question]')
    if question_index != -1:
        output = output[:question_index]
        
    answer_index = output.find('[Answer]')
    if answer_index == -1: 
        answer_index2 = output.find('correct answer is')
        output_answer = output[answer_index+len('correct answer is'):]
        
        output_first_char = output_answer.strip()[0] if len(output_answer.strip()) > 0 else 'X'
        
        if output_first_char in labels:
            return output_answer.strip()[0]
        return 'X'
        
    else: 
        output_answer = output[answer_index+len('[Answer]'):]
        for label in labels:
            if label in output_answer:
                return label

def extract_label_zeroshot(output: str):
    labels = ['A', 'B', 'C', 'D', 'E']
    output = output.strip()
    for label in labels:
        if label in output:
            return label
        
def result_evaluation(result_path: str):
    result_df = pd.read_json(result_path)

    result_df['prediction'] = result_df.output.apply(extract_label)
    result_df['correct'] = result_df.prediction == result_df.label
    result_df_extracted = result_df[result_df.prediction != 'X']

    total_correct = result_df.correct.sum()/len(result_df) if len(result_df) > 0 else 0
    print(f'Total Accuracy: {total_correct:.4f}')
    extracted_ratio = len(result_df_extracted)/len(result_df) if len(result_df) > 0 else 0
    print(f'Extracted Ratio: {extracted_ratio:.4f}')
    extracted_correct = result_df_extracted.correct.sum()/len(result_df_extracted) if len(result_df_extracted) > 0 else 0
    print(f'Extracted Accuracy: {extracted_correct:.4f}')
    return (total_correct, extracted_ratio, extracted_correct)