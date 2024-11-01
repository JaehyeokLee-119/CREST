import json 
from tqdm import tqdm 

def generate_option_string(answers:list):
    """
    - answers: List[str]
    -> 'A. answer[0]\nB. answer[1]\nC. answer[2]\nD. answer[3]'
    """
    option_string = ''
    for i in range(len(answers)):
        option_string += f'{chr(65+i)}. {answers[i]}\n'
    return option_string

def construct_fewshot(oneshot_input, oneshot_output, fewshot_path, CoT, fewshot_count=20):
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
    fewshots = fewshots[:fewshot_count]
    return "\n\n".join(fewshots)

def generate_prompt_train(rows, tokenizer, instruction=None, template=None, test=False):
    if test == True:
        rows = rows.take(5)
        
    prompt_list = []
    for i in range(len(rows['id_string'])):
        id_string = rows['id_string'][i]
        if '{passage}' in template:
            context = rows['context'][i] if rows['context'][i] != None else ''
        else:
            context = ''
            
        question = rows['question'][i]
        
        if '{rationale}' in template:
            rationale = rows['rationale'][i] if rows['rationale'][i] != None else ''
        else:
            rationale = ''
        answers = rows['answers'][i]
        label = rows['label'][i] if rows['label'][i] != None else ''
        
        processed = template.format(
                instruction=instruction.strip(),
                passage=context.strip(),
                question=question.strip(),
                option_string=generate_option_string(answers).strip(),
                rationale=rationale.strip(),
                given_answer=label.strip(),
            )
        eos_token_string = tokenizer.eos_token if tokenizer.eos_token else ""
        
        prompt_list.append(
            processed + eos_token_string
        )
    return prompt_list

def generate_prompt_test(rows, tokenizer, instruction=None, template=None):
    prompt_list = []
    for i in tqdm(range(len(rows['id_string'])), desc="Generating test prompt"):
        id_string = rows['id_string'][i]
        if '{passage}' in template:
            context = rows['context'][i] if rows['context'][i] != None else ''
        else:
            context = ''
            
        question = rows['question'][i]
        answers = rows['answers'][i]
        label = rows['label'][i] if rows['label'][i] != None else ''
        
        processed = template.format(
                instruction=instruction.strip(),
                passage=context.strip(),
                question=question.strip(),
                option_string=generate_option_string(answers).strip(),
            )
        
        prompt_list.append(
            processed
        )
    return prompt_list

def is_reprediction_target(generated: str):
    generated = generated.strip()
    if '[Answer]' in generated:
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
