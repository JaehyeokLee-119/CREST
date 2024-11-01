from tqdm import tqdm 

def eliminate_answer_in_rationale(rationale: str):
    rationale_list = rationale.strip().split('\n')
    last_line = rationale_list[-1]
    last_line_clean = last_line.replace('(', '').replace(')', '')
    if 'answer' in last_line_clean and any([letter in last_line_clean for letter in ['A.', 'B.', 'C.', 'D.', 'E.']]):
        refined_rationale = '\n'.join(rationale_list[:-1])
        return refined_rationale
    else:
        return None

def generate_option_string(answers:list):
    option_string = ''
    for i in range(len(answers)):
        option_string += f'{chr(65+i)}. {answers[i]}\n'
    return option_string


def rationale_refining(rationale:str):
    cut_point = rationale.lower().find('[question]')
    if cut_point != -1:
        refined_rationale = rationale[:cut_point]
    else:
        cut_point2 = rationale.lower().find('\n\n[')
        if cut_point2 != -1:
            refined_rationale = rationale[:cut_point2]
        else:
            refined_rationale = rationale
        
    return refined_rationale.strip()


def longest_inputs_and_max_seq_len(input_prompts, max_batch_size, tokenizer=None):
    if tokenizer == None:
        tokenized_inputs = input_prompts
    else:
        tokenized_inputs = [tokenizer.encode(input_prompt) for input_prompt in tqdm(input_prompts, desc="Tokenizing")]
        
    longest_input_index = 0
    max_sum_seq_len = 0
    
    for i in range(0, len(tokenized_inputs), max_batch_size):
        current_inputs = tokenized_inputs[i:i+max_batch_size]
        current_sum_seq_len = sum([len(current_input) for current_input in current_inputs])
        
        if current_sum_seq_len > max_sum_seq_len:
            max_sum_seq_len = current_sum_seq_len
            longest_input_index = i
    
    longest_inputs = input_prompts[longest_input_index:longest_input_index+max_batch_size]
    max_seq_len = max([len(longest_input) for longest_input in longest_inputs]) if longest_inputs else 0
    return longest_inputs, max_seq_len
    
    
def diverse_question(template, instruction, row, correct_n:int, incorrect_n:int, fewshots:str=None):    
    id_string = row['id_string']
    passage = row['context']
    question = row['question']
    answers = row['answers']
    label = row['label']
    
    result_list = []
    answer_list = ['A', 'B', 'C', 'D', 'E'][:len(answers)]
    
    if label == None:
        label = 'A'
    fewshots = '' if fewshots is None else fewshots
    for i in range(len(answer_list)):
        repeat = correct_n if answer_list[i] == label else incorrect_n
        for _ in range(repeat):
            given_answer = f'{answers[i]}'
            result_list.append({
                'id_string': id_string,
                'given_answer': chr(65+i),
                'label': label,
                'input_string': templating_for_generate_rationale(template, instruction, passage, question, answers, given_answer, fewshots)
            })
                
    return result_list

def self_consistency(template, instruction, row, samples:int, fewshots:str=None):    
    id_string = row['id_string']
    passage = row['context']
    question = row['question']
    answers = row['answers']
    label = row['label']
    answer_list = ['A', 'B', 'C', 'D', 'E'][:len(answers)]
    
    result_list = []
    
    fewshots = '' if fewshots is None else fewshots
    
    for _ in range(samples):
        given_answer = f'{label}'
        result_list.append({
            'id_string': id_string,
            'given_answer': label,
            'label': label,
            'input_string': templating_for_generate_rationale(template, instruction, passage, question, answers, given_answer, fewshots)
        })
                
    return result_list


def fewshot_generation(template=None, data=None, fewshot_count=5, answer_format='sentence'):
    result = ''
    i = 0
    for row in data:
        if i == fewshot_count:
            break
        passage = row['context'] or 'None'
        question = row['question']
        answers = row['answers']
        if answer_format == 'sentence':
            given_answer = answers[ord(row['given_answer']) - 65]
        elif answer_format == 'letter':
            given_answer = row['given_answer']
        elif answer_format == 'both':
            given_answer = f'{row["given_answer"]}. {answers[ord(row["given_answer"]) - 65]}'
            
        rationale = row['rationale']
        option_string = generate_option_string(answers)
        result += template.format(passage=passage.strip(), question=question.strip(), option_string=option_string.strip(), given_answer=given_answer.strip(), rationale=rationale.strip())
    return result 

def templating_for_generate_rationale(template:str, instruction:str, passage:str, question:str, answers:list, given_answer:str, fewshots:str):
    '''templating for generating rationale with given answer'''
    option_string = generate_option_string(answers)
    passage = '' if passage is None else passage
    result = template.format(instruction=instruction.strip(), passage=passage.strip(), question=question.strip(), 
                             option_string=option_string.strip(), given_answer=given_answer.strip(), fewshots=fewshots.strip())
    return result

def templating_for_predict_answer(template:str, instruction:str, passage:str, question:str, answers:list, rationale:str):
    '''templating for predicting answer with given rationale'''
    option_string = generate_option_string(answers)
    if passage is None:
        passage = ''
    result = template.format(instruction=instruction.strip(), passage=passage.strip(), question=question.strip(), option_string=option_string.strip(), rationale=rationale.strip())
    return result

def templating_for_predict_answer_without_rationale(template:str, instruction:str, passage:str, question:str, answers:list):
    '''templating for predicting answer without rationale'''
    option_string = generate_option_string(answers)
    passage = '' if passage is None else passage
    result = template.format(instruction=instruction.strip(), passage=passage.strip(), question=question.strip(), option_string=option_string.strip())
    return result

def templating_for_predict_binary_answer(template:str, instruction:str, passage:str, question:str, answers:list, rationale:str, target_option:str):
    '''templating yes or no question'''
    option_string = generate_option_string(answers)
    target_option_str = answers[ord(target_option) - 65]
    if passage is None:
        passage = ''
        
    result = template.format(instruction=instruction.strip(), passage=passage.strip(), question=question.strip(), 
                             option_string=option_string.strip(), rationale=rationale.strip(), target_option=target_option_str.strip())
    return result

def prediction_extraction(generated:str, answers=None):
    '''extract the answer from the generated string'''
    if answers==None:
        answers = ['A', 'B', 'C', 'D']
        
    answer_position_leftmost = {answer: generated.find(answer) for answer in answers}
    for key in answer_position_leftmost.keys():
        if answer_position_leftmost[key] == -1:
            answer_position_leftmost[key] = 99999
    
    if all([answer_position_leftmost[key] == 99999 for key in answer_position_leftmost.keys()]):
        answers = ['a', 'b', 'c', 'd', 'e'][:len(answers)]
        answer_position_leftmost = {answer: generated.find(answer) for answer in answers}
        for key in answer_position_leftmost.keys():
            if answer_position_leftmost[key] == -1:
                answer_position_leftmost[key] = 99999 
                
    answer = min(answer_position_leftmost, key=answer_position_leftmost.get)
    return answer
