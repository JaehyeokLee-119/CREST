import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class Generator:
    def __init__(self, model, 
                 tokenizer, 
                 max_gen_length, 
                 temperature, 
                 top_p, 
                 stop_string="[", 
                 padding_side='left'):
        
        self.model = model
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = padding_side
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.tokenizer = tokenizer
        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p
        self.stop_string = stop_string
        self.model.generation_config.pad_token_id = tokenizer.pad_token_id
        
    def generating(self, input_prompts):
        input_ids = self.tokenizer(input_prompts, return_tensors="pt", padding=True).to("cuda")
        if self.temperature > 0:
            output_ids = self.model.generate(
                **input_ids, 
                do_sample=True,
                max_new_tokens=self.max_gen_length,  
                top_p=self.top_p,
                temperature=self.temperature, 
                tokenizer=self.tokenizer,
                # stop_strings=self.stop_string,
            )
        else:
            output_ids = self.model.generate(
                **input_ids, 
                do_sample=False,
                max_new_tokens=self.max_gen_length,  
                tokenizer=self.tokenizer,
                stop_strings=self.stop_string,
            )
            
        result = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        result_only_output = [r[len(i):] for i, r in zip(input_prompts, result)]
        result_only_output = [r[:r.find(self.stop_string)].strip() for r in result_only_output]
        
        return [{'generated_text': i} for i in result_only_output]

class Generator_vllm:
    def __init__(self, model, 
                 max_gen_length, 
                 temperature, 
                 top_p, 
                 stop_string=None, 
                 tensor_parallel_size=1):
        
        if stop_string is None:
            stop = None
        else:
            stop = [stop_string]
            
        self.sampling_params = SamplingParams(
            top_p=top_p,
            temperature=temperature, 
            max_tokens=max_gen_length, 
            stop=stop
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(model, tensor_parallel_size=tensor_parallel_size, enforce_eager=True, swap_space=64)
        
    def generating(self, input_prompts):
        completions = self.llm.generate(input_prompts, self.sampling_params)
        
        completion_outputs = [c.outputs for c in completions]
        completion_texts = [c[0].text for c in completion_outputs]
        
        return [{'generated_text': i} for i in completion_texts]