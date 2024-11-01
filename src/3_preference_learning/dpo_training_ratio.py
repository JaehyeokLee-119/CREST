import os

from typing import Dict, Optional
from torch.utils.data import DataLoader

from utils.utils import combine_query, result_evaluation

import json
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset, concatenate_datasets
from peft import LoraConfig, PeftModel, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    set_seed,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from tqdm import tqdm
from trl import DPOConfig, DPOTrainer
import wandb
import fire 

def get_paired_dataset(
    data_dir: str = None,
    sanity_check: bool = False,
    max_datas: Optional[int] = None,
    shuffle: Optional[bool] = False,
    num_proc=24,
    seed=42,
) -> Dataset:
    
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "id_string": samples["id_string"],
            "prompt": samples["query"],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    dataset = load_dataset(
        "json",
        data_files=data_dir,
        split="train",
    )
    original_columns = dataset.column_names
    
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))
    else:
        if max_datas:
            dataset = dataset.select(range(min(len(dataset), max_datas)))
            
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def main(
    # Model and Data
    base_model_name: str = None,
    sft_peft_path: str = None,
    working_dir: str = None,
    train_data_oq: str = None,
    train_data_cs: str = None,
    ratio_lambda: float = 0.6, # z:tilde(z) = (1-lambda):lambda
    test_data: str = None,
    wandb_project_name: str = None,
    use_nll_loss: bool = False,
    sanity_check: bool = True,
    # From No SFT
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_target_modules: str = "gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj",
    lora_dropout: float = 0.05,
    # Hyper parameters
    max_steps: int = 3000, 
    dpo_epochs: int = 1,
    learning_rate: float = 8e-6,
    dpo_beta: float = 0.1,
    dpo_nll_loss_alpha: float = 1.0,
    lr_scheduler_type = "cosine", 
    warmup_steps = 100, 
    weight_decay = 0.05, 
    optimizer_type = "paged_adamw_32bit", 
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    simulated_batch_size = 8,
    seed = 42,
    max_length: int = 1024, 
    max_prompt_length: int = 512, 
    # training parameters
    restart_epoch: int = 0, 
    gradient_checkpointing = True,
    gradient_checkpointing_use_reentrant = False,
    save_steps: int = 3000,
    logging_steps: int = 1,
    do_test: bool = False,
    report_to: str = 'wandb',
    test_batch_size = 8,
    test_template_path = None,
    test_max_new_tokens = 560
):
    gradient_accumulation_steps = int(simulated_batch_size / per_device_train_batch_size)
    
    template_dict = json.load(open(test_template_path))
    test_instruction = template_dict['instruction']
    test_template = template_dict['first_input'] + '\n\n' + template_dict['oneshot_input'] + template_dict['final_output']
    
    # Saving Training arguments at Working directory
    os.makedirs(working_dir, exist_ok=True)
    script_args = locals().copy()
    with open(os.path.join(working_dir, "setting.json"), "w") as f:
        json.dump(script_args, f, indent=2)
    
    ratio_lambda = float(ratio_lambda)
    learning_rate = float(learning_rate)
    use_nll_loss = bool(use_nll_loss)
    
    wandb_project_name = wandb_project_name.replace("/", "").replace("\\", "").replace("#", "").replace("?", "").replace("%", "").replace(":", "").replace(":", "")
    
    print(f"[Use NLL Loss] {use_nll_loss}, type:{type(use_nll_loss)}")
    set_seed(seed)
    
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=False,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if max_steps > 0:
        max_datas = max_steps * simulated_batch_size
    else:
        max_datas = None
    print(f"number of data item: {max_datas}")
    
    # WANDB_PROJECT
    os.environ["WANDB_PROJECT"] = wandb_project_name
    
    
    ### START TRAIN ###
    # train epoch (updating reference model)
    for epoch in range(restart_epoch+1, dpo_epochs+1):
        # Load SFT models   
        ''' 0) Dataset construction '''
        max_datas_oq = int(max_datas * (1-ratio_lambda))
        max_datas_cs = int(max_datas * ratio_lambda)
        
        print(f"ratio_lambda (cs_data ratio): {ratio_lambda}, max_datas_oq: {max_datas_oq}, max_datas_cs: {max_datas_cs}")
        train_dataset_oq = get_paired_dataset(data_dir=train_data_oq, shuffle=True, 
                                           max_datas=max_datas_oq, sanity_check=sanity_check,
                                           seed=seed)
        train_dataset_cs = get_paired_dataset(data_dir=train_data_cs, shuffle=True, 
                                           max_datas=max_datas_cs, sanity_check=sanity_check,
                                           seed=seed)
        if max_datas_oq == 0:
            train_dataset = train_dataset_cs
            print(f"only oq: total train data: {len(train_dataset)}")
        elif max_datas_cs == 0:
            train_dataset = train_dataset_oq
            print(f"only cs: total train data: {len(train_dataset)}")
        else:
            train_dataset = concatenate_datasets([train_dataset_oq, train_dataset_cs])
            print(f"""
    train_oq: {len(train_dataset_oq)}, train_cs: {len(train_dataset_cs)} -> total: {len(train_dataset)}
            """)
        
        wandb_run_name = f'{wandb_project_name} {epoch} in {dpo_epochs} epochs'
        wandb_run_name = wandb_run_name.replace(" ", "_")
        output_dir = os.path.join(working_dir, f"epoch_{epoch}")
        
        # Load model (New Or From previous epoch model)
        if epoch == 1:
            peft_path_from_previous_epoch = sft_peft_path
        else:
            # Load previous epoch model
            peft_path_from_previous_epoch = os.path.join(working_dir, f"epoch_{epoch-1}","dpo_train")
        print(f"Load model from {peft_path_from_previous_epoch}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            low_cpu_mem_usage=True,
            device_map={"": Accelerator().local_process_index},
            quantization_config=bnb_config,
        )
        base_model.config.use_cache = False
        
        if peft_path_from_previous_epoch == None:
            print("---<< Start with base model (not SFT) >>---")
            loraconfig = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules.split(","),
                lora_dropout=lora_dropout,
            )
            model = PeftModel(
                base_model,
                adapter_name="dpo_train",
                peft_config=loraconfig,
            )
            ref_adapter_name = None
        else:
            model = PeftModel.from_pretrained(
                base_model,
                peft_path_from_previous_epoch,
                low_cpu_mem_usage=True,
                device_map={"": Accelerator().local_process_index},
                quantization_config=bnb_config,
                is_trainable=True,
                adapter_name="dpo_train",
            )
            model.load_adapter(peft_path_from_previous_epoch, "reference")
            ref_adapter_name = "reference"
            
        model.train()
        
        if max_steps > 0:
            # initialize training arguments:
            training_args = DPOConfig(
                model_adapter_name="dpo_train",
                ref_adapter_name=ref_adapter_name,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                max_steps=max_steps,
                output_dir=output_dir,
                logging_steps=logging_steps,
                save_steps=save_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                learning_rate=learning_rate,
                report_to=report_to,
                lr_scheduler_type=lr_scheduler_type,
                warmup_steps=warmup_steps,
                optim=optimizer_type,
                bf16=True,
                remove_unused_columns=False,
                run_name=wandb_run_name,
                gradient_checkpointing_kwargs=dict(use_reentrant=gradient_checkpointing_use_reentrant),
                seed=seed,
                max_prompt_length=max_prompt_length,
                max_length=max_length,
            )
        else:
            training_args = DPOConfig(
                model_adapter_name="dpo_train",
                ref_adapter_name=ref_adapter_name,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                output_dir=output_dir,
                logging_steps=logging_steps,
                num_train_epochs=1,
                save_steps=save_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                learning_rate=learning_rate,
                report_to=report_to,
                lr_scheduler_type=lr_scheduler_type,
                warmup_steps=warmup_steps,
                optim=optimizer_type,
                bf16=True,
                remove_unused_columns=False,
                run_name=wandb_run_name,
                gradient_checkpointing_kwargs=dict(use_reentrant=gradient_checkpointing_use_reentrant),
                seed=seed,
                max_prompt_length=max_prompt_length,
                max_length=max_length,
            )
        
        ''' 1) DPO Training '''
        dpo_trainer = DPOTrainer(
            model,
            args=training_args,
            beta=dpo_beta,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
            
        dpo_trainer.train()
        dpo_trainer.save_model(output_dir)
        
        dpo_trainer.model = None
        base_model = None
        model = None
        del base_model
        del model
        del dpo_trainer
        torch.cuda.empty_cache()
        
        if do_test == False:
            model = None
            del model
            torch.cuda.empty_cache()
            
            wandb.finish()
            continue
        #### Evaluate on Test Dataset
        def collate_fn_nolabel(inputs):
            # inputs is a list of input (str)
            inputs = [x.strip() for x in inputs]
            input_dict = tokenizer(inputs, padding=True, return_tensors="pt")
            return input_dict
        
        def collate_fn_nolabel_for_repredict(inputs):
            # inputs is a list of input (str)
            inputs = [x['input'].strip() for x in inputs]
            input_dict = tokenizer(inputs, padding=True, return_tensors="pt")
            return input_dict
    
        def inference(dataloader, model, tokenizer, max_new_tokens=560):
            outputs = []
            for batch in tqdm(dataloader, desc="Generating"):
                with torch.no_grad():
                    if hasattr(model, 'current_device'):
                        input_ids = batch['input_ids'].to(model.current_device) # current_device
                        attention_mask = batch['attention_mask'].to(model.current_device) # current_device
                    else:
                        input_ids = batch['input_ids'].to(model.device) # current_device
                        attention_mask = batch['attention_mask'].to(model.device) # current_device
                
                    output = model.generate(input_ids=input_ids, 
                                        attention_mask=attention_mask, 
                                        max_new_tokens=max_new_tokens,
                                        pad_token_id=tokenizer.eos_token_id,
                                        early_stopping=True,
                                        do_sample=False,
                                        )
                    input_decoded  = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
                    
                    output_decoded = [x.replace(input_decoded[ix], '').strip() for ix, x in enumerate(output_decoded)]
                    outputs.extend(output_decoded)
            return outputs
        
        device = 'auto'
        model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map=device)
        model = PeftModelForCausalLM.from_pretrained(model, os.path.join(working_dir, f"epoch_{epoch}","dpo_train"), device_map=device)
        model.config.pad_token_id = model.config.eos_token_id
        model.eval()
        
        ''' 2) Prediction '''
        ## 1. First Prediction
        inference_output_path = os.path.join(output_dir, "inference_result_data.jsonl")
        repredicted_output_path = os.path.join(output_dir, f'inference_result_data-repredicted.jsonl')
        test_dataset = load_dataset("json", data_files=test_data, split="train")
        if '.jsonl' in test_data:
            test_dataset = test_dataset.rename_column('context', 'passage')
        if sanity_check:
            test_dataset = test_dataset.select(range(min(len(test_dataset), 20)))
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        input_strings = [combine_query(ex, template=test_template, instruction=test_instruction, 
                                       tokenizer=tokenizer) for ex in test_dataset]
        dataloader = DataLoader(input_strings, batch_size=test_batch_size, collate_fn=collate_fn_nolabel)
        
        result_dict_list = []
        outputs = inference(dataloader, model, tokenizer, max_new_tokens=test_max_new_tokens)
        for i in range(len(outputs)):
            current_result = {
                "id_string": test_dataset['id_string'][i],
                "input": input_strings[i],
                "label": test_dataset['label'][i],
                "output": outputs[i]
            }
            result_dict_list.append(current_result)
                
        json.dump(result_dict_list, open(inference_output_path, 'w'), indent=2)
        
        ''' 3) Re-Prediction '''
        ## 2. Re-prediction
        predicted_test_dataset_dict = json.load(open(inference_output_path))
        repredict_test_dataset = load_dataset("json", data_files=inference_output_path, split="train")
        
        previous_outputs = repredict_test_dataset['output']
        index_no_answer = [i for i, ex in enumerate(previous_outputs) if '[Answer]' not in ex]
        input_strings_no_answer = [repredict_test_dataset[i] for i in index_no_answer]
        
        for i in range(len(input_strings_no_answer)):
            input_strings_no_answer[i]['input'] += input_strings_no_answer[i]['output'] + '\n\n[Answer]\nTherefore, the answer is '
        
        dataloader_repredict = DataLoader(input_strings_no_answer, batch_size=test_batch_size, collate_fn=collate_fn_nolabel_for_repredict)
        outputs_repredict = inference(dataloader_repredict, model, tokenizer, max_new_tokens=test_max_new_tokens)
        
        print(f"Repredicted {len(outputs_repredict)} answers")
        
        for ix, i in enumerate(index_no_answer):
            predicted_test_dataset_dict[i]['input'] = input_strings_no_answer[ix]['input']
            predicted_test_dataset_dict[i]['output'] = f'[Answer] The answer is {outputs_repredict[ix]}'
        
        json.dump(predicted_test_dataset_dict, open(repredicted_output_path, 'w'), indent=2)
        
        r_total_correct, r_extracted_ratio, r_extracted_correct = result_evaluation(repredicted_output_path)
        total_correct, extracted_ratio, extracted_correct = result_evaluation(inference_output_path)
        output_statis_dict = {
            "test/epoch": epoch,
            "test/total_correct": total_correct,
            "test/extracted_ratio": extracted_ratio,
            "test/extracted_correct": extracted_correct,
            "test/repredicted_total_correct": r_total_correct,
            "test/repredicted_extracted_ratio": r_extracted_ratio,
            "test/repredicted_extracted_correct": r_extracted_correct,
        }
        
        wandb.log(output_statis_dict)
        json.dump(output_statis_dict, open(os.path.join(output_dir, "test_result.json"), 'w'), indent=4)
        
        model = None
        del model
        torch.cuda.empty_cache()
        
        wandb.finish()
        

if __name__ == "__main__":
    fire.Fire(main)