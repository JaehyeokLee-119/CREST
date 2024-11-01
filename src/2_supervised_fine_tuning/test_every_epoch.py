import os
from test import main as test_main
import fire

def main(
    model_dir,
    eval_data_path,
    eval_dir_name,
    CoT=False,
    batch_size=4,
    max_new_tokens=15,
    template_dict_path= None,
    base_model: str = None
):
    print(f"current working directory: {os.getcwd()}")
    print(f"visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    if model_dir != None:
        target_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and 'checkpoint' in d]
        for model_weight_dir in target_dirs:
            output_path = os.path.join(model_weight_dir, eval_dir_name)
            os.makedirs(output_path, exist_ok=True)
            
            print(f"model path: {model_weight_dir}\noutput path: {output_path}")
            test_main(
                base_model=base_model,
                eval_data_path=eval_data_path,
                model_dir=model_weight_dir,
                output_dir=output_path,
                CoT=CoT,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                template_dict_path=template_dict_path,
                fewshot_path=None
            )
            
if __name__ == '__main__':
    fire.Fire(main)


