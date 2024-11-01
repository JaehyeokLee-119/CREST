# home=$(pwd)
cd ${home}/src/1_rationale_generation

base_model_name=meta-llama/Meta-Llama-3-8B
train_data=${home}/resources/data/ReClor-train.jsonl
fewshot_sample=${home}/resources/fewshots/ReClor.json
output_directory=${home}/outputs/data_ReClor
template=${home}/resources/templates/template_ReClor_inference_CoT.json
N=1
gpu=0

CUDA_VISIBLE_DEVICES=${gpu} python rationale_generation.py \
    --base_model_name ${base_model_name} \
    --batch_size 48 \
    --max_gen_len 512 \
    --data ${train_data} \
    --output_directory ${output_directory} \
    --correct_n ${N} --incorrect_n 0 \
    --fewshot_samples_path ${fewshot_sample} \
    --template_file_path ${template}

CUDA_VISIBLE_DEVICES=${gpu} python answer_prediction.py \
    --base_model_name ${base_model_name} \
    --batch_size 64 \
    --max_gen_len 5 \
    --data ${train_data} \
    --rationale_directory ${output_directory} \
    --template_file_path ${template}

python filtering.py \
    --prediction_directory ${output_directory}
    
CUDA_VISIBLE_DEVICES=${gpu} python consistency_test.py \
    --base_model_name ${base_model_name} \
    --batch_size 96 \
    --max_gen_len 5 \
    --data ${train_data} \
    --rationale_directory ${output_directory} \
    --template_file_path ${template}

python consistency_evaluation.py \
    --rationale_directory ${output_directory}
    