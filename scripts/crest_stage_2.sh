# home=$(pwd)
src_home=${home}/src/2_supervised_fine_tuning
cd ${src_home}

model_dir=${home}/models/ReClor/M_SFT
st1_output_directory=${home}/outputs/data_ReClor
base_model_name=meta-llama/Meta-Llama-3-8B

gpu=0
epoch=4
lr=5e-5
threshold=3 # threshold = F-t (Num of follow-up questions - tolerance)
key='WANDB project name'

python filtering.py \
    --stage1_output_dir ${st1_output_directory}

CUDA_VISIBLE_DEVICES=${gpu} python train.py \
    --base_model ${base_model_name} \
    --train_data_path ${st1_output_directory}/4-hq_rationale_with_consistency_score_${threshold}.json \
    --output_dir ${model_dir}/${key} \
    --sanity_check 0 \
    --lr ${lr} \
    --num_epochs ${epoch} \
    --key ${key} \
    --logging_frequency 5 \
    --train_on_completion_only True \
    --CoT True \
    --simulated_batch 8 \
    --per_device_train_batch_size 2 \
    --padding_side 'left' \
    --template_dict_path ${home}/resources/templates/template_ReClor_inference_CoT.json

# test on validation set
CUDA_VISIBLE_DEVICES=${gpu} python test_every_epoch.py \
    --base_model ${base_model_name} \
    --model_dir ${model_dir}/${key} \
    --eval_data_path ${home}/resources/data/ReClor-val.jsonl \
    --eval_dir_name val \
    --CoT True \
    --batch_size 16 \
    --max_new_tokens 512 \
    --template_dict_path ${home}/resources/templates/template_ReClor_inference_CoT.json

# test on test set
CUDA_VISIBLE_DEVICES=${gpu} python test_every_epoch.py \
    --base_model ${base_model_name} \
    --model_dir ${model_dir}/${key} \
    --eval_data_path ${home}/resources/data/ReClor-test.jsonl \
    --eval_dir_name test \
    --CoT True \
    --batch_size 16 \
    --max_new_tokens 512 \
    --template_dict_path ${home}/resources/templates/template_ReClor_inference_CoT.json

