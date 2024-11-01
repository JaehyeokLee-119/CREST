# home=$(pwd)
src_home=${home}/src/2_supervised_fine_tuning
cd ${src_home}

model_dir=${home}/models/ReClor/M_Label
base_model_name=meta-llama/Meta-Llama-3-8B

gpu=0
epoch=4
lr=5e-5
key='WANDB project name'

CUDA_VISIBLE_DEVICES=${gpu} python train.py \
    --base_model ${base_model_name} \
    --train_data_path ${home}/resources/data/ReClor-train_passage.jsonl \
    --output_dir ${model_dir}/${key} \
    --sanity_check 0 \
    --lr ${lr} \
    --key ${key} \
    --logging_frequency 5 \
    --train_on_completion_only True \
    --CoT False \
    --simulated_batch 8 \
    --per_device_train_batch_size 4 \
    --padding_side 'left' \
    --template_dict_path ${home}/resources/templates/template_ReClor_inference_direct.json \
    --num_epochs ${epoch}

CUDA_VISIBLE_DEVICES=${gpu} python test_every_epoch.py \
    --base_model ${base_model_name} \
    --model_dir ${model_dir}/${key} \
    --eval_data_path ${home}/resources/data/ReClor-val.jsonl \
    --eval_dir_name val \
    --CoT False \
    --batch_size 4 \
    --max_new_tokens 150 \
    --template_dict_path ${home}/resources/templates/template_ReClor_inference_direct.json

CUDA_VISIBLE_DEVICES=${gpu} python test_every_epoch.py \
    --base_model ${base_model_name} \
    --model_dir ${model_dir}/${key} \
    --eval_data_path ${home}/resources/data/ReClor-test.jsonl \
    --eval_dir_name test \
    --CoT False \
    --batch_size 4 \
    --max_new_tokens 150 \
    --template_dict_path ${home}/resources/templates/template_ReClor_inference_direct.json
