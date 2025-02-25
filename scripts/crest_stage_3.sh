# home=$(pwd)
src_home=${home}/src/3_preference_learning
cd ${src_home}

base_model_name=meta-llama/Meta-Llama-3-8B

python generate_pair_data.py \
    --stage_1_result ${home}/outputs/data_ReClor \
    --output_dir ${home}/outputs/data_ReClor_stage3 \
    --template_file_path ${home}/resources/templates/template_ReClor_inference_CoT.json

gpu=0
key='TEST'

SFT_model='' # Enter the path of the SFT model (trained in stage 2)
# Example: ${home}/models/ReClor/M_SFT/<key_used_in_stage_2>/checkpoint-1000

working_dir=${home}/models/ReClor/M_CREST
training_z=${home}/outputs/data_ReClor_stage3/only_z/qjk_train_data.json
training_tz=${home}/outputs/data_ReClor_stage3/only_tz/qjk_train_data.json
lambda=0.6
lr=6e-6
epoch=4

# if SFT_model == '', stop with error message 'invalid SFT model path'
if [ -z ${SFT_model} ]; then
    echo "Input SFT model path. Example) \${home}/models/ReClor/M_SFT/<key_used_in_stage_2>/checkpoint-1000"
else
    # Preference learning with dpo
    CUDA_VISIBLE_DEVICES=${gpu} python dpo_training_ratio.py \
        --wandb_project_name ${key} \
        --base_model_name ${base_model_name} \
        --sft_peft_path ${SFT_model} \
        --working_dir ${working_dir}/${key} \
        --train_data_oq ${training_z} \
        --train_data_cs ${training_tz} \
        --save_steps 3000 \
        --max_steps 3000 \
        --ratio_lambda ${lambda} \
        --learning_rate ${lr} \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --test_batch_size 4 \
        --simulated_batch_size 8 \
        --restart_epoch 0 \
        --dpo_epochs ${epoch} \
        --sanity_check False \
        --do_test False \
        --test_template_path ${home}/resources/templates/template_ReClor_inference_CoT.json

    # test on validation set
    for ep in $(seq 1 ${epoch}); do
        CUDA_VISIBLE_DEVICES=${gpu} python dpo_model_test.py \
            --base_model_name ${base_model_name} \
            --model_dir ${working_dir}/${key}/epoch_${ep} \
            --output_dir ${working_dir}/${key}/epoch_${ep}/valid \
            --test_data ${home}/resources/data/ReClor-val.jsonl \
            --test_batch_size 4 \
            --test_template_path ${home}/resources/templates/template_ReClor_inference_CoT.json
    done

    # test on test set
    for ep in $(seq 1 ${epoch}); do
        CUDA_VISIBLE_DEVICES=${gpu} python dpo_model_test.py \
            --base_model_name ${base_model_name} \
            --model_dir ${working_dir}/${key}/epoch_${ep} \
            --output_dir ${working_dir}/${key}/epoch_${ep}/test \
            --test_data ${home}/resources/data/ReClor-test.jsonl \
            --test_batch_size 4 \
            --test_template_path ${home}/resources/templates/template_ReClor_inference_CoT.json
    done
fi
