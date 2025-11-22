#!/bin/bash
#$ -N lean_meme-context-bernoulli-model_olmo2_7bit_lora256-split_seed_SEED_SPLIT
#$ -q long.q@supergpu*
#$ -l gpu=1,gpu_ram=22G,ram_free=20G,mem_free=20G,h=!(supergpu5|supergpu19|supergpu17|supergpu20)
#$ -e /mnt/matylda5/iyusuf/exps/better_score/logs/$JOB_NAME_$JOB_ID.e
#$ -o /mnt/matylda5/iyusuf/exps/better_score/logs/$JOB_NAME_$JOB_ID.o

# lean_meme-context-beta-model_olmo2_7bit_lora256-split_seed_SEED_SPLIT
## -N lean_meme-context-beta-model_olmo2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3-train_human_from_full_llmj_full
#-loss_sum_priorkl
#-skip_rationale_skip_question-b2
N_GPUS=1
echo "Running job $JOB_NAME with ID $JOB_ID on host $(hostname)"
eval "$(conda shell.bash hook 2> /dev/null)"
conda activate transformers_27_env
#lean_meme-context-beta-model_olmo2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3-train_llmj
#lean_meme-context-beta-train_human-val_M1_6_M2_5-model_olmo2_1bit_lora256-split_SPLIT
#lean_meme-context-beta-model_olmo2_7bit_lora256-split_seed_SEED_SPLIT
#lean_meme-context-beta-model_gemma3_12bit_lora256-split_seed_SEED_SPLIT
export TRANSFORMERS_OFFLINE=1
export HF_HOME=~/work/huggingface_cache/
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
TENSORBOARD_DIR=/mnt/matylda5/iyusuf/exps/better_score/tacl/runs/
MODELS_DIR=/mnt/matylda5/iyusuf/exps/better_score/tacl/models/

mkdir -p $TENSORBOARD_DIR
mkdir -p $MODELS_DIR

cd /mnt/matylda5/iyusuf/exps/better_score || {
    echo "No such directory /mnt/matylda5/iyusuf/exps/better_score"
    exit 1
}
args=(
    # General training arguments
    # Training arguments
    --max_data_length 1000
    --max_steps=4000
    --val_steps=100
    --batch_size=1
    --accumulation_steps=4
    --warmup_steps=100
    --peak_lr=5e-5
    --lr_ratio_classifier=1.0
    --min_lr_ratio=0.01
    --weight_decay=0.0
    --max_grad_norm=5.0
    --early_stopping_patience=50
    # Data arguments
    --num_workers=2
    # Model and data related arguments
    --score_type="bernoulli"
#    --model="allenai/OLMo-2-0425-1B-Instruct"
#    --tokenizer="allenai/OLMo-2-0425-1B-Instruct"
#    --tokenizer="allenai/OLMo-2-1124-7B-Instruct"
#    --model="/mnt/matylda5/iyusuf/exps/better_score/tacl/models//lean_meme-context-beta-model_olmo2_1bit_full-split_unseen_modality_gemma_3n_2b_audio_flamingo_3-train_llmj_full-steps10k/best/model/lm/"
#    --model="/mnt/matylda5/iyusuf/exps/better_score/tacl/models/lean_meme-context-beta-model_olmo2_7bit_lora256-split_seed_SEED_SPLIT-train_full_llmj-steps10k/best/model/lm/"
    --model="allenai/OLMo-2-1124-7B-Instruct"
#    --model="google/gemma-3-1b-it"
#    --model="Qwen/Qwen3-4B-Instruct-2507"
#    --model="Qwen/Qwen3-0.6B"
#    --model="google/embeddinggemma-300m"
#    --model="google/gemma-3-270m"
#    --model="meta-llama/Llama-3.2-1B-Instruct"
    --layers_to_use=-1
#    --use_cls_token
    --log_dir=$TENSORBOARD_DIR/$JOB_NAME
    --output_dir=$MODELS_DIR/$JOB_NAME
    --lora_rank=256
    --quantization_level="4bit"
#    --skip_rationale
#    --skip_question
#    --add_transcript
#    --dataset_sampling_weights 1 4
#    --train_data /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_llmj_p1.json /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_human.json
#    --train_data /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_human.json
#    --train_data
#    /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_human.json
#    /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_llmj_p1.json
#    /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/val_llmj_p1.json
#    /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/test_llmj_p1.json
#    /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_human.json
#    /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_binary_para-a.json
#    /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_binary_para-q.json
#    /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_binary_orig.json
#    /mnt/matylda4/kesiraju/tools/potato/ratings_for_meme/llmj_prompt1.1_mmau_test_mini.json
#    /mnt/matylda4/kesiraju/tools/potato/ratings_for_meme/llmj_prompt1.1_MMAR.json
#    /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/llmj6/train_llmj2.1.json

#     --train_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/SPLIT/train_reformatted.json
#     --val_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/SPLIT/dev_reformatted.json

#     --train_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/seed_SEED/SPLIT/train_full_llmj_w_context_expected_answer.json
#     --train_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/seed_SEED/SPLIT/train_concatenated_w_context_expected_answer.json
#     --train_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/seed_SEED/SPLIT/train_llmj_w_context_expected_answer.json

    --train_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/seed_SEED/SPLIT/train_reformatted.json
    --val_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/seed_SEED/SPLIT/dev_reformatted.json
#    --train_data
#    /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/unseen_modality_gemma_3n_2b_audio_flamingo_3/train_reformatted.json
#    /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/unseen_modality_gemma_3n_2b_audio_flamingo_3/train_llmj_w_context_expected_answer.json
#    --val_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/unseen_modality_gemma_3n_2b_audio_flamingo_3/dev_llmj_w_context_expected_answer.json
#    --train_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/unseen_modality_gemma_3n_2b_audio_flamingo_3/train_reformatted.json
#    /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/unseen_modality_gemma_3n_2b_audio_flamingo_3/train_full_llmj_w_context_expected_answer.json



#    --val_data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/unseen_modality_gemma_3n_2b_audio_flamingo_3/dev_reformatted.json


#    --dataset_sampling_weights 1 4 1 1 1


#    --train_data /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/llmj/train_rankRANK.json
#    --train_data /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_llmj_p1.json
#    --load_checkpoint /mnt/matylda5/iyusuf/exps/better_score/models/lean_meme-context-beta-train_human-val_M1_6_M2_5/best/model
#     --load_checkpoint /mnt/matylda5/iyusuf/exps/better_score/tacl/models//lean_meme-context-beta-model_olmo2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3-train_llmj/best/model
)

export CUDA_VISIBLE_DEVICES=$(python ~/free_gpus.py $N_GPUS) || {
    echo "Could not obtain GPU."
    exit 1
}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#python -u lean_meme.py "${args[@]}"

export TOKENIZERS_PARALLELISM=false
#if [ ! -f $MODELS_DIR/$JOB_NAME/.done ]; then
if [ ! -f $MODELS_DIR/$JOB_NAME/checkpoint_4000/metrics.yaml ]; then
    echo "Training model $JOB_NAME"
    torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS lean_meme.py "${args[@]}" && touch $MODELS_DIR/$JOB_NAME/.done
else
    echo "Model $JOB_NAME already trained. Exiting."
    exit 0
fi
