#!/bin/bash
#$ -N infer_EXPNAME
#$ -q long.q@supergpu*
#$ -l gpu=1,gpu_ram=20G,ram_free=20G,mem_free=20G
#$ -e /mnt/matylda5/iyusuf/exps/better_score/logs/$JOB_NAME_$JOB_ID.e
#$ -o /mnt/matylda5/iyusuf/exps/better_score/logs/$JOB_NAME_$JOB_ID.o

# meme-context-beta-model_olmo2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3-train_llmj_full-steps10k
#infer_meme-context-beta-model_olmo2_7bit_lora256-split_seed_SEED_SPLIT
#infer_meme-context-beta-model_llama3_2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3
#infer_meme-context-bernoulli-train_llm_p21-test_M1_6_M2_5-model_gemma12bit_lora256-infer_M1_6_M2_5-relabel
N_GPUS=1
echo "Running job $JOB_NAME with ID $JOB_ID on host $(hostname)"
eval "$(conda shell.bash hook 2> /dev/null)"
conda activate transformers_27_env

export TRANSFORMERS_OFFLINE=1
export HF_HOME=~/work/huggingface_cache/
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
TENSORBOARD_DIR=/mnt/matylda5/iyusuf/exps/better_score/tacl/runs/
MODELS_DIR=/mnt/matylda5/iyusuf/exps/better_score/tacl/models/
INFERENCE_DIR=/mnt/matylda5/iyusuf/exps/better_score/tacl/new_inference/
#INFERENCE_DIR=/mnt/matylda5/iyusuf/exps/better_score/tacl/new_inference_dev/

cd /mnt/matylda5/iyusuf/exps/better_score || {
    echo "No such directory /mnt/matylda5/iyusuf/exps/better_score"
    exit 1
}


data=$(cat $MODELS_DIR/lean_EXPNAME/args.yaml | grep val_data | sed "s dev test g" | cut -d' ' -f2)
args=(
    --batch_size 1
#    --add_transcript
#    --skip_rationale
#    --skip_question
    --test_set_is_labeled
    --num_workers 2
#    --tokenizer_path "meta-llama/Llama-3.2-1B-Instruct"
#    --tokenizer_path "google/gemma-3-12b-it"
#    --tokenizer_path "allenai/OLMo-2-1124-7B-Instruct"
#    --tokenizer_path "allenai/OLMo-2-0425-1B-Instruct"
#    --meme_model_path $MODELS_DIR/lean_meme-context-beta-train_llm_p21-val_M1_6_M2_5-model_gemma12bit_lora256-g4/best/model
#    --meme_model $MODELS_DIR/lean_meme-context-beta-model_olmo2_1bit_lora256-split_SPLIT/best/model
#    --meme_model $MODELS_DIR/lean_meme-context-beta-model_gemma3_12bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3/best/model
#    --meme_model $MODELS_DIR/lean_meme-context-beta-model_llama3_2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3/best/model
#     --meme_model $MODELS_DIR/infer_meme-context-beta-model_llama3_2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3/best/model
#    --meme_model $MODELS_DIR/lean_meme-context-beta-model_olmo2_7bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3/best/model
#    --meme_model $MODELS_DIR/lean_meme-context-beta-model_olmo2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3-train_human_from_llmj/best/model
#    --meme_model $MODELS_DIR/unlayered_meme-context-beta-model_olmo2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3/best/model
#    --meme_model $MODELS_DIR/lean_meme-context-beta-model_olmo2_1bit_full-split_unseen_modality_gemma_3n_2b_audio_flamingo_3-train_llmj_full-steps10k/best/model
#    --meme_model $MODELS_DIR/lean_meme-context-beta-model_olmo2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3-train_human_from_full_llmj_full/best/model
#    --meme_model $MODELS_DIR/lean_meme-context-beta-model_olmo2_1bit_lora256-split_unseen_modality_gemma_3n_2b_audio_flamingo_3-train_llmj_full-steps10k/best/model
    --meme_model $MODELS_DIR/lean_EXPNAME/best/model
#    --meme_model $MODELS_DIR/lean_meme-context-beta-model_olmo2_7bit_lora256-split_seed_SEED_SPLIT/best/model
#    --data /mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/test_human.json
#    --data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/SPLIT/test_reformatted.json
#    --data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/seed_SEED/SPLIT/test_reformatted.json
#    --data /mnt/matylda4/kesiraju/tools/meme_server/data/splits/splits_unseen_lalm/unseen_modality_gemma_3n_2b_audio_flamingo_3/test_reformatted.json
    --data $data
    --output_dir=$INFERENCE_DIR/$JOB_NAME
)

export CUDA_VISIBLE_DEVICES=$(python ~/free_gpus.py $N_GPUS) || {
    echo "Could not obtain GPU."
    exit 1
}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ ! -f $INFERENCE_DIR/$JOB_NAME/.done ]; then
  python -u meme_infer.py "${args[@]}" && touch $INFERENCE_DIR/$JOB_NAME/.done
else
  echo "Job already done. To rerun delete $INFERENCE_DIR/$JOB_NAME/.done"
fi
