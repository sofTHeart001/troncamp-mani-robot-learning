#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
ckpt_setting=${INTERACT_CKPT_SETTING:-${task_config}-${expert_data_num}}
kl_weight=${INTERACT_KL_WEIGHT:-10.0}
chunk_size=${INTERACT_CHUNK_SIZE:-50}
hidden_dim=${INTERACT_HIDDEN_DIM:-512}
batch_size=${INTERACT_BATCH_SIZE:-8}
dim_feedforward=${INTERACT_DIM_FEEDFORWARD:-3200}
nheads=${INTERACT_NHEADS:-8}
dropout=${INTERACT_DROPOUT:-0.1}
latent_dim=${INTERACT_LATENT_DIM:-32}
use_cvae=${INTERACT_USE_CVAE:-1}
cvae_encoder_layers=${INTERACT_CVAE_ENCODER_LAYERS:-4}
num_blocks=${INTERACT_NUM_BLOCKS:-3}
num_cls_tokens_arm=${INTERACT_NUM_CLS_TOKENS_ARM:-7}
num_cls_tokens_image=${INTERACT_NUM_CLS_TOKENS_IMAGE:-5}
n_pre_decoder_layers=${INTERACT_N_PRE_DECODER_LAYERS:-2}
n_post_decoder_layers=${INTERACT_N_POST_DECODER_LAYERS:-2}
n_sync_decoder_layers=${INTERACT_N_SYNC_DECODER_LAYERS:-1}
num_epochs=${INTERACT_NUM_EPOCHS:-6000}
lr=${INTERACT_LR:-1e-5}
lr_backbone=${INTERACT_LR_BACKBONE:-1e-5}
save_freq=${INTERACT_SAVE_FREQ:-2000}
warmup_steps=${INTERACT_WARMUP_STEPS:-0}
grad_clip=${INTERACT_GRAD_CLIP:-0}
val_ratio=${INTERACT_VAL_RATIO:-0.2}

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./inter_act_ckpt/inter-act-${task_name}/${ckpt_setting} \
    --policy_class InterACT \
    --kl_weight "${kl_weight}" \
    --chunk_size "${chunk_size}" \
    --hidden_dim "${hidden_dim}" \
    --batch_size "${batch_size}" \
    --dim_feedforward "${dim_feedforward}" \
    --nheads "${nheads}" \
    --dropout "${dropout}" \
    --use_cvae "${use_cvae}" \
    --latent_dim "${latent_dim}" \
    --cvae_encoder_layers "${cvae_encoder_layers}" \
    --num_blocks "${num_blocks}" \
    --num_cls_tokens_arm "${num_cls_tokens_arm}" \
    --num_cls_tokens_image "${num_cls_tokens_image}" \
    --n_pre_decoder_layers "${n_pre_decoder_layers}" \
    --n_post_decoder_layers "${n_post_decoder_layers}" \
    --n_sync_decoder_layers "${n_sync_decoder_layers}" \
    --num_epochs "${num_epochs}" \
    --lr "${lr}" \
    --lr_backbone "${lr_backbone}" \
    --save_freq "${save_freq}" \
    --warmup_steps "${warmup_steps}" \
    --grad_clip "${grad_clip}" \
    --val_ratio "${val_ratio}" \
    --state_dim 16 \
    --seed ${seed}
