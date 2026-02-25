#!/bin/bash
# On-policy Cartridge distillation via Tokasaurus + veRL
#
# Prerequisites:
#   1. Modal Tokasaurus server running:
#      modal deploy verl/verl/workers/rollout/tokasaurus_rollout/modal_tokasaurus.py
#
#   2. Training data prepared (see examples/cartridge_distill/prepare_data.py)
#
#   3. cartridges package installed:
#      pip install -e cartridges/
#
# Usage:
#   bash examples/cartridge_distill/run_cartridge_distill.sh

set -x

TOKASAURUS_URL="${TOKASAURUS_URL:-https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=~/data/cartridge_distill/train.parquet \
    data.val_files=~/data/cartridge_distill/val.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct \
    actor_rollout_ref.model.external_lib=cartridges \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.cartridge.enabled=True \
    actor_rollout_ref.actor.cartridge.checkpoint_path=hazyresearch/cartridge-wauoq23f \
    actor_rollout_ref.actor.cartridge.num_tokens=2048 \
    actor_rollout_ref.actor.cartridge.lr=0.02 \
    \
    actor_rollout_ref.rollout.name=tokasaurus \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.custom.tokasaurus_url=${TOKASAURUS_URL} \
    'actor_rollout_ref.rollout.custom.cartridges=[{id: hazyresearch/cartridge-wauoq23f, source: huggingface}]' \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=cartridge_distill \
    trainer.experiment_name=longhealth_llama3b \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    "$@"
