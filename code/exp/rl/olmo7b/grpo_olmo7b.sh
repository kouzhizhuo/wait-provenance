#!/bin/bash
# One verl GRPO training phase for the OLMo-3-1025-7B fork (D42/D43). Copy of infra/scripts/grpo_p1.sh (untouched) with: OLMo defaults
# (model dir = base + OLMo RL-Zero chat template, data = RAW-problem MATH L3-5 parquet, reward = P1 grader + math-verify), MICRO=2 GMU=0.5 (measured: 37.7 GB peak
# per GPU on H800-80GB), and the rope fix (olmo3_rope_fix via model.external_lib; PYTHONPATH must contain code/olmo7b).
# Env: TAG GPUS SEED BETA STEPS SAVE_FREQ TEST_FREQ VAL_BEFORE   (REF: BETA=0.001, beta=0: BETA=0 => use_kl_loss=False). Resumable (resume_mode=auto).
set -x
R=${ARR_ROOT:?set ARR_ROOT}; TAG=${TAG:?}; GPUS=${GPUS:?}; SEED=${SEED:-1}; BETA=${BETA:-0.001}; STEPS=${STEPS:-400}
SAVE_FREQ=${SAVE_FREQ:-10}; TEST_FREQ=${TEST_FREQ:-20}; VAL_BEFORE=${VAL_BEFORE:-True}
MODEL=${MODEL:-Olmo-3-1025-7B-rlzero}; DATA_DIR=${DATA_DIR:-$R/data/math_verl_olmo}; REWARD_FILE=${REWARD_FILE:-$R/code/reward_math_p1grader.py}
NG=$(echo $GPUS | tr ',' '\n' | wc -l); RUN=$R/runs/P1/$TAG; mkdir -p $RUN
source $R/envs/rl/bin/activate
export CACHE_ROOT=/tmp/arr_cache_${SLURM_JOB_ID:-$$}; mkdir -p $CACHE_ROOT
export TRITON_CACHE_DIR=$CACHE_ROOT/triton VLLM_CACHE_ROOT=$CACHE_ROOT/vllm TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/inductor
export CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false RAY_DEDUP_LOGS=0 VLLM_LOGGING_LEVEL=WARNING
export PYTHONPATH=$R/code/olmo7b${PYTHONPATH:+:$PYTHONPATH} ARR_ROOT=$R
export RAY_TMPDIR=/tmp/ray_$(echo $TAG | cut -c1-10)_$$; mkdir -p $RAY_TMPDIR   # AF_UNIX path limit 107 bytes
if [ "$BETA" = "0" ] || [ "$BETA" = "0.0" ]; then KL="actor_rollout_ref.actor.use_kl_loss=False"; else KL="actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=$BETA"; fi
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo data.seed=$SEED \
  data.train_files=$DATA_DIR/train.parquet data.val_files=$DATA_DIR/val.parquet \
  data.train_batch_size=128 data.max_prompt_length=1024 data.max_response_length=3072 data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path=$R/models/$MODEL actor_rollout_ref.model.external_lib=olmo3_rope_fix \
  actor_rollout_ref.model.use_remove_padding=False +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO:-2} actor_rollout_ref.actor.entropy_coeff=0 $KL \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GMU:-0.5} actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO:-2} actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO:-2} \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  reward.custom_reward_function.path=$REWARD_FILE reward.custom_reward_function.name=compute_score \
  trainer.logger='["console"]' trainer.project_name=P1 trainer.experiment_name=$TAG \
  trainer.n_gpus_per_node=$NG trainer.nnodes=1 trainer.total_training_steps=$STEPS trainer.total_epochs=1000 \
  trainer.save_freq=$SAVE_FREQ trainer.test_freq=$TEST_FREQ trainer.val_before_train=$VAL_BEFORE \
  trainer.default_local_dir=$RUN/ckpt trainer.rollout_data_dir=$RUN/rollouts trainer.resume_mode=auto
