# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

# SELF MODIFICATION: Remove REMAX AND DAPO FILTER


import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from cores.allocation_v1 import allocate_rollout


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        
        # Initialize per-question statistics tracking
        self.per_question_statistics = {}
        self.current_epoch = 0
        
        # EMA decay for std score estimation (higher = more weight on recent epochs)
        self.ema_decay = getattr(self.config.algorithm, 'ema_decay', 0.7)
        
        # Minimum repeat times for each question (to avoid cases when std = 0)
        self.min_repeat_times = getattr(self.config.algorithm, 'min_repeat_times', 4)
        
        # Enable/disable adaptive repeat times (default: True)
        self.enable_adaptive_repeat = getattr(self.config.algorithm, 'enable_adaptive_repeat', True)
        print(f"Adaptive repeat training: {'enabled' if self.enable_adaptive_repeat else 'disabled'}")
        
        # Enable selection-based allocation from a fixed rollout pool per question
        # If enabled, we will:
        # 1) Generate a fixed number (rollout.n) of rollouts per question to estimate std (accuracy)
        # 2) Compute per-question allocation based on estimated std
        # 3) Randomly select k responses per question from the generated pool according to allocation
        # 4) Use the selected responses for policy update (no new generations)
        self.enable_selection_allocation = getattr(self.config.algorithm, 'enable_selection_allocation', False)
        self.selection_random_seed = getattr(self.config.algorithm, 'selection_random_seed', None)

        # Estimated std score for each question (used for adaptive repeat times)
        self.estimated_std_score_per_question = {}
        
        # Initialize accumulated token tracking
        self.accumulated_pool_tokens = 0
        self.accumulated_selected_tokens = 0
        self.accumulated_saved_tokens = 0

        # Initialize accumulated rollout tracking
        self.accumulated_pool_rollouts = 0
        self.accumulated_selected_rollouts = 0
        self.accumulated_saved_rollouts = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            self.current_epoch = epoch
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                # Calculate repeat plan
                if self.enable_selection_allocation:
                    # For selection-based allocation, generate a fixed pool per question for std estimation
                    default_repeat = self.config.actor_rollout_ref.rollout.n
                    question_uuids = batch_dict.get('index', [])
                    if not isinstance(question_uuids, (list, tuple, np.ndarray)):
                        question_uuids = [question_uuids]
                    repeat_times_per_question_gen = {uuid: default_repeat for uuid in question_uuids}
                    # Placeholder for allocation used later in advantage computation
                    repeat_times_per_question = None
                    # Apply uniform repeating to build the generation pool
                    gen_batch = self._apply_adaptive_repeat(gen_batch, batch_dict, repeat_times_per_question_gen)
                else:
                    # Calculate adaptive repeat times for each question (historical std-based)
                    repeat_times_per_question = self._calculate_adaptive_repeat_times(batch_dict)
                    # Log repeat times for monitoring
                    if self.current_epoch > 0 and self.global_steps % 10 == 0:  # Log every 10 steps after first epoch
                        total_repeats = sum(repeat_times_per_question.values())
                        avg_repeats = total_repeats / len(repeat_times_per_question) if repeat_times_per_question else 0
                        print(f"Epoch {self.current_epoch}, Step {self.global_steps}: "
                              f"Adaptive repeats - Total: {total_repeats}, Avg: {avg_repeats:.2f}, "
                              f"Questions: {len(repeat_times_per_question)}")
                    
                    gen_batch = self._apply_adaptive_repeat(gen_batch, batch_dict, repeat_times_per_question)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    if self.enable_selection_allocation:
                        new_batch = self._apply_adaptive_repeat(new_batch, batch_dict, repeat_times_per_question_gen)
                    else:
                        new_batch = self._apply_adaptive_repeat(new_batch, batch_dict, repeat_times_per_question)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                            
                        # Update per-question statistics
                        self._update_per_question_statistics(batch_dict, reward_extra_infos_dict)

                        # Always track pool tokens/rollouts (total generated)
                        pool_tokens = self._count_generated_tokens(new_batch)
                        pool_rollouts = len(new_batch)
                        self.accumulated_pool_tokens += pool_tokens
                        self.accumulated_pool_rollouts += pool_rollouts
                        
                        # For selection-based allocation, select from pool based on std and then apply KL after selection
                        if self.enable_selection_allocation:
                            # Compare tokens before vs after selection
                            selected_batch, repeat_times_per_question = self._allocate_and_select_from_pool(new_batch, batch_dict, batch_budget=len(batch_dict['index'])*self.config.actor_rollout_ref.rollout.n//2, upper=self.config.actor_rollout_ref.rollout.n)
                            selected_tokens = self._count_generated_tokens(selected_batch)
                            selected_rollouts = len(selected_batch)
                            saved_tokens = max(0, pool_tokens - selected_tokens)
                            saved_rollouts = max(0, pool_rollouts - selected_rollouts)
                            saved_ratio = (saved_tokens / pool_tokens) if pool_tokens > 0 else 0.0

                            # Update accumulated tracking
                            self.accumulated_selected_tokens += selected_tokens
                            self.accumulated_saved_tokens += saved_tokens
                            self.accumulated_selected_rollouts += selected_rollouts
                            self.accumulated_saved_rollouts += saved_rollouts

                            metrics.update({
                                "tokens/pool_total": pool_tokens,
                                "tokens/selected_total": selected_tokens,
                                "tokens/saved_total": saved_tokens,
                                "tokens/saved_ratio": saved_ratio,
                                "tokens/accumulated_pool_total": self.accumulated_pool_tokens,
                                "tokens/accumulated_selected_total": self.accumulated_selected_tokens,
                                "tokens/accumulated_saved_total": self.accumulated_saved_tokens,
                                "tokens/accumulated_saved_ratio": (self.accumulated_saved_tokens / self.accumulated_pool_tokens) if self.accumulated_pool_tokens > 0 else 0.0,
                                "rollouts/pool_total": pool_rollouts,
                                "rollouts/selected_total": selected_rollouts,
                                "rollouts/saved_total": saved_rollouts,
                                "rollouts/accumulated_pool_total": self.accumulated_pool_rollouts,
                                "rollouts/accumulated_selected_total": self.accumulated_selected_rollouts,
                                "rollouts/accumulated_saved_total": self.accumulated_saved_rollouts,
                                "rollouts/accumulated_saved_ratio": (self.accumulated_saved_rollouts / self.accumulated_pool_rollouts) if self.accumulated_pool_rollouts > 0 else 0.0,
                            })
                            
                            # compute rewards. apply_kl_penalty if available on the selected subset
                            if self.config.algorithm.use_kl_in_reward:
                                selected_batch, kl_metrics = apply_kl_penalty(
                                    selected_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                selected_batch.batch["token_level_rewards"] = selected_batch.batch["token_level_scores"]

                            batch = selected_batch
                        else:
                            # When not using selection allocation, all pool items are used
                            self.accumulated_selected_tokens += pool_tokens
                            self.accumulated_selected_rollouts += pool_rollouts
                            
                            metrics.update({
                                "tokens/pool_total": pool_tokens,
                                "tokens/accumulated_pool_total": self.accumulated_pool_tokens,
                                "tokens/accumulated_selected_total": self.accumulated_selected_tokens,
                                "rollouts/pool_total": pool_rollouts,
                                "rollouts/accumulated_pool_total": self.accumulated_pool_rollouts,
                                "rollouts/accumulated_selected_total": self.accumulated_selected_rollouts,
                            })
                            
                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                new_batch, kl_metrics = apply_kl_penalty(
                                    new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(
                                    kl_metrics
                                )  # TODO: This will be cleared if we use multiple genenration batches
                            else:
                                new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                            batch = new_batch

                    # === Updating ===
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    
                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        
                        # Calculate effective number of repeats for this batch
                        # In selection mode, this uses the allocated repeats based on the pool selection
                        effective_num_repeat = self._get_effective_num_repeat(batch_dict, repeat_times_per_question if repeat_times_per_question is not None else {})
                        
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=effective_num_repeat,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # Add per-question statistics to metrics
                per_question_summary = self.get_per_question_statistics_summary()
                
                # Only include extreme stats if we have finalized epoch data
                # (extreme stats are only meaningful after epoch finalization)
                basic_metrics = {
                    "per_question/total_questions": per_question_summary["total_questions"],
                    "per_question/avg_mean_acc": per_question_summary["avg_mean_acc"],
                    "per_question/avg_mean_score": per_question_summary["avg_mean_score"],
                    "per_question/avg_std_acc": per_question_summary["avg_std_acc"],
                    "per_question/avg_std_score": per_question_summary["avg_std_score"]
                }
                
                metrics.update(basic_metrics)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
            
            # Finalize epoch statistics at the end of each epoch
            self._finalize_epoch_statistics()
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
    
    def _update_per_question_statistics(self, batch_dict, reward_extra_infos_dict):
        """
        Update per-question statistics with acc and scores from reward_extra_infos_dict.
        
        Args:
            batch_dict: The original batch dictionary containing 'index' field with question UUIDs
            reward_extra_infos_dict: Dictionary containing 'acc' and 'score' arrays
        """
        if 'index' not in batch_dict:
            print("Warning: 'index' field not found in batch_dict, skipping per-question statistics update")
            return
            
        question_uuids = batch_dict['index']
        if not isinstance(question_uuids, (list, tuple, np.ndarray)):
            question_uuids = [question_uuids]
            
        # Extract acc and scores from reward_extra_infos_dict
        acc_values = reward_extra_infos_dict.get('acc', [])
        score_values = reward_extra_infos_dict.get('score', [])
        
        # Handle case where we have different numbers of entries
        max_entries = max(len(question_uuids), len(acc_values), len(score_values))
        
        for i in range(max_entries):
            # Get question UUID (cycle through if fewer UUIDs than entries)
            question_uuid = question_uuids[i % len(question_uuids)] if len(question_uuids) > 0 else f"unknown_{i}"
            
            # Get acc and score values (use 0.0 as default if not available)
            acc = acc_values[i] if i < len(acc_values) else 0.0
            score = score_values[i] if i < len(score_values) else 0.0
            
            # Initialize question statistics if not exists
            if question_uuid not in self.per_question_statistics:
                self.per_question_statistics[question_uuid] = {
                    'current_epoch_acc_values': [],  # Accumulator for current epoch
                    'current_epoch_score_values': [],  # Accumulator for current epoch
                    'mean_acc_per_epoch': [],  # List of mean acc for each epoch
                    'mean_score_per_epoch': [],  # List of mean score for each epoch
                    'std_acc_per_epoch': [],  # List of std acc for each epoch
                    'std_score_per_epoch': [],  # List of std score for each epoch
                    'count_per_epoch': [],  # List of counts for each epoch
                    'last_processed_epoch': -1
                }
            
            # Update statistics for current epoch
            stats = self.per_question_statistics[question_uuid]
            stats['current_epoch_acc_values'].append(acc)
            stats['current_epoch_score_values'].append(score)
    
    def _finalize_epoch_statistics(self):
        """
        Finalize statistics for the current epoch and save to file.
        """
        import json
        import os
        
        for question_uuid, stats in self.per_question_statistics.items():
            # Only process if we haven't processed this epoch for this question yet
            if stats['last_processed_epoch'] < self.current_epoch:
                current_acc_values = stats['current_epoch_acc_values']
                current_score_values = stats['current_epoch_score_values']
                
                if current_acc_values or current_score_values:
                    # Calculate statistics for this epoch
                    mean_acc = np.mean(current_acc_values) if current_acc_values else 0.0
                    mean_score = np.mean(current_score_values) if current_score_values else 0.0
                    std_acc = np.std(current_acc_values) if len(current_acc_values) > 1 else 0.0
                    std_score = np.std(current_score_values) if len(current_score_values) > 1 else 0.0
                    count = len(current_acc_values)
                    
                    # Append to epoch lists
                    stats['mean_acc_per_epoch'].append(mean_acc)
                    stats['mean_score_per_epoch'].append(mean_score)
                    stats['std_acc_per_epoch'].append(std_acc)
                    stats['std_score_per_epoch'].append(std_score)
                    stats['count_per_epoch'].append(count)
                    
                    # Clear current epoch accumulators
                    stats['current_epoch_acc_values'] = []
                    stats['current_epoch_score_values'] = []
                    stats['last_processed_epoch'] = self.current_epoch
                    
                    # Update EMA estimation for std_score
                    if question_uuid not in self.estimated_std_score_per_question:
                        # First epoch for this question
                        self.estimated_std_score_per_question[question_uuid] = std_score
                    else:
                        # EMA update: new_estimate = decay * current_std + (1 - decay) * old_estimate
                        old_estimate = self.estimated_std_score_per_question[question_uuid]
                        self.estimated_std_score_per_question[question_uuid] = (
                            self.ema_decay * std_score + (1 - self.ema_decay) * old_estimate
                        )
        
        # Save to file
        self._save_per_question_statistics_to_file()
        
    
    def _save_per_question_statistics_to_file(self):
        """
        Save per-question statistics to a JSON file.
        """
        import json
        import os
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.config.trainer.default_local_dir, "per_question_stats")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for saving (exclude current epoch accumulators)
        save_data = {}
        for question_uuid, stats in self.per_question_statistics.items():
            save_data[question_uuid] = {
                'mean_acc_per_epoch': stats['mean_acc_per_epoch'],
                'mean_score_per_epoch': stats['mean_score_per_epoch'],
                'std_acc_per_epoch': stats['std_acc_per_epoch'],
                'std_score_per_epoch': stats['std_score_per_epoch'],
                'count_per_epoch': stats['count_per_epoch'],
                'total_epochs_seen': len(stats['mean_acc_per_epoch'])
            }
        
        # Save to file
        filename = f"per_question_statistics_epoch_{self.current_epoch}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Saved per-question statistics to {filepath}")
        
        # Also save the latest as a separate file for easy access
        latest_filepath = os.path.join(output_dir, "per_question_statistics_latest.json")
        with open(latest_filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def get_per_question_statistics_summary(self):
        """
        Get a summary of per-question statistics.
        Handles both current epoch data (before finalization) and finalized epoch data.
        
        Returns:
            dict: Summary containing total questions tracked and average metrics across all questions
        """
        if not self.per_question_statistics:
            return {"total_questions": 0, "avg_mean_acc": 0.0, "avg_mean_score": 0.0, "avg_std_acc": 0.0, "avg_std_score": 0.0}
        
        total_questions = len(self.per_question_statistics)
        
        # Collect statistics - use finalized data if available, otherwise compute from current epoch
        current_mean_accs = []
        current_mean_scores = []
        current_std_accs = []
        current_std_scores = []
        
        for stats in self.per_question_statistics.values():
            # Try to use finalized epoch data first
            if (stats['mean_acc_per_epoch'] and stats['std_acc_per_epoch'] and 
                stats['mean_score_per_epoch'] and stats['std_score_per_epoch']):
                # Use latest finalized epoch data
                current_mean_accs.append(stats['mean_acc_per_epoch'][-1])
                current_mean_scores.append(stats['mean_score_per_epoch'][-1])
                current_std_accs.append(stats['std_acc_per_epoch'][-1])
                current_std_scores.append(stats['std_score_per_epoch'][-1])
            elif (stats['current_epoch_acc_values'] or stats['current_epoch_score_values']):
                # Fall back to current epoch data (compute on the fly)
                acc_values = stats['current_epoch_acc_values']
                score_values = stats['current_epoch_score_values']
                
                if acc_values:
                    mean_acc = np.mean(acc_values)
                    std_acc = np.std(acc_values) if len(acc_values) > 1 else 0.0
                    current_mean_accs.append(mean_acc)
                    current_std_accs.append(std_acc)
                
                if score_values:
                    mean_score = np.mean(score_values)
                    std_score = np.std(score_values) if len(score_values) > 1 else 0.0
                    current_mean_scores.append(mean_score)
                    current_std_scores.append(std_score)
        
        return {
            "total_questions": total_questions,
            "avg_mean_acc": np.mean(current_mean_accs) if current_mean_accs else 0.0,
            "avg_mean_score": np.mean(current_mean_scores) if current_mean_scores else 0.0,
            "avg_std_acc": np.mean(current_std_accs) if current_std_accs else 0.0,
            "avg_std_score": np.mean(current_std_scores) if current_std_scores else 0.0,
        }
    
    def _save_checkpoint(self):
        """
        Override parent method to include per-question statistics in checkpoints.
        """
        # Call parent save method first
        super()._save_checkpoint()
        
        # Save per-question statistics
        import json
        import os
        
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        
        # Save per-question statistics
        stats_data = {
            'per_question_statistics': self.per_question_statistics,
            'estimated_std_score_per_question': self.estimated_std_score_per_question,
            'current_epoch': self.current_epoch,
            'ema_decay': self.ema_decay,
            'min_repeat_times': self.min_repeat_times,
            'enable_adaptive_repeat': self.enable_adaptive_repeat
        }
        
        stats_path = os.path.join(local_global_step_folder, "per_question_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"Saved per-question statistics to {stats_path}")
    
    def _load_checkpoint(self):
        """
        Override parent method to restore per-question statistics from checkpoints.
        """
        # Get the resumed global steps from parent method
        resumed_steps = super()._load_checkpoint()
        
        if self.config.trainer.resume_mode != "disable":
            import json
            import os
            
            # Find the checkpoint folder
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            
            # Construct the global step folder path
            if hasattr(self, 'global_steps') and self.global_steps > 0:
                global_step_folder = os.path.join(checkpoint_folder, f"global_step_{self.global_steps}")
                stats_path = os.path.join(global_step_folder, "per_question_stats.json")
                
                if os.path.exists(stats_path):
                    try:
                        with open(stats_path, 'r') as f:
                            stats_data = json.load(f)
                        
                        self.per_question_statistics = stats_data.get('per_question_statistics', {})
                        self.estimated_std_score_per_question = stats_data.get('estimated_std_score_per_question', {})
                        self.current_epoch = stats_data.get('current_epoch', 0)
                        self.ema_decay = stats_data.get('ema_decay', self.ema_decay)
                        # Only restore min_repeat_times if it was saved, otherwise keep config value
                        if 'min_repeat_times' in stats_data:
                            self.min_repeat_times = stats_data['min_repeat_times']
                        # Only restore enable_adaptive_repeat if it was saved, otherwise keep config value
                        if 'enable_adaptive_repeat' in stats_data:
                            self.enable_adaptive_repeat = stats_data['enable_adaptive_repeat']
                        
                        print(f"Loaded per-question statistics from {stats_path}")
                        print(f"Restored {len(self.per_question_statistics)} question statistics")
                        print(f"Resumed from epoch {self.current_epoch}")
                        print(f"Adaptive repeat mode: {'enabled' if self.enable_adaptive_repeat else 'disabled'}")
                    except Exception as e:
                        print(f"Warning: Failed to load per-question statistics: {e}")
                        # Initialize empty if loading fails
                        self.per_question_statistics = {}
                        self.estimated_std_score_per_question = {}
                else:
                    print(f"No per-question statistics found at {stats_path}, starting fresh")
        
        return resumed_steps
    
    def _calculate_adaptive_repeat_times(self, batch_dict):
        """
        Calculate adaptive repeat times for each question based on estimated std scores.
        If adaptive repeat is disabled, returns uniform repeat times.
        
        Args:
            batch_dict: The original batch dictionary containing 'index' field with question UUIDs
            
        Returns:
            dict: Mapping from question UUID to repeat times
        """
        
        question_uuids = batch_dict['index']
        if not isinstance(question_uuids, (list, tuple, np.ndarray)):
            question_uuids = [question_uuids]
        
        # If adaptive repeat is disabled, always use uniform repeat times
        if not self.enable_adaptive_repeat:
            default_repeat = self.config.actor_rollout_ref.rollout.n
            print(f"Adaptive repeat disabled - using uniform repeat times: {default_repeat}")
            return {uuid: default_repeat for uuid in question_uuids}
        
        # For first epoch, use default repeat times
        if self.current_epoch == 0:
            default_repeat = self.config.actor_rollout_ref.rollout.n
            return {uuid: default_repeat for uuid in question_uuids}
        
        # Calculate adaptive repeat times for subsequent epochs
        batch_size = len(question_uuids)
        total_budget = self.config.actor_rollout_ref.rollout.n * batch_size
        
        # Get estimated std scores for questions in this batch
        batch_std_scores = []
        for uuid in question_uuids:
            if uuid in self.estimated_std_score_per_question:
                batch_std_scores.append(self.estimated_std_score_per_question[uuid])
            else:
                # Use a default std score for new questions
                batch_std_scores.append(0.1)  # Default moderate variance
        
        # Avoid division by zero
        total_std_score = sum(batch_std_scores)
        if total_std_score == 0:
            # If all std scores are 0, use minimum repeat times
            return {uuid: self.min_repeat_times for uuid in question_uuids}
        
        # Calculate repeat times proportional to std scores
        repeat_times = {}
        for i, uuid in enumerate(question_uuids):
            proportion = batch_std_scores[i] / total_std_score
            repeat_time = max(self.min_repeat_times, int(total_budget * proportion))  # At least min_repeat_times
            repeat_times[uuid] = repeat_time
        
        # Ensure total doesn't exceed budget (due to rounding)
        total_assigned = sum(repeat_times.values())
        if total_assigned > total_budget:
            # Proportionally reduce but respect minimum repeat times
            factor = total_budget / total_assigned
            for uuid in repeat_times:
                repeat_times[uuid] = max(self.min_repeat_times, int(repeat_times[uuid] * factor))
        
        return repeat_times
    
    def _apply_adaptive_repeat(self, gen_batch, batch_dict, repeat_times_per_question):
        """
        Apply adaptive repeat times to generate batch based on per-question requirements.
        
        Args:
            gen_batch: The generation batch (DataProto)
            batch_dict: Original batch dictionary with question UUIDs
            repeat_times_per_question: Dict mapping question UUID to repeat times
            
        Returns:
            DataProto: Batch with adaptive repeating applied
        """
        
        question_uuids = batch_dict['index']
        if not isinstance(question_uuids, (list, tuple, np.ndarray)):
            question_uuids = [question_uuids]
        
        # Create a list to hold repeated batches
        repeated_batches = []
        
        for i, uuid in enumerate(question_uuids):
            # Get the repeat times for this question
            repeat_times = repeat_times_per_question.get(uuid, 1)
            
            # Extract the single item for this question
            single_item = gen_batch[i:i+1]  # DataProto slice
            
            # Repeat this single item
            repeated_item = single_item.repeat(repeat_times=repeat_times, interleave=False)
            
            repeated_batches.append(repeated_item)
        
        # Concatenate all repeated batches
        if repeated_batches:
            final_batch = DataProto.concat(repeated_batches)
        else:
            # Fallback to original if no batches created
            final_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        
        return final_batch
    
    def _get_effective_num_repeat(self, batch_dict, repeat_times_per_question):
        """
        Calculate the effective number of repeats for advantage computation.
        Since different questions may have different repeat times, we use the average.
        
        Args:
            batch_dict: Original batch dictionary with question UUIDs
            repeat_times_per_question: Dict mapping question UUID to repeat times
            
        Returns:
            float: Effective number of repeats (average across questions)
        """
        if 'index' not in batch_dict or not repeat_times_per_question:
            return self.config.actor_rollout_ref.rollout.n
        
        question_uuids = batch_dict['index']
        if not isinstance(question_uuids, (list, tuple, np.ndarray)):
            question_uuids = [question_uuids]
        
        # Calculate average repeat times
        total_repeats = sum(repeat_times_per_question.get(uuid, 1) for uuid in question_uuids)
        avg_repeats = total_repeats / len(question_uuids) if len(question_uuids) > 0 else 1
        
        return avg_repeats

    def _count_generated_tokens(self, data_proto: DataProto) -> int:
        """
        Count total generated tokens using the response mask.

        Args:
            data_proto: DataProto containing generated sequences

        Returns:
            int: total number of generated tokens across the batch
        """
        try:
            response_mask = compute_response_mask(data_proto)
            if isinstance(response_mask, torch.Tensor):
                return int(response_mask.sum().item())
            # Fallback for unexpected types
            return int(torch.tensor(response_mask).sum().item())
        except Exception:
            # If response mask cannot be computed, return 0 as a safe fallback
            raise ValueError("Failed to compute response mask")
            # return 0

    # allocate_rollout(question_accs, batch_budget, upper=32)
    def _allocate_and_select_from_pool(self, pool_batch, 
                                       original_batch_dict,
                                       batch_budget,
                                       upper,
                                       ):
        """
        From a pool of generated responses (uniform fixed count per question),
        estimate per-question std of accuracy, compute allocation, and select k
        responses per question without regenerating.

        Args:
            pool_batch: DataProto containing generated responses and reward extras (including 'acc')
            original_batch_dict: Original batch dict containing 'index' field

        Returns:
            tuple: (selected_batch: DataProto, repeat_times_per_question: dict)
        """

        # Normalize question uuid list
        question_uuids = original_batch_dict['index']
        if not isinstance(question_uuids, (list, tuple, np.ndarray)):
            question_uuids = [question_uuids]

        # Access per-sample uuids and accuracies in the pool
        pool_uuids = pool_batch.non_tensor_batch.get('index')
        pool_acc = pool_batch.non_tensor_batch.get('acc')

        # If acc is missing, we cannot estimate std; return as-is
        if pool_uuids is None or pool_acc is None:
            raise ValueError("Pool batch must contain 'index' and 'acc' in non_tensor_batch")

        # Build mapping uuid -> indices in the pool
        uuid_to_indices: dict = {}
        for idx, u in enumerate(pool_uuids):
            uuid_to_indices.setdefault(u, []).append(idx)

        # Compute std of accuracy per question
        acc_per_uuid: dict[str, float] = {}
        for uuid in question_uuids:
            idxs = uuid_to_indices.get(uuid)
            if len(idxs) <= 1:
                acc_per_uuid[uuid] = 0.0
            else:
                acc_vals = np.asarray([pool_acc[i] for i in idxs], dtype=float)
                acc_per_uuid[uuid] = float(np.mean(acc_vals))


        # Compute allocation based on std without normalizing by total std
        # Rescale std in [0, 0.5] -> [0, 1] and use scaled_std * rollout.n as allocation
        default_repeat = self.config.actor_rollout_ref.rollout.n

        allocated_budget = allocate_rollout(list(acc_per_uuid.values()), batch_budget=batch_budget, upper=upper)
        allocated_budget = [budget + self.min_repeat_times for budget in allocated_budget]  # Add a small constant to avoid zero allocation
        repeat_times_per_question: dict[str, int] = {}
        for uuid, alloc in zip(question_uuids, allocated_budget):
            repeat_times_per_question[uuid] = alloc
        
        # Randomly select indices per question according to allocation
        if self.selection_random_seed is not None:
            rng = np.random.default_rng(self.selection_random_seed + int(self.global_steps))
        else:
            rng = np.random.default_rng()

        selected_indices: list[int] = []
        per_uuid_selected_indices: dict[str, list[int]] = {}
        for uuid in question_uuids:
            available = uuid_to_indices.get(uuid)
            k = min(len(available), repeat_times_per_question.get(uuid))
            if k <= 0:
                continue
            if k == len(available):
                chosen = available
            else:
                chosen = rng.choice(available, size=k, replace=False).tolist()
            per_uuid_selected_indices[uuid] = chosen
            selected_indices.extend(chosen)

        # If nothing selected due to corner cases, fall back to full pool
        if len(selected_indices) == 0:
            selected_indices = list(range(len(pool_batch)))
            # Adjust repeats to uniform minimum within pool size
            for uuid in question_uuids:
                repeat_times_per_question[uuid] = min(default_repeat, max(self.min_repeat_times, 1))

        # Create the selected subset DataProto
        selected_batch = pool_batch.select_idxs(selected_indices)
        return selected_batch, repeat_times_per_question
    