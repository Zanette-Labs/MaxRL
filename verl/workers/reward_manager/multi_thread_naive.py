# Copyright 2024 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional
import time
import signal

import ray
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.utils.reward_score.math_verify import extract_solution

# -----------------------------------------------------------------------------
# math_verify imports
# -----------------------------------------------------------------------------
try:
    from math_verify import verify, parse
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    raise RuntimeError("Please install math-verify: pip install math-verify")

# Extraction configs - MUST match MathVerifyScorer for consistency
_GOLD_EXTRACTION_CONFIG = (LatexExtractionConfig(),)  # GT uses LaTeX extraction (from \boxed{})
_PRED_EXTRACTION_CONFIG = (ExprExtractionConfig(), LatexExtractionConfig())  # Pred uses both


# =============================================================================
# Per-item timeout helpers
# =============================================================================

class _ItemTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ItemTimeout


# =============================================================================
# MathVerify scorer (ONE per actor process)
# =============================================================================

class MathVerifyScorer:
    """
    Thread-safe math scorer. One instance per Ray actor process.
    The key insight is that math_metric MUST be created once per Ray actor process,
    NOT called directly in the main process.
    """
    def __init__(self):
        # No longer need math_metric - we use parse + verify directly
        pass

    def compute_score(
        self,
        model_output: str,
        ground_truth_unboxed: str,
        timeout_score: float,
        per_item_timeout_s: int,
    ) -> Tuple[float, Optional[str]]:
        """
        Compute score and extract predicted answer in ONE pass.
        
        Returns:
            (score, extracted_answer) - extracted_answer is None if extraction failed
        """
        # hard per-item timeout using SIGALRM
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(per_item_timeout_s)
        try:
            # Parse GT (boxed format) and prediction in ONE call each
            gt_boxed = f"\\boxed{{{ground_truth_unboxed}}}"
            
            # Extract GT answer
            gt_parsed = parse(gt_boxed, extraction_config=_GOLD_EXTRACTION_CONFIG)
            
            # Extract predicted answer - this is the SAME extraction we return for majority voting
            pred_parsed = parse(model_output, extraction_config=_PRED_EXTRACTION_CONFIG)
            
            # Get extracted answer string for majority voting
            extracted_answer = None
            if pred_parsed:
                extracted_answer = str(pred_parsed[0]).strip() if pred_parsed else None
                if not extracted_answer:
                    extracted_answer = None
            
            # Verify if prediction matches GT
            if gt_parsed and pred_parsed:
                is_correct = verify(gt_parsed, pred_parsed)
                score = 1.0 if is_correct else 0.0
            else:
                score = 0.0
            
            return float(score), extracted_answer
            
        except (_ItemTimeout, TimeoutException):
            # Timeout from either our alarm or math_verify's internal timeout
            return float(timeout_score), None
        except BaseException:
            # Catch ALL exceptions including those raised from signal handlers
            # in unexpected contexts (e.g., weakref cleanup, GC, etc.)
            # This is necessary because signal handlers are not thread-safe
            return 0.0, None
        finally:
            signal.alarm(0)
    
# =============================================================================
# Ray actor for parallel reward computation
# =============================================================================

@ray.remote(
    max_restarts=0,
    max_task_retries=0,
)
class RewardScoreActor:
    """
    Ray actor that holds its own MathVerifyScorer instance.
    This ensures thread safety as each actor runs in its own process.
    """
    def __init__(self):
        self._scorer = MathVerifyScorer()

    def compute_scores_batch(
        self,
        batch: List[Tuple[int, str, str]],  # (item_idx, response_str, ground_truth_unboxed)
        timeout_score: float,
        per_item_timeout_s: int,
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Compute scores and extract answers for a batch of items.
        
        Returns:
            List of (item_idx, {"score": float, "accuracy": float, "extracted_answer": str|None})
        """
        out: List[Tuple[int, Dict[str, Any]]] = []
        for item_idx, response_str, ground_truth in batch:
            score, extracted_answer = self._scorer.compute_score(
                response_str,
                ground_truth,
                timeout_score,
                per_item_timeout_s,
            )

            out.append((item_idx, {
                "score": float(score), 
                "accuracy": float(score),
                "extracted_answer": extracted_answer,
            }))
        return out


# =============================================================================
# Majority voting helper
# =============================================================================

def most_frequent_answer_with_score(
    answer_score_pairs: List[Tuple[Optional[str], float]]
) -> Tuple[Optional[str], bool]:
    """
    Given a list of (answer, score) pairs, determine majority voting result.
    
    Algorithm:
    1. Group answers by exact string match (None is treated as a special answer)
    2. Find the largest group (majority/mode)
    3. Check if ANY answer in the largest group has score > 0
    
    Returns:
        (majority_answer, is_correct) where is_correct is True if the majority
        group contains at least one answer with score > 0.
    """
    if not answer_score_pairs:
        return None, False
    
    # Group by exact string match (including None), track scores for each group
    # groups[answer_string] = list of scores
    # None and empty string are both treated as None
    groups: Dict[Optional[str], List[float]] = defaultdict(list)
    for ans, score in answer_score_pairs:
        # Normalize answer: strip if string, treat empty string as None
        if ans is not None:
            normalized_ans = ans.strip() if ans else ""
            # Empty string after normalization should be treated as None
            if not normalized_ans:
                normalized_ans = None
        else:
            normalized_ans = None
        groups[normalized_ans].append(score)
    
    # Find the largest group
    if not groups:
        return None, False
    
    majority_answer = max(groups.keys(), key=lambda k: len(groups[k]))
    majority_scores = groups[majority_answer]
    
    # If majority answer is None (no answer extracted), majority voting fails
    # This prevents cases where extract_solution() finds nothing but math_metric()
    # still returns accuracy=1 from finding answers elsewhere
    if majority_answer is None:
        return None, False
    
    # Check if any answer in the majority group is correct (score > 0)
    is_correct = any(s > 0 for s in majority_scores)
    
    return majority_answer, is_correct


# Helper to get the number of reward actors
def get_num_reward_actors(
    cpu_per_actor: float = 1.0,
    max_actors: Optional[int] = None,
    min_actors: int = 1,
) -> int:
    resources = ray.available_resources()
    num_cpus = int(resources.get("CPU", 0))

    if num_cpus <= 0:
        return min_actors

    n = int(num_cpus // cpu_per_actor)

    if max_actors is not None:
        n = min(n, max_actors)

    return max(n, min_actors)


# =============================================================================
# Multi-Thread Naive Reward Manager
# =============================================================================

@register("multi_thread")
class MultiThreadNaiveRewardManager:
    """
    A thread-safe reward manager that uses Ray actors to parallelize 
    math reward computation.
    
    Key features:
    - Uses multiple Ray actors, each with its own MathVerifyScorer instance
    - Supports per-item and per-batch timeouts
    - Calculates accuracy interval fractions for analysis
    - Supports majority voting for evaluation
    """
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        num_reward_actors: Optional[int] = None,
        batch_size: int = 8,
        in_flight_batches_per_actor: int = 4,
        per_item_timeout_s: int = 10,
        per_batch_timeout_s: float = 120.0,
        poll_interval_s: float = 0.5,
        timeout_score: float = 0.0,
        enable_majority_voting: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.enable_majority_voting = enable_majority_voting

        self._batch_size = int(batch_size)
        self._timeout_score = float(timeout_score)
        self._per_item_timeout_s = int(per_item_timeout_s)
        self._per_batch_timeout_s = float(per_batch_timeout_s)
        self._poll_interval_s = float(poll_interval_s)

        if isinstance(num_reward_actors, int):
            self.num_reward_actors = num_reward_actors
        else:
            self.num_reward_actors = int(get_num_reward_actors())

        print("\nReward model is using this many reward actors: ", self.num_reward_actors, "\n")

        self._actors = [RewardScoreActor.remote() for _ in range(self.num_reward_actors)]
        self._next_actor = 0
        self._max_inflight_batches = self.num_reward_actors * in_flight_batches_per_actor

        # Accuracy interval scheme for logging
        accuracy_intervals = [(0.0, 0.0)]
        low = 0.0
        n_pow = 10
        while n_pow >= 0:
            high = 2.0 ** (-n_pow)
            accuracy_intervals.append((low, high))
            low = high
            n_pow -= 1
        accuracy_intervals.append((1.0, 1.0))
        self.accuracy_intervals = accuracy_intervals

    def _pick_actor(self):
        """Round-robin actor selection."""
        a = self._actors[self._next_actor]
        self._next_actor = (self._next_actor + 1) % len(self._actors)
        return a

    def _compute_accuracy_interval_fractions(
        self, accuracies: List[float]
    ) -> Dict[str, float]:
        """Compute fractions for accuracy intervals."""
        if not accuracies:
            result = {}
            for i, (low, high) in enumerate(self.accuracy_intervals):
                if (
                    i == 0 
                    or i == len(self.accuracy_intervals) - 1
                ):
                    label = f"[{low}, {high}]"
                elif i == len(self.accuracy_intervals) - 2:
                    label = f"({low}, {high})"
                else:
                    label = f"({low}, {high}]"
                result[label] = 0.0
            return result

        counts = [0 for _ in self.accuracy_intervals]
        n = len(accuracies)

        for acc in accuracies:
            for idx, (low, high) in enumerate(self.accuracy_intervals):
                in_interval = False

                if idx == 0:
                    in_interval = (acc == 0.0)
                elif idx == len(self.accuracy_intervals) - 1:
                    in_interval = (acc == 1.0)
                else:
                    is_second_to_last = (idx == len(self.accuracy_intervals) - 2)
                    greater_than_lower_bound = (acc > low)
                    less_than_higher_bound = (acc < high)
                    less_than_or_equal_to_higher_bound = (acc <= high)

                    if is_second_to_last:
                        in_interval = (
                            greater_than_lower_bound
                            and less_than_higher_bound
                        )
                    else:
                        in_interval = (
                            greater_than_lower_bound
                            and less_than_or_equal_to_higher_bound
                        )

                if in_interval:
                    counts[idx] += 1
                    break

        result = {}
        for i, (low, high) in enumerate(self.accuracy_intervals):
            if (
                i == 0 
                or i == len(self.accuracy_intervals) - 1
            ):
                label = f"[{low}, {high}]"
            elif i == len(self.accuracy_intervals) - 2:
                label = f"({low}, {high})"
            else:
                label = f"({low}, {high}]"
            result[label] = counts[i] / n

        return result
    
    def _normalize_answer(self, ans: str) -> str:
        """Minimal normalization of answer."""
        return ans.strip()

    def __call__(self, data: DataProto, return_dict: bool = False):
        # Preserve shortcut behavior if rm_scores already exist
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: Dict[str, list] = defaultdict(list)

        # Logging counters
        num_item_timeouts = 0
        num_batch_timeouts = 0
        num_batch_exceptions = 0
        num_batches_ok = 0

        already_print_data_sources: Dict[str, int] = {}

        prompt_to_accuracies = defaultdict(lambda: defaultdict(list))
        prompt_to_answers = defaultdict(lambda: defaultdict(list))
        prompt_to_gt_answer = defaultdict(dict)

        n = len(data)

        # Decode all items once
        items: List[Dict[str, Any]] = []
        for i in range(n):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            attn_mask = data_item.batch["attention_mask"]

            valid_prompt_len = attn_mask[:prompt_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = data_item.batch["responses"]
            valid_resp_len = attn_mask[prompt_len:].sum()
            valid_resp_ids = response_ids[:valid_resp_len]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            
            # Get prompt_key for grouping responses by question
            # Priority: prompt_id > full prompt decode > (ground_truth + prompt hash)
            has_prompt_id = "prompt_id" in data_item.non_tensor_batch
            if has_prompt_id:
                prompt_key = data_item.non_tensor_batch["prompt_id"]
            else:
                # Decode full prompt (not just valid part) for better uniqueness
                full_prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                # If full prompt is too short (likely just system prompt), use ground_truth as part of key
                if len(full_prompt_str) < 100:
                    # Use hash of prompt_ids tensor + ground_truth for uniqueness
                    prompt_hash = hash(tuple(prompt_ids.tolist())) if hasattr(prompt_ids, 'tolist') else hash(str(prompt_ids))
                    prompt_key = f"{ground_truth}_{prompt_hash}"
                else:
                    prompt_key = full_prompt_str
            
            # Debug print for first few items
            if i < 5:
                print(f"[DEBUG prompt_key] i={i}, has_prompt_id={has_prompt_id}, "
                      f"prompt_key_type={type(prompt_key).__name__}, "
                      f"prompt_key_len={len(str(prompt_key))}, "
                      f"prompt_key_preview={str(prompt_key)[:80]}...")
            # Get finish_reason if available (from vLLM rollout)
            # "stop" = normal EOS, "length" = max length reached
            finish_reason = data_item.non_tensor_batch.get("finish_reasons", None)

            items.append(
                dict(
                    i=i,
                    response=response_str,
                    ground_truth=ground_truth,
                    data_source=data_source,
                    prompt_key=prompt_key,
                    valid_resp_len=valid_resp_len,
                    finish_reason=finish_reason,
                )
            )

        # Batching scheduler for parallel execution
        pending: Set[ray.ObjectRef] = set()
        start_time: Dict[ray.ObjectRef, float] = {}
        ref_to_batch: Dict[ray.ObjectRef, List[int]] = {}
        results: Dict[int, Dict[str, float]] = {}

        next_i = 0

        def submit_one_batch():
            nonlocal next_i
            if next_i >= n:
                return

            batch: List[int] = []
            while next_i < n and len(batch) < self._batch_size:
                batch.append(next_i)
                next_i += 1

            payload = [(i, items[i]["response"], items[i]["ground_truth"]) for i in batch]
            ref = self._pick_actor().compute_scores_batch.remote(
                payload,
                self._timeout_score,
                self._per_item_timeout_s,
            )

            pending.add(ref)
            start_time[ref] = time.time()
            ref_to_batch[ref] = batch

        # Prime the pipeline with initial batches
        while len(pending) < self._max_inflight_batches and next_i < n:
            submit_one_batch()

        # Gather results
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=self._poll_interval_s)
            now = time.time()

            for ref in ready:
                pending.remove(ref)
                batch = ref_to_batch.pop(ref)
                start_time.pop(ref, None)

                try:
                    pairs = ray.get(ref)
                    num_batches_ok += 1
                    for idx, out in pairs:
                        results[idx] = out
                        if out.get("score", None) == self._timeout_score:
                            num_item_timeouts += 1
                except Exception:
                    num_batch_exceptions += 1
                    for i in batch:
                        results[i] = {"score": self._timeout_score, "accuracy": 0.0}

            # Batch-level safety timeout
            for ref in list(pending):
                if now - start_time.get(ref, now) > self._per_batch_timeout_s:
                    pending.remove(ref)
                    batch = ref_to_batch.pop(ref)
                    start_time.pop(ref, None)
                    try:
                        ray.cancel(ref, force=True)
                    except Exception:
                        pass
                    num_batch_timeouts += 1
                    for i in batch:
                        results[i] = {"score": self._timeout_score, "accuracy": 0.0}

            # Backfill with new batches
            while len(pending) < self._max_inflight_batches and next_i < n:
                submit_one_batch()

        # Post-process and compute metrics
        num_truncated = 0  # Count responses that reached max length
        is_validation = data.meta_info.get("validate", False)
        for i, info in enumerate(items):
            out = results.get(i, {"score": self._timeout_score, "accuracy": 0.0})
            
            # If response reached max length (truncated), set reward to 0
            # finish_reason == "length" means generation stopped due to max_tokens limit
            # Only apply this penalty during training, not validation
            finish_reason = info.get("finish_reason", None)
            if finish_reason == "length" and not is_validation:
                out = {"score": 0.0, "accuracy": 0.0}
                num_truncated += 1
            
            # Convert valid_resp_len to int if it's a tensor
            valid_resp_len = info["valid_resp_len"]
            if hasattr(valid_resp_len, 'item'):
                valid_resp_len = valid_resp_len.item()
            valid_resp_len = int(valid_resp_len)
            
            reward_tensor[i, valid_resp_len - 1] = float(out["score"])

            ds = info["data_source"]
            pk = info["prompt_key"]
            ground_truth = info["ground_truth"]
            prompt_to_accuracies[ds][pk].append(float(out["accuracy"]))

            # Examine prints for debugging
            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0

            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1
                print("[data_source]", ds)
                print("[response]", info["response"])
                print("[ground_truth]", info["ground_truth"])
                print("[score]", out["score"])
                print("[accuracy]", out["accuracy"])

            # Use extracted_answer from Ray Actor (already extracted during scoring)
            # Store (answer, score) pair for majority voting (if enabled)
            # Include ALL responses: those with extracted answers and those without (None)
            extracted_answer = out.get("extracted_answer")
            if extracted_answer is not None:
                normalized_answer = self._normalize_answer(extracted_answer)
                # Empty string after normalization should be treated as None
                if not normalized_answer:
                    normalized_answer = None
            else:
                # No answer extracted, treat as None
                normalized_answer = None
            
            # Always store extracted_answer and prompt_key in reward_extra_info for external use
            # This allows incremental processing to accumulate and compute majority voting at the end
            reward_extra_info["extracted_answer"].append(normalized_answer)
            reward_extra_info["accuracy"].append(float(out["accuracy"]))
            reward_extra_info["prompt_key"].append(pk)
            reward_extra_info["data_source"].append(ds)
            
            if self.enable_majority_voting:
                prompt_to_answers[ds][pk].append((
                    normalized_answer,
                    float(out["accuracy"])  # Use accuracy (same as score) for correctness check
                ))

            # Extract & store GT answer once per prompt
            if pk not in prompt_to_gt_answer[ds]:
                if ground_truth is not None:
                    prompt_to_gt_answer[ds][pk] = self._normalize_answer(ground_truth)
                else:
                    raise ValueError("No GT answer!")

        # Log truncation statistics
        if num_truncated > 0:
            print(f"[RewardManager] {num_truncated}/{n} responses truncated (reached max length), reward set to 0")

        # Compute interval fractions per datasource + global
        all_prompt_accuracies: List[float] = []
        per_ds_interval_fractions: Dict[str, float] = {}
        per_ds_prompt_accuracies: Dict[str, List[float]] = {}  # Store per-ds accuracies for aggregation

        for ds, prompts in prompt_to_accuracies.items():
            per_prompt_accs = []
            for pk, accs in prompts.items():
                if accs:
                    mean_acc = sum(accs) / len(accs)
                    per_prompt_accs.append(mean_acc)
                    all_prompt_accuracies.append(mean_acc)
            per_ds_prompt_accuracies[ds] = per_prompt_accs  # Store for aggregation
            frac_dict = self._compute_accuracy_interval_fractions(per_prompt_accs)
            for interval_label, frac in frac_dict.items():
                per_ds_interval_fractions[f"{ds}/{interval_label}"] = frac

        # =================================================================
        # Special patch: Aggregate math500_level1-5 into math500_all for interval fractions
        # =================================================================
        # Match data sources that end with math500_level1-5 (handles both "math500_level1" and "guanning-ai/math500_level1")
        math500_level_suffixes = {f"math500_level{i}" for i in range(1, 6)}
        found_interval_levels = [ds for ds in per_ds_prompt_accuracies.keys() if any(ds.endswith(suffix) for suffix in math500_level_suffixes)]
        
        if len(found_interval_levels) > 0:
            # Combine all accuracies from math500_level1-5
            math500_all_accs = []
            for ds in found_interval_levels:
                math500_all_accs.extend(per_ds_prompt_accuracies[ds])
            
            # Compute interval fractions for math500_all
            math500_all_frac_dict = self._compute_accuracy_interval_fractions(math500_all_accs)
            for interval_label, frac in math500_all_frac_dict.items():
                per_ds_interval_fractions[f"math500_all/{interval_label}"] = frac
            
            # Also compute and store mean accuracy for math500_all
            math500_all_mean_acc = sum(math500_all_accs) / len(math500_all_accs) if math500_all_accs else 0.0
            print(f"[RewardManager] math500_all interval fractions computed from {len(found_interval_levels)} levels, "
                  f"mean_accuracy={math500_all_mean_acc:.4f}, num_prompts={len(math500_all_accs)}")

        global_interval_fractions = self._compute_accuracy_interval_fractions(all_prompt_accuracies)

        # Calculate majority voting metrics (only if enabled AND during validation)
        per_ds_majority_vote_accuracy = None
        global_majority_vote_accuracy = None
        global_mv_num_prompts = None
        per_ds_mv_num_prompts = None

        if self.enable_majority_voting and is_validation:
            # Majority voting: find most frequent answer per prompt, check if any response with that answer has accuracy > 0
            per_ds_majority_vote_accuracy = {}
            per_ds_mv_num_prompts = {}
            global_mv_correct = 0
            global_mv_total = 0
            
            # Track math500_level correct and total counts directly
            math500_level_suffixes = {f"math500_level{i}" for i in range(1, 6)}
            math500_all_mv_correct = 0
            math500_all_mv_total = 0
            
            for ds, prompts in prompt_to_answers.items():
                ds_correct = 0
                ds_total = 0
                
                # Check if this datasource is a math500_level
                is_math500_level = any(ds.endswith(suffix) for suffix in math500_level_suffixes)
                
                for prompt_key, ans_score_list in prompts.items():
                    # ans_score_list is List[Tuple[Optional[str], float]] - (answer, accuracy) pairs
                    if not ans_score_list:
                        continue
                    
                    # Find majority answer and check if it's correct
                    majority_ans, is_correct = most_frequent_answer_with_score(ans_score_list)
                    
                    ds_total += 1
                    global_mv_total += 1
                    if is_correct:
                        ds_correct += 1
                        global_mv_correct += 1
                    
                    # If this is a math500_level, directly accumulate correct and total counts
                    if is_math500_level:
                        math500_all_mv_total += 1
                        if is_correct:
                            math500_all_mv_correct += 1
                
                per_ds_majority_vote_accuracy[ds] = (ds_correct / ds_total) if ds_total else 0.0
                per_ds_mv_num_prompts[ds] = ds_total

            global_majority_vote_accuracy = (global_mv_correct / global_mv_total) if global_mv_total else 0.0
            global_mv_num_prompts = global_mv_total
            
            # Compute math500_all majority voting accuracy from directly accumulated counts
            if math500_all_mv_total > 0:
                math500_all_mv_accuracy = math500_all_mv_correct / math500_all_mv_total
                per_ds_majority_vote_accuracy["math500_all"] = math500_all_mv_accuracy
                per_ds_mv_num_prompts["math500_all"] = math500_all_mv_total

        if return_dict:
            result = {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "accuracy_interval_fractions_per_prompt_global": global_interval_fractions,
                "accuracy_interval_fractions_per_prompt_per_datasource": per_ds_interval_fractions,
            }
            # Only add majority voting metrics if enabled
            if self.enable_majority_voting:
                result["majority_vote_accuracy_global"] = global_majority_vote_accuracy
                result["majority_vote_accuracy_per_datasource"] = per_ds_majority_vote_accuracy
                result["majority_vote_num_prompts_global"] = global_mv_num_prompts
                result["majority_vote_num_prompts_per_datasource"] = per_ds_mv_num_prompts
            return result

        return reward_tensor

