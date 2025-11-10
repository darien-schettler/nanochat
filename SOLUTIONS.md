# 🔓 Nanochat Bug Hunt - Solutions Guide

**⚠️ SPOILER ALERT ⚠️**

This document contains detailed solutions for all bugs in the challenge. Only refer to this if you're truly stuck or want to verify your solutions after completing a level.

---

## 🟢 Level 1: Easy - Tokenizer & Data Loading

### Bug #1: BOS Token Not Being Prepended

**Location:** `nanochat/tokenizer.py` - `RustBPETokenizer.encode()` method (around line 227)

**Symptom:** When encoding text with `prepend="<|bos|>"`, the BOS token doesn't appear at the start of the token sequence.

**Root Cause:** The prepending logic has been commented out.

**Fix:**
```python
# Buggy code:
# if prepend is not None:
#     ids.insert(0, prepend_id)

# Fixed code:
if prepend is not None:
    ids.insert(0, prepend_id)
```

Also fix the batch encoding version:
```python
# if prepend is not None:
#     for ids_row in ids:
#         ids_row.insert(0, prepend_id)

# Fixed:
if prepend is not None:
    for ids_row in ids:
        ids_row.insert(0, prepend_id)
```

**Why it matters:** BOS tokens are crucial for delimiting sequences and helping the model understand where documents begin.

---

### Bug #2: Wrong Tensor Dtype

**Location:** `nanochat/dataloader.py` - `tokenizing_distributed_data_loader()` function (around line 43)

**Symptom:** Runtime error when trying to move tensors to GPU, or dtype mismatch errors during training.

**Root Cause:** `inputs_cpu` is created with `dtype=torch.int64` but then converted to `torch.int32`, causing issues.

**Fix:**
```python
# Buggy code:
inputs_cpu = scratch[:-1].to(dtype=torch.int64)  # Wrong dtype!

# Fixed code:
inputs_cpu = scratch[:-1].to(dtype=torch.int32)  # Correct dtype
```

**Why it matters:** Consistent dtypes prevent runtime errors and ensure efficient memory usage. Token indices should be int32.

---

### Bug #3: Off-by-One Error in Targets

**Location:** `nanochat/dataloader.py` - `tokenizing_distributed_data_loader()` function (around line 44)

**Symptom:** Model learns to predict the current token instead of the next token, resulting in poor training.

**Root Cause:** Targets are created from `scratch[:-1]` instead of `scratch[1:]`, making them identical to inputs.

**Fix:**
```python
# Buggy code:
targets_cpu = scratch[:-1]  # Same as inputs!

# Fixed code:
targets_cpu = scratch[1:]  # Shifted by one position
```

**Why it matters:** Autoregressive language modeling requires predicting the next token given the current context. This is fundamental to how LLMs work.

---

## 🟡 Level 2: Medium - Training Pipeline

### Bug #1: Missing Gradient Accumulation Normalization

**Location:** `scripts/base_train.py` - Training loop (around line 271)

**Symptom:** Loss explodes when using gradient accumulation with large batch sizes. Training becomes unstable.

**Root Cause:** Loss is not divided by `grad_accum_steps` before calling `.backward()`, causing gradients to be accumulated without normalization.

**Fix:**
```python
# Buggy code:
loss.backward()  # Unnormalized!

# Fixed code:
loss = loss / grad_accum_steps  # Normalize first
loss.backward()
```

**Why it matters:** Gradient accumulation simulates larger batch sizes by accumulating gradients over multiple forward passes. Without normalization, gradients grow linearly with accumulation steps, causing instability.

---

### Bug #2: Learning Rate Warmup Off by 100x

**Location:** `scripts/base_train.py` - `get_lr_multiplier()` function (around line 160)

**Symptom:** Model doesn't learn during the warmup phase. Loss barely decreases initially.

**Root Cause:** Warmup calculation divides by `warmup_iters * 100` instead of just `warmup_iters`.

**Fix:**
```python
# Buggy code:
return (it + 1) / (warmup_iters * 100)  # 100x too small!

# Fixed code:
return (it + 1) / warmup_iters  # Correct warmup
```

**Why it matters:** Learning rate warmup prevents training instability at the start. A warmup that's 100x too slow effectively prevents the model from learning.

---

### Bug #3: Optimizer Parameters Swapped

**Location:** `nanochat/gpt.py` - `setup_optimizers()` method (around line 227)

**Symptom:** Poor training dynamics. Matrix parameters don't benefit from Muon's advantages, embeddings don't update properly.

**Root Cause:** Matrix parameters are assigned to AdamW and lm_head parameters to Muon, which is backwards.

**Fix:**
```python
# Buggy code:
adam_groups = [
    dict(params=matrix_params, lr=unembedding_lr * dmodel_lr_scale),  # Wrong!
    dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
]
muon_optimizer = MuonFactory(lm_head_params, **muon_kwargs)  # Wrong!

# Fixed code:
adam_groups = [
    dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),  # Correct
    dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
]
muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)  # Correct
```

**Why it matters:** Muon optimizer is specifically designed for matrix parameters (attention, MLP weights). AdamW is better for embeddings. Using the wrong optimizer for each parameter type reduces training efficiency.

---

### Bug #4: Inverted SFT Masking Logic

**Location:** `scripts/chat_sft.py` - `sft_data_generator()` function (around line 114)

**Symptom:** Model trains on user messages instead of assistant messages. Generates weird outputs with repeated special tokens.

**Root Cause:** Mask condition is inverted - masking where `mask_tensor == 1` instead of `mask_tensor == 0`.

**Fix:**
```python
# Buggy code:
row_targets[mask_tensor == 1] = -1  # Inverted logic!

# Fixed code:
row_targets[mask_tensor == 0] = -1  # Correct: mask where we shouldn't train
```

**Why it matters:** In supervised fine-tuning, we only want to train on assistant responses, not user prompts. The mask indicates which tokens to train on (1) vs ignore (0). Inverting this trains the model on the wrong data.

---

## 🔴 Level 3: Hard - Architecture & Performance

### Bug #1: Rotary Embedding Frequency Calculation

**Location:** `nanochat/gpt.py` - `_precompute_rotary_embeddings()` method (around line 192)

**Symptom:** Model performance degrades on longer sequences. Attention patterns break down at higher positions.

**Root Cause:** Frequency calculation uses `head_dim * 2` instead of just `head_dim`.

**Fix:**
```python
# Buggy code:
inv_freq = 1.0 / (base ** (channel_range / (head_dim * 2)))  # Wrong!

# Fixed code:
inv_freq = 1.0 / (base ** (channel_range / head_dim))  # Correct
```

**Why it matters:** Rotary Position Embeddings (RoPE) encode position information through rotation frequencies. The wrong calculation produces incorrect frequencies, breaking the model's ability to understand relative positions, especially in longer sequences.

---

### Bug #2: KV Cache Position Tracking

**Location:** `nanochat/engine.py` - `KVCache.insert_kv()` method (around line 150)

**Symptom:** Generated text becomes corrupted or repetitive after ~50 tokens. Quality degrades during long generations.

**Root Cause:** Cache position increments after the first layer instead of the last layer.

**Fix:**
```python
# Buggy code:
if layer_idx == 0:  # Wrong layer!
    self.pos = t1

# Fixed code:
if layer_idx == self.kv_cache.size(0) - 1:  # Last layer
    self.pos = t1
```

**Why it matters:** The position should only increment once per forward pass, after all layers have processed. Incrementing at the first layer causes the cache to get out of sync with the actual sequence position.

---

### Bug #3: MQA Enable Condition Inverted

**Location:** `nanochat/gpt.py` - `CausalSelfAttention.forward()` method (around line 87)

**Symptom:** Shape errors when using Multi-Query Attention (n_head != n_kv_head). Attention doesn't work correctly.

**Root Cause:** The condition for enabling GQA is inverted - uses `==` instead of `!=`.

**Fix:**
```python
# Buggy code:
enable_gqa = self.n_head == self.n_kv_head  # Inverted!

# Fixed code:
enable_gqa = self.n_head != self.n_kv_head  # Correct
```

**Why it matters:** Group Query Attention (GQA) / Multi-Query Attention (MQA) requires special handling to broadcast key/value heads to match query heads. The flag should be True when the numbers differ, not when they're equal.

---

### Bug #4: Memory Leak in KV Cache

**Location:** `nanochat/engine.py` - `KVCache.reset()` and `insert_kv()` methods

**Symptom:** Memory usage grows unboundedly during long conversations or multiple generations. Eventually causes OOM.

**Root Cause:** Two issues:
1. `reset()` doesn't actually clear the cache
2. Cache growth allocates 2x more memory than needed

**Fix:**

In `reset()` method (around line 94):
```python
# Buggy code:
def reset(self):
    self.pos = 0
    # Missing: self.kv_cache = None

# Fixed code:
def reset(self):
    self.pos = 0
    self.kv_cache = None  # Actually free the memory
```

In `insert_kv()` method (around line 138):
```python
# Buggy code:
additional_shape[4] = t_needed * 2  # Growing 2x more than needed!

# Fixed code:
additional_shape[4] = t_needed - self.kv_cache.size(4)  # Grow by exact amount needed
```

**Why it matters:** Memory leaks cause crashes in production. Proper cache management is critical for long-running inference and multi-turn conversations.

---

### Bug #5: Tool Output Token Order

**Location:** `nanochat/engine.py` - `generate()` method (around line 289)

**Symptom:** Calculator results appear at wrong positions in the output. Special tokens in wrong order.

**Root Cause:** Output tokens are injected in the wrong order - result comes before `output_start` token.

**Fix:**
```python
# Buggy code:
state.forced_tokens.extend(result_tokens)  # Wrong order!
state.forced_tokens.append(output_end)
state.forced_tokens.append(output_start)

# Fixed code:
state.forced_tokens.append(output_start)  # Correct order
state.forced_tokens.extend(result_tokens)
state.forced_tokens.append(output_end)
```

**Why it matters:** Tool use requires precise token ordering to maintain the conversation structure. Wrong ordering breaks the model's understanding of where tool outputs begin and end.

---

## 🎓 Key Takeaways

### Common Bug Patterns Found:
1. **Off-by-one errors** - Very common in sequence processing (Bug #3)
2. **Inverted conditions** - Boolean logic mistakes (Bugs #4, #8)
3. **Missing normalization** - Forgetting to scale values (Bug #1)
4. **Wrong parameters** - Values off by factors (Bug #2)
5. **Resource leaks** - Not cleaning up properly (Bug #9)
6. **Incorrect math** - Formula errors (Bugs #6, #11)
7. **State management** - Tracking positions/indices wrong (Bug #7)

### Debugging Strategies That Work:
- **Check symptoms first** - What's the observable behavior?
- **Look for comments** - Bug locations are marked
- **Verify shapes** - Print tensor shapes when confused
- **Test edge cases** - Long sequences, large batches, etc.
- **Compare with reference** - Check master branch if stuck
- **Use visualization** - Plot values to spot anomalies

### Production Implications:
These bugs represent real issues that can occur in production LLM systems:
- Training instability (Bugs #1, #2)
- Poor model quality (Bugs #3, #4, #6, #8)
- Runtime crashes (Bugs #5, #7, #9)
- Incorrect outputs (Bugs #10, #11, #12)

Understanding how to find and fix these issues is crucial for building reliable LLM applications.

---

## 🏆 Congratulations!

If you found all 12 bugs, you've demonstrated:
- Deep understanding of transformer architecture
- Strong PyTorch debugging skills
- Attention to detail in complex systems
- Ability to reason about training dynamics
- Knowledge of modern LLM optimizations

These skills are essential for working on production LLM systems!

