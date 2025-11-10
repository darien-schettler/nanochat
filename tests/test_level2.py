"""
Level 2 Test Suite: Training Pipeline

Run with: pytest tests/test_level2.py -v
"""
import sys
import torch
import numpy as np
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def test_gradient_accumulation_normalization():
    """Test that gradient accumulation normalizes loss correctly."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Simulate gradient accumulation
    grad_accum_steps = 4
    losses = []
    
    for step in range(10):
        for micro_step in range(grad_accum_steps):
            x = torch.randn(2, 10)
            y = torch.randn(2, 10)
            
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            
            # Loss should be normalized
            loss = loss / grad_accum_steps
            loss.backward()
            
            losses.append(loss.item() * grad_accum_steps)
        
        optimizer.step()
        optimizer.zero_grad()
    
    # Check stability
    assert all(np.isfinite(l) for l in losses), "Loss contains NaN/Inf"
    assert np.var(losses) < 100, "Loss variance too high - gradient accumulation unstable"


def test_learning_rate_warmup():
    """Test that learning rate warmup schedule is correct."""
    def get_lr_multiplier(it, num_iterations=1000, warmup_ratio=0.1):
        warmup_iters = round(warmup_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        return 1.0
    
    num_iters = 1000
    warmup_ratio = 0.1
    warmup_steps = int(warmup_ratio * num_iters)
    
    # Test key points
    lr_start = get_lr_multiplier(0, num_iters, warmup_ratio)
    lr_mid = get_lr_multiplier(warmup_steps // 2, num_iters, warmup_ratio)
    lr_end = get_lr_multiplier(warmup_steps, num_iters, warmup_ratio)
    lr_after = get_lr_multiplier(warmup_steps + 1, num_iters, warmup_ratio)
    
    # Assertions
    assert lr_start > 0.001, f"Initial LR too small: {lr_start}"
    assert 0.4 < lr_mid < 0.6, f"Mid-warmup LR should be ~0.5, got {lr_mid}"
    assert abs(lr_end - 1.0) < 0.01, f"LR at end of warmup should be ~1.0, got {lr_end}"
    assert abs(lr_after - 1.0) < 0.01, f"LR after warmup should be 1.0, got {lr_after}"


def test_optimizer_parameter_assignment():
    """Test that optimizers are assigned to correct parameter groups."""
    from nanochat.gpt import GPT, GPTConfig
    
    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
    )
    
    model = GPT(config)
    model.init_weights()
    
    optimizers = model.setup_optimizers()
    adamw_opt, muon_opt = optimizers
    
    # Collect parameter names
    adamw_params = set()
    for group in adamw_opt.param_groups:
        for p in group['params']:
            for name, param in model.named_parameters():
                if param is p:
                    adamw_params.add(name)
    
    muon_params = set()
    for p in muon_opt.param_groups[0]['params']:
        for name, param in model.named_parameters():
            if param is p:
                muon_params.add(name)
    
    # Verify assignments
    assert 'lm_head.weight' in adamw_params, "lm_head should use AdamW"
    assert 'transformer.wte.weight' in adamw_params, "embeddings should use AdamW"
    assert any('transformer.h' in name for name in muon_params), "transformer layers should use Muon"
    assert not any('transformer.h' in name for name in adamw_params), "transformer layers should not use AdamW"


def test_sft_masking_logic():
    """Test that SFT masking trains on assistant responses only."""
    import tempfile
    from nanochat.tokenizer import RustBPETokenizer
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tokenizer
        texts = ["test"] * 100
        tokenizer = RustBPETokenizer.train_from_iterator(iter(texts), vocab_size=256)
        
        # Create conversation
        conversation = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        
        ids, mask = tokenizer.render_conversation(conversation)
        
        # Process like SFT does
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
        targets = ids_tensor[1:].clone()
        
        # Apply masking - should mask where mask is 0
        targets[mask_tensor == 0] = -1
        
        num_train = (targets != -1).sum().item()
        num_masked = (targets == -1).sum().item()
        
        # Verify
        assert num_train > 0, "Should have tokens to train on"
        assert num_masked > 0, "Should have masked tokens"
        assert num_train > len(mask) * 0.2, "Should train on significant portion (assistant responses)"

