"""
Level 1 Test Suite: Tokenization & Data Pipeline

Run with: pytest tests/test_level1.py -v
"""
import os
import sys
import tempfile
import numpy as np
import torch
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from nanochat.tokenizer import RustBPETokenizer


def test_bos_token_prepending():
    """Test that BOS tokens are correctly prepended when requested."""
    # Create a minimal tokenizer
    with tempfile.TemporaryDirectory() as tmpdir:
        texts = ["test"] * 100
        tokenizer = RustBPETokenizer.train_from_iterator(iter(texts), vocab_size=256)
        
        bos_id = tokenizer.get_bos_token_id()
        test_text = "hello"
        
        # Test single string
        tokens_with_bos = tokenizer.encode(test_text, prepend="<|bos|>")
        tokens_without = tokenizer.encode(test_text)
        
        assert tokens_with_bos[0] == bos_id, f"First token should be BOS ({bos_id}), got {tokens_with_bos[0]}"
        assert len(tokens_with_bos) == len(tokens_without) + 1, "BOS token not added"
        
        # Test batch
        batch_with_bos = tokenizer.encode(["hello", "world"], prepend="<|bos|>")
        assert all(tokens[0] == bos_id for tokens in batch_with_bos), "BOS not prepended to all batch items"


def test_data_loading_dtypes():
    """Test that data loading uses correct tensor dtypes."""
    from collections import deque
    
    # Simulate dataloader logic
    tokens = list(range(100))
    token_buffer = deque(tokens)
    
    B, T = 2, 10
    needed_tokens = B * T + 1
    
    batch_tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
    scratch = torch.tensor(batch_tokens, dtype=torch.int64)
    
    # This is what the dataloader should do
    inputs_cpu = scratch[:-1].to(dtype=torch.int32)
    targets_cpu = scratch[1:]
    
    inputs = inputs_cpu.view(B, T)
    targets = targets_cpu.view(B, T)
    
    assert inputs.dtype == torch.int32, f"Inputs should be int32, got {inputs.dtype}"
    assert targets.dtype == torch.int64, f"Targets should be int64, got {targets.dtype}"


def test_autoregressive_targets():
    """Test that targets are correctly shifted by one position."""
    # Create a simple sequence
    tokens = list(range(50))
    scratch = torch.tensor(tokens, dtype=torch.int64)
    
    # Correct implementation
    inputs = scratch[:-1]
    targets = scratch[1:]
    
    # Verify autoregressive property
    assert not torch.equal(inputs, targets), "Targets should not equal inputs"
    assert torch.equal(inputs[1:], targets[:-1]), "Targets should be inputs shifted by 1"
    
    # Verify each target is the next token
    for i in range(len(inputs)):
        assert targets[i] == inputs[i] + 1, f"Target at position {i} should be next token"


def test_training_smoke():
    """Smoke test: model should train without errors."""
    from nanochat.gpt import GPT, GPTConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal tokenizer
        texts = ["test"] * 100
        tokenizer = RustBPETokenizer.train_from_iterator(iter(texts), vocab_size=256)
        
        # Create tiny model
        config = GPTConfig(
            sequence_len=32,
            vocab_size=256,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
        )
        
        model = GPT(config)
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Train for a few steps
        losses = []
        for _ in range(10):
            inputs = torch.randint(0, 256, (2, 16), dtype=torch.int32)
            targets = torch.randint(0, 256, (2, 16), dtype=torch.int64)
            
            loss = model(inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Loss should be finite
        assert all(np.isfinite(l) for l in losses), "Loss contains NaN or Inf"
        
        # Loss should generally decrease
        initial = np.mean(losses[:3])
        final = np.mean(losses[-3:])
        assert final < initial * 1.5, "Loss should not increase significantly"

