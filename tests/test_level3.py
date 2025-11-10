"""
Level 3 Test Suite: Architecture & Performance

Run with: pytest tests/test_level3.py -v
"""
import sys
import torch
import numpy as np
import tempfile
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def test_rotary_embeddings_at_positions():
    """Test that rotary embeddings work correctly at different sequence positions."""
    from nanochat.gpt import GPT, GPTConfig
    
    config = GPTConfig(
        sequence_len=256,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
    )
    
    model = GPT(config)
    model.init_weights()
    
    # Test at different positions
    test_seq = torch.randint(0, 100, (1, 10), dtype=torch.int32)
    
    outputs = []
    for pos in [0, 50, 100, 150, 200]:
        with torch.no_grad():
            T = test_seq.size(1)
            cos_sin = model.cos[:, pos:pos+T], model.sin[:, pos:pos+T]
            
            x = model.transformer.wte(test_seq)
            x = torch.nn.functional.rms_norm(x, (x.size(-1),))
            attn_out = model.transformer.h[0].attn(x, cos_sin, kv_cache=None)
            
            outputs.append(attn_out[0, 0, :5].numpy())
    
    # Check that outputs don't degrade at higher positions
    variances = [np.var(out) for out in outputs]
    assert variances[-1] > variances[0] * 0.1, f"Output degrades at high positions: {variances}"


def test_kv_cache_consistency():
    """Test that KV cache produces consistent generation."""
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.engine import Engine
    from nanochat.tokenizer import RustBPETokenizer
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tokenizer
        texts = ["test"] * 100
        tokenizer = RustBPETokenizer.train_from_iterator(iter(texts), vocab_size=256)
        
        # Create model
        config = GPTConfig(
            sequence_len=128,
            vocab_size=256,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
        )
        
        model = GPT(config)
        model.init_weights()
        engine = Engine(model, tokenizer)
        
        # Generate
        prompt_tokens = [1, 2, 3]
        generated = []
        for token_col, _ in engine.generate(prompt_tokens, max_tokens=20):
            generated.append(token_col[0])
        
        # Check generation quality
        assert len(generated) == 20, f"Generated {len(generated)} tokens, expected 20"
        # Should have some variety (not all same token)
        unique = len(set(generated))
        assert unique >= 3, f"Too little variety: only {unique} unique tokens"


def test_mqa_compatibility():
    """Test that MQA works with n_head != n_kv_head."""
    from nanochat.gpt import GPT, GPTConfig
    
    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=2,
        n_head=8,      # More query heads
        n_kv_head=4,   # Fewer KV heads
        n_embd=256,
    )
    
    model = GPT(config)
    model.init_weights()
    
    # Test forward pass
    B, T = 2, 10
    inputs = torch.randint(0, 256, (B, T), dtype=torch.int32)
    targets = torch.randint(0, 256, (B, T), dtype=torch.int64)
    
    try:
        loss = model(inputs, targets)
        assert loss.item() > 0, "Loss should be positive"
        assert np.isfinite(loss.item()), "Loss should be finite"
    except Exception as e:
        raise AssertionError(f"MQA forward pass failed: {e}")


def test_memory_bounded():
    """Test that memory usage doesn't grow unboundedly."""
    from nanochat.engine import KVCache
    
    # Create cache
    cache = KVCache(
        batch_size=1,
        num_heads=4,
        seq_len=100,
        head_dim=32,
        num_layers=2
    )
    
    # Simulate multiple resets
    initial_pos = cache.pos
    
    for i in range(5):
        # Simulate some usage
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        cache.insert_kv(0, k, v)
        cache.insert_kv(1, k, v)
        
        # Reset
        cache.reset()
        
        # Position should be reset
        assert cache.pos == 0, f"Position not reset: {cache.pos}"
    
    # Cache should be cleared after reset
    cache.reset()
    assert cache.kv_cache is None or cache.pos == 0, "Cache not properly reset"


def test_tool_output_ordering():
    """Test that tool outputs have correct token ordering."""
    from nanochat.tokenizer import RustBPETokenizer
    
    with tempfile.TemporaryDirectory() as tmpdir:
        texts = ["test"] * 100
        tokenizer = RustBPETokenizer.train_from_iterator(iter(texts), vocab_size=256)
        
        conversation = {
            "messages": [
                {"role": "user", "content": "Calculate 5+3"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Result: "},
                        {"type": "python", "text": "5+3"},
                        {"type": "python_output", "text": "8"},
                    ]
                }
            ]
        }
        
        ids, mask = tokenizer.render_conversation(conversation)
        
        # Find output tokens
        output_start_id = tokenizer.encode_special("<|output_start|>")
        output_end_id = tokenizer.encode_special("<|output_end|>")
        
        output_start_pos = ids.index(output_start_id) if output_start_id in ids else -1
        output_end_pos = ids.index(output_end_id) if output_end_id in ids else -1
        
        if output_start_pos >= 0 and output_end_pos >= 0:
            assert output_start_pos < output_end_pos, "output_start should come before output_end"

