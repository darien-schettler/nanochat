# 🐛 Nanochat Bug Hunt Challenge

Welcome to the Nanochat debugging challenge! This repository contains three progressively difficult bug-hunting exercises designed to test your understanding of LLM implementation details.

## 🎯 Overview

You'll work through three branches, each containing intentional bugs in the nanochat codebase. Each branch has an accompanying Jupyter notebook that will help you discover, understand, and fix the bugs.

**Total bugs to find: 12** (3 easy + 4 medium + 5 hard)

## 📋 Prerequisites

- Python 3.10 or higher
- Basic understanding of PyTorch
- Familiarity with transformers (helpful but not required)
- A sense of adventure! 🚀

## 🚀 Quick Start

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/darien-schettler/nanochat.git
cd nanochat

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --extra cpu  # Or --extra gpu if you have CUDA

# Install Rust (needed for tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build the Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/darien-schettler/nanochat.git
cd nanochat

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Rust (needed for tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build the Rust tokenizer
pip install maturin
maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## 📚 Challenge Structure

### Level 1: Easy - Tokenizer & Data Loading
**Branch:** `bug-hunt-easy`  
**Notebook:** `tokenizer_debug.ipynb`  
**Bugs:** 3  
**Estimated time:** 30-45 minutes

Topics covered:
- BPE tokenization
- Special token handling
- PyTorch tensor dtypes
- Autoregressive training data preparation

```bash
git checkout bug-hunt-easy
jupyter notebook tokenizer_debug.ipynb
```

### Level 2: Medium - Training Pipeline
**Branch:** `bug-hunt-medium`  
**Notebook:** `training_pipeline_debug.ipynb`  
**Bugs:** 4  
**Estimated time:** 1-2 hours

Topics covered:
- Gradient accumulation
- Learning rate scheduling
- Optimizer configuration
- Supervised fine-tuning masking

```bash
git checkout bug-hunt-medium
jupyter notebook training_pipeline_debug.ipynb
```

### Level 3: Hard - Architecture & Performance
**Branch:** `bug-hunt-hard`  
**Notebook:** `architecture_challenges.ipynb`  
**Bugs:** 5  
**Estimated time:** 2-3 hours

Topics covered:
- Rotary positional embeddings
- KV cache management
- Multi-Query Attention (MQA)
- Memory optimization
- Tool use integration

```bash
git checkout bug-hunt-hard
jupyter notebook architecture_challenges.ipynb
```

## 🎓 Learning Objectives

By completing this challenge, you will:

- Understand the full LLM training pipeline from tokenization to inference
- Learn about modern transformer optimizations (RoPE, MQA, KV caching)
- Practice debugging complex PyTorch code
- Gain insight into common pitfalls in LLM implementation
- Experience real-world bugs that can occur in production systems

## 💡 Tips

1. **Read the notebooks carefully** - They contain hints and diagnostic code
2. **Use the visualization tools** - The notebooks include helpers to inspect data
3. **Check the symptoms** - Each bug has observable wrong behavior
4. **Look at the comments** - Bug locations are marked with "BUG" comments
5. **Test your fixes** - The notebooks include verification cells
6. **Don't skip levels** - Each level builds on concepts from the previous one

## 🔍 What to Look For

### Common Bug Patterns

- **Off-by-one errors** - Very common in sequence processing
- **Shape mismatches** - Especially with attention mechanisms
- **Incorrect normalization** - In gradients, losses, or learning rates
- **Inverted logic** - Boolean conditions that should be flipped
- **Missing operations** - Commented out or forgotten code
- **Wrong parameters** - Values off by factors of 2, 10, 100, etc.

### Files to Focus On

- `nanochat/tokenizer.py` - Tokenization logic
- `nanochat/dataloader.py` - Data loading and batching
- `nanochat/gpt.py` - Model architecture
- `nanochat/engine.py` - Inference engine with KV cache
- `scripts/base_train.py` - Base model training
- `scripts/chat_sft.py` - Supervised fine-tuning

## 🏆 Completion

Once you've fixed all bugs in a level:

1. The notebook will confirm your fixes with ✅ messages
2. You can compare your fixes to the clean `master` branch
3. Move on to the next level!

## 📝 Notes

- **No GPU required** - All notebooks can run on CPU (though slower)
- **Small models** - We use tiny models for quick iteration
- **Self-contained** - Each notebook sets up its own minimal environment
- **Educational focus** - Bugs are chosen for learning value, not difficulty

## 🤔 Stuck?

If you're stuck on a bug:

1. Re-read the notebook hints carefully
2. Look for "BUG" comments in the code
3. Check the symptom description - what's the observable behavior?
4. Try printing tensor shapes and values
5. Use the notebook's diagnostic cells
6. Remember: The satisfaction of finding the bug is worth the struggle!

**Need solutions?** See `SOLUTIONS.md` (but try not to peek too early!)

## 🎉 After Completion

Congratulations on completing the challenge! You now have hands-on experience with:

- Modern LLM architecture implementation
- PyTorch optimization techniques
- Debugging complex machine learning systems
- Production-grade code quality considerations

Consider exploring the full nanochat codebase to see how these concepts scale to real training runs!

## 📚 Additional Resources

- [Nanochat Main README](README.md) - Full project documentation
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary position embeddings
- [GQA Paper](https://arxiv.org/abs/2305.13245) - Group Query Attention

## 🙏 Acknowledgments

This challenge is built on top of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy, which itself builds on [nanoGPT](https://github.com/karpathy/nanoGPT).

---

**Good luck, and happy debugging!** 🐛🔍
