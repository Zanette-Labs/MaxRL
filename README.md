# Maximum Likelihood Reinforcement Learning (MaxRL)

This is the official code repository for the paper **"Maximum Likelihood Reinforcement Learning"**.

📄 [Paper](#) | 🌐 [Project Page](https://zanette-labs.github.io/MaxRL)

---

# How to Run

## Setup Environment

```bash
conda create -n maxrl python=3.10 -y
conda activate maxrl
bash scripts/install_vllm_sglang_mcore.sh
```

## SmolLM on GSM8k

1. Download and preprocess data

```bash
python examples/data_preprocess/standard.py --data_source guanning/gsm8k --no_test
python examples/data_preprocess/standard.py --data_source guanning-ai/gsm8k-platinum --no_train
```

2. Setup path configurations in `math/smollm.sh` 

3. `bash math/smollm.sh`

## 17x17 Maze

1. Download preprocessed data

```bash
huggingface-cli download guanning-ai/maze_17x17_1m --repo-type dataset --local-dir ./maze/data/
```

2. Setup path configurations in `maze/maze_17.sh`

3. `bash maze/maze_17.sh`
