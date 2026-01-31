# Maximum Likelihood Reinforcement Learning (MaxRL)

This is the repository that contains the source code for the paper "Maximum Likelihood Reinforcement Learning".

## Overview

**Maximum likelihood** training is the foundational optimization principle behind modern machine learning. **Reinforcement learning**, by contrast, originated in optimal control and sequential decision-making where the end-to-end process is non-differentiable.

Many modern learning problems—navigation, program synthesis, structured prediction, and *multi-step reasoning in LLMs*—are non-differentiable but admit a **binary notion of correctness**. We introduce **Maximum Likelihood Reinforcement Learning (MaxRL)**, a framework that bridges the gap between RL and maximum likelihood for correctness-based tasks.

## Core Insight

For correctness-based tasks, a probabilistic modeling lens reveals that the probability of success defines a **likelihood over correct outcomes**. This leads to two different optimization objectives:

| Formulation | Gradient | Behavior |
|-------------|----------|----------|
| **RL (Control)** | $\nabla_\theta J_{\mathrm{RL}} = \mathbb{E}_x[\nabla_\theta p_\theta(x)]$ | Optimization dominated by easy examples |
| **ML (Modeling)** | $\nabla_\theta J_{\mathrm{ML}} = \mathbb{E}_x[\frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x)]$ | Concentrated effort on hard, uncertain inputs |

## Key Result: Maclaurin Expansion

The log-likelihood admits a **Maclaurin (failure-series) expansion**:

$$\log p = -\sum_{k=1}^{\infty}\frac{(1-p)^k}{k} = -\sum_{k=1}^{\infty}\frac{\mathrm{fail@}k}{k}$$

Differentiating yields:

$$\nabla_\theta J_{\mathrm{ML}}(x) = \sum_{k=1}^{\infty}\frac{1}{k}\,\nabla_\theta \mathrm{pass@}k(x)$$

**Maximum likelihood optimizes an infinite harmonic mixture of pass@k gradients.** Standard RL corresponds to retaining only the first-order term—*reinforcement learning is a first-order approximation of maximum likelihood in correctness space*.

## MaxRL Algorithm

MaxRL truncates the expansion at level $T = N$ (number of rollouts), yielding a simple gradient estimator:

```
for each prompt x in batch:
    Sample N rollouts from π_θ(·|x)
    K ← number of successful rollouts
    μ̂ ← K / N  # empirical pass rate
    
    if K > 0:
        A(y) = (r(y) - μ̂) / μ̂  # MaxRL advantage
    else:
        A(y) = 0

PolicyGradientUpdate(advantages)
```

**The key difference from REINFORCE: normalize by K (successful samples) instead of N (total samples).**

## Experiments

We evaluate MaxRL across four experimental settings:

1. **ImageNet**: Validation against exact maximum likelihood (cross-entropy)
2. **Maze Navigation**: Infinite data regime with procedurally generated mazes
3. **GSM8K**: Data-scarce regime with math reasoning
4. **Large-Scale LLM Training**: Qwen3-1.7B and Qwen3-4B on math benchmarks (AIME 2025, BeyondAIME, MATH-500, Minerva)

### Results

- MaxRL consistently **Pareto dominates GRPO**: higher pass@1 *and* improved pass@k
- Achieving the same pass@k requires **2.3× – 19.2× fewer samples** than GRPO
- MaxRL maintains higher training coverage throughout training

## Citation

If you find our paper useful for your work, please cite:

```bibtex
@article{tajwar2025maxrl,
  title={Maximum Likelihood Reinforcement Learning},
  author={Tajwar, Fahim and Zeng, Guanning and Zhou, Yueer and Song, Yuda and 
          Arora, Daman and Jiang, Yiding and Schneider, Jeff and 
          Salakhutdinov, Ruslan and Feng, Haiwen and Zanette, Andrea},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

Corresponding Authors: [Fahim Tajwar](mailto:ftajwar@andrew.cmu.edu), [Andrea Zanette](mailto:azanette@andrew.cmu.edu)

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Template adapted from [Nerfies](https://github.com/nerfies/nerfies.github.io).
