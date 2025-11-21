# GRPO:Zero

GRPO training with minimal dependencies (and low GPU memory usage!). We implement almost everything from scratch and only depend on `tokenizers` for tokenization and `pytorch` for training.

* No `transformers` and `vLLM` dependencies!
* The default config is set to run on a single A40 GPU (48GB VRAM) for a few hours to get good results. (An A40 costs `$0.44` per hour if you rent it from RunPod.)
* We also support training with a 24GB VRAM GPU (e.g., an RTX 4090 GPU) by offloading the optimizer to CPU. Fortunately, this only adds a small overhead to the training because we only update the policy network a few hundred times during the entire training process.
* We support several improvements over the original GRPO algorithm from the [DAPO project](https://arxiv.org/abs/2503.14476), including:

  * **Token-level policy gradient loss**: every token is equally weighted in the policy gradient loss.
  * **Removing KL Divergence**: the KL divergence is not used in the policy gradient loss. This reduces GPU memory usage as we no longer need the reference policy network.
  * **Overlong episode filtering**: skips unfinished episodes that exceed context length limits. This stabilizes training. Though disabled by default to observe model learning under limited context length. Set `skip_unfinished_episodes` to `true` to enable it.

## Algorithm 

Group Relative Policy Optimization (GRPO) is an algorithm proposed by Deepseek for training large language models with reinforcement learning. The idea is simple: for each question, we randomly sample multiple answers. The advantage of an answer is then defined as the normalized reward. This gets rid of the value estimation network. In particular, we implement the following algorithm:

1. For each training step, randomly sample $N$ questions $q_1, q_2, \cdots, q_N$.
2. For each question $q_i$, sample $M$ answers $a_{i,1}, a_{i,2}, \cdots, a_{i,M}$.
3. Compute the reward $r_{i,j}$ for each answer $a_{i,j}$.
4. Compute the mean and std of the rewards for each question $q_i$.

$$
\begin{aligned}
\mu_i &\leftarrow \text{mean}(r_{i,1}, r_{i,2}, \cdots, r_{i,M}) \\
\sigma_i &\leftarrow \text{std}(r_{i,1}, r_{i,2}, \cdots, r_{i,M})
\end{aligned}
$$

5. For each token $t$ in the answer $a_{i,j}$, compute the advantage as

$$A_{i,j}[t] \leftarrow \frac{r_{i,j} - \mu_i}{\sigma_i}$$

6. Compute policy gradient using PPO surrogate objective. For simplicity, we will only do one policy update per iteration, in which the gradient of the PPO objective is equivalent to following vanilla policy gradient estimation (per token).

$$
\nabla_\theta \log \pi_\theta(a_{i,j}[t]) \cdot A_{i,j}[t]
$$

7. Update the policy network $\pi(\theta)$ using the gradient. Go back to step 1.

## GSM8K Task

We train the Qwen2.5 models on the **GSM8K** dataset, a grade-school math word-problem dataset. Each question describes a short math scenario, and the goal is to produce the correct final numeric answer.

Example:

```
Question:  
A book costs 12 dollars and a pen costs 3 dollars. If Jenny buys 4 books and 2 pens, how much does she spend?

Answer:  (step-by-step reasoning...)  →  54
```

## Reward Function

To solve GSM8K, the model is trained (using GRPO) to generate chain-of-thought reasoning before giving the final numerical answer. The expected output format is:

```
<think>Model step-by-step reasoning</think>
<answer>Final numeric answer</answer>
```

The reward has two components:

1. **Format Reward**
   The model gets **0.1** reward if it produces the exact required XML-style format with `<think>`…`</think>` and `<answer>`…`</answer>` tags. Otherwise 0.

2. **Answer Reward**
   The model receives a reward of **1** if:

   * the final answer inside `<answer>`…`</answer>` is a valid number
   * and matches the ground-truth GSM8K answer exactly

   Otherwise, the reward is **0**.


## Training

We use the `Qwen2.5-3B-Instruct` model for training. To train the model, run the following commands:

```bash
# 激活你的 conda 环境（假设你已经有了合适的环境）
# conda activate your_env_name

# 安装 git-lfs
(apt update; apt install git-lfs -y; )git lfs install

pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip -r requirements.txt

# 下载数据集和预训练模型
python download.py

# 训练模型
python train.py

```
## Acknowledgements

This project builds upon the work of several outstanding projects:

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) for pioneering the GRPO algorithm.
- [DAPO](https://arxiv.org/abs/2503.14476) for their enhancements to the original GRPO algorithm.
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) for their implementation of GRPO and creation of the [CountDown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) dataset.
- [nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment/tree/main) for their clear implementation and tutorial on the GRPO algorithm.
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) for developing the high-quality pretrained model used in this project.