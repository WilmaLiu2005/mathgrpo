# 配置文件超参数设置汇总

本文档汇总了所有 YAML 配置文件的超参数设置情况，便于对比和选择。

## 配置文件列表

1. `config.yaml` - 基础配置
2. `config_1.yaml` - 长度分组配置
3. `config_kl.yaml` - KL 惩罚配置（已废弃）
4. `config_dynamic_length_grouping.yaml` - 动态长度分组（稳定版）
5. `config_dynamic_length_grouping_new.yaml` - 动态长度分组（新版本）
6. `config_multiple.yaml` - Prefix 训练 + 动态裁剪
7. `config_deepseek_prefix_0_2.yaml` - Prefix 训练（use_prob=0.2）
8. `config_deepseek_prefix_0_5.yaml` - Prefix 训练（use_prob=0.5）
9. `config_deepseek_prefix_0_5_5epoch.yaml` - Prefix 训练（use_prob=0.5, 5 epochs）
10. `config_deepseek_prefix_0_8.yaml` - Prefix 训练（use_prob=0.8/1.0）

---

## 超参数对比表

### 基础训练参数

| 配置文件名 | Device | Batch Size | Micro Batch | LR | Max Grad Norm | Weight Decay | Epochs | Temperature |
|-----------|--------|------------|-------------|-----|---------------|--------------|--------|-------------|
| config.yaml | cuda:4 | 128 | 8 | 1.0e-5 | 1.0 | 0.0 | 2 | 0.9 |
| config_1.yaml | cuda:1 | 256 | 2 | 1.0e-5 | 1.0 | 0.0 | 2 | 1.2 |
| config_kl.yaml | cuda:1 | 128 | 8 | 1.0e-5 | 1.0 | 0.0 | 3 | 0.9 |
| config_dynamic_length_grouping.yaml | cuda:5 | 128 | 8 | 1.0e-5 | 1.0 | 0.0 | 2 | 0.9 |
| config_dynamic_length_grouping_new.yaml | cuda:6 | 128 | 8 | 1.0e-5 | 1.0 | 0.0 | 2 | 0.9 |
| config_multiple.yaml | cuda:0 | 64 | 4 | 1.0e-5 | 1.0 | 0.0 | 3 | 0.7 |
| config_deepseek_prefix_0_2.yaml | cuda:3 | 128 | 8 | 1.0e-5 | 1.0 | 0.0 | 3 | 0.9 |
| config_deepseek_prefix_0_5.yaml | cuda:0 | 128 | 8 | 1.0e-5 | 1.0 | 0.0 | 3 | 0.9 |
| config_deepseek_prefix_0_5_5epoch.yaml | cuda:1 | 128 | 8 | 1.0e-5 | 1.0 | 0.0 | **5** | 0.9 |
| config_deepseek_prefix_0_8.yaml | cuda:3 | 128 | 8 | 1.0e-5 | 1.0 | 0.0 | 2 | 0.9 |

### 数据相关参数

| 配置文件名 | Num Questions/Batch | Max Prompt Len | Max Gen Len | Test Size |
|-----------|---------------------|----------------|-------------|-----------|
| config.yaml | 4 | 512 | 1024 | 128 |
| config_1.yaml | 32 | 512 | 1024 | 256 |
| config_kl.yaml | 32 | 512 | 1024 | 128 |
| config_dynamic_length_grouping.yaml | 4 | 512 | 1024 | 128 |
| config_dynamic_length_grouping_new.yaml | 4 | 512 | 1024 | 128 |
| config_multiple.yaml | 16 | 512 | **512** | 128 |
| config_deepseek_prefix_0_2.yaml | 32 | 512 | 1024 | 128 |
| config_deepseek_prefix_0_5.yaml | 32 | 512 | 1024 | 128 |
| config_deepseek_prefix_0_5_5epoch.yaml | 32 | 512 | 1024 | 128 |
| config_deepseek_prefix_0_8.yaml | 4 | 512 | 1024 | 128 |

### GRPO 算法参数

| 配置文件名 | Dynamic Clipping | Epsilon Low | Epsilon High | Clip Ratio | Use KL Penalty | KL Coeff | Length Grouping |
|-----------|------------------|-------------|-------------|------------|----------------|----------|-----------------|
| config.yaml | ❌ | 0.3 | 0.3 | 0.2 | ❌ | 0.02 | ❌ |
| config_1.yaml | ❌ | 0.3 | 0.3 | 0.2 | ❌ | 0.05 | ✅ |
| config_kl.yaml | ❌ | 0.3 | 0.3 | 0.2 | ❌ | 0.02 | ❌ |
| config_dynamic_length_grouping.yaml | ❌ | 0.3 | 0.3 | 0.2 | ❌ | 0.02 | ✅ |
| config_dynamic_length_grouping_new.yaml | ❌ | 0.3 | 0.3 | 0.2 | ❌ | 0.02 | ✅ |
| config_multiple.yaml | ✅ | **0.2** | **0.4** | 0.2 | ❌ | 0.01 | ✅ |
| config_deepseek_prefix_0_2.yaml | ❌ | 0.3 | 0.3 | 0.2 | ❌ | 0.02 | ❌ |
| config_deepseek_prefix_0_5.yaml | ❌ | 0.3 | 0.3 | 0.2 | ❌ | 0.02 | ❌ |
| config_deepseek_prefix_0_5_5epoch.yaml | ❌ | 0.3 | 0.3 | 0.2 | ❌ | 0.02 | ❌ |
| config_deepseek_prefix_0_8.yaml | ❌ | 0.3 | 0.3 | 0.2 | ❌ | 0.02 | ❌ |

### Prefix 训练参数

| 配置文件名 | Enable Prefix | Prefix Use Prob | Prefix Dropout Prob | Prefix SFT Coeff | Memory Efficient AdamW |
|-----------|---------------|-----------------|---------------------|------------------|------------------------|
| config.yaml | ❌ | - | 1.0 | 0.2 | ❌ |
| config_1.yaml | ❌ | - | - | - | ❌ |
| config_kl.yaml | ❌ | - | 1.0 | 0.2 | ❌ |
| config_dynamic_length_grouping.yaml | ❌ | - | 1.0 | 0.2 | ❌ |
| config_dynamic_length_grouping_new.yaml | ❌ | - | 1.0 | 0.2 | ❌ |
| config_multiple.yaml | ✅ | **0.8** | 1.0 | 0.2 | ✅ |
| config_deepseek_prefix_0_2.yaml | ✅ | **0.2** | 1.0 | 0.2 | ❌ |
| config_deepseek_prefix_0_5.yaml | ✅ | **0.5** | 1.0 | 0.2 | ❌ |
| config_deepseek_prefix_0_5_5epoch.yaml | ✅ | **0.5** | 1.0 | 0.2 | ❌ |
| config_deepseek_prefix_0_8.yaml | ✅ | **1.0** | 1.0 | 0.2 | ❌ |

### 评估和保存参数

| 配置文件名 | Eval Interval | Ckpt Save Interval | Wandb Run Name |
|-----------|---------------|-------------------|----------------|
| config.yaml | 50 | 100 | final_baseline_0101 |
| config_1.yaml | 10 | 100 | dynamic_length_grouping_1221 |
| config_kl.yaml | 10 | 100 | final_baseline_1226 |
| config_dynamic_length_grouping.yaml | 50 | 100 | dynamic_length_grouping_without_kl_0101 |
| config_dynamic_length_grouping_new.yaml | 50 | 100 | dynamic_length_new_group_0102 |
| config_multiple.yaml | 10 | 100 | multiple_wo_kl_1231 |
| config_deepseek_prefix_0_2.yaml | 10 | 100 | deepseek_prefix_1228_prefix_use_prob_0_2 |
| config_deepseek_prefix_0_5.yaml | 10 | 100 | deepseek_prefix_1228_prefix_use_prob_0_5 |
| config_deepseek_prefix_0_5_5epoch.yaml | 10 | 100 | deepseek_prefix_1230_prefix_use_prob_0_5_5epoch |
| config_deepseek_prefix_0_8.yaml | 50 | 100 | deepseek_prefix_0101 |

---

## 配置分类

### 1. 基础配置（无特殊功能）

- **config.yaml**: 标准 GRPO 配置，无 prefix，无长度分组，固定裁剪
- **config_kl.yaml**: 与 config.yaml 相同，可能是早期实验配置

### 2. 长度分组配置

- **config_1.yaml**: 
  - 启用长度分组 (`use_length_grouping: true`)
  - 更大的 batch size (256) 和问题数 (32)
  - 更高的采样温度 (1.2)
  - 更大的 test size (256)

- **config_dynamic_length_grouping.yaml**: 
  - 启用长度分组
  - 标准 batch size (128)
  - 较低的温度 (0.9)

- **config_dynamic_length_grouping_new.yaml**: 
  - 与 config_dynamic_length_grouping.yaml 相同
  - 可能是新实验版本

### 3. Prefix 训练配置

所有 `config_deepseek_prefix_*` 文件都启用 prefix 训练，主要区别在于 `prefix_use_prob`：

- **config_deepseek_prefix_0_2.yaml**: `prefix_use_prob=0.2` (20% 使用 prefix)
- **config_deepseek_prefix_0_5.yaml**: `prefix_use_prob=0.5` (50% 使用 prefix)
- **config_deepseek_prefix_0_5_5epoch.yaml**: `prefix_use_prob=0.5`, 5 epochs
- **config_deepseek_prefix_0_8.yaml**: `prefix_use_prob=1.0` (100% 使用 prefix)

### 4. 混合配置

- **config_multiple.yaml**: 
  - 启用 prefix 训练 (`prefix_use_prob=0.8`)
  - 启用动态裁剪 (`use_dynamic_clipping: true`)
  - 启用长度分组
  - **内存优化配置**（batch_size=64, micro_batch_size=4, max_gen_len=512）
  - 启用内存高效 AdamW
  - 较低的温度 (0.7)

---

## 关键差异总结

### 内存优化配置

**config_multiple.yaml** 是唯一的内存优化配置：
- `batch_size: 64` (其他为 128 或 256)
- `micro_batch_size: 4` (其他为 8 或 2)
- `max_gen_len: 512` (其他为 1024)
- `memory_efficient_adamw: true` (其他为 false)
- `temperature: 0.7` (其他多为 0.9)

### 动态裁剪配置

只有 **config_multiple.yaml** 使用动态裁剪：
- `use_dynamic_clipping: true`
- `epsilon_low: 0.2`
- `epsilon_high: 0.4`

### 训练轮数

- 大部分配置：2-3 epochs
- **config_deepseek_prefix_0_5_5epoch.yaml**: 5 epochs

### Prefix 使用概率

- `0.2`: 20% 使用 prefix（更接近无 prefix 的分布）
- `0.5`: 50% 使用 prefix（平衡）
- `0.8`: 80% 使用 prefix（更接近全 prefix 的分布）
- `1.0`: 100% 使用 prefix（完全使用）

---

## 推荐使用场景

1. **基础实验**: `config.yaml`
2. **长度分组实验**: `config_dynamic_length_grouping.yaml`
3. **Prefix 训练（平衡）**: `config_deepseek_prefix_0_5.yaml`
4. **Prefix 训练（长期）**: `config_deepseek_prefix_0_5_5epoch.yaml`
5. **内存受限环境**: `config_multiple.yaml`
6. **动态裁剪实验**: `config_multiple.yaml`

---

## 注意事项

1. **内存使用**: Prefix 训练会显著增加内存使用，建议使用 `config_multiple.yaml` 的优化配置
2. **训练稳定性**: 较低的温度 (0.7) 和较小的 batch size 有助于提高稳定性
3. **评估频率**: 不同配置的 `eval_interval` 不同（10 或 50），影响评估频率
4. **设备分配**: 注意不同配置使用的 GPU 设备，避免冲突

