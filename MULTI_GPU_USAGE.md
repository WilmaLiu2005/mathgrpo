# 多 GPU 使用说明

## 配置方法

在 `config.yaml` 中添加以下配置：

```yaml
model:
  pretrained_model_path: "/path/to/model"
  device: "cuda:0"              # 主设备（当 use_multi_gpu=true 时，这是第一个 GPU）
  dtype: "bfloat16"
  use_multi_gpu: true            # 是否使用多 GPU 训练（true=启用，false=单 GPU）
  device_ids: [0, 1, 2, 3]        # 使用的 GPU 设备 ID 列表（仅在 use_multi_gpu=true 时生效）
                                   # 如果不指定，将使用所有可用 GPU
```

## 使用示例

### 示例 1：使用 4 张 GPU（GPU 0-3）

```yaml
model:
  pretrained_model_path: "/home/student008/GRPO-zero/Qwen2.5-3B-base/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
  device: "cuda:0"
  dtype: "bfloat16"
  use_multi_gpu: true
  device_ids: [0, 1, 2, 3]
```

### 示例 2：使用所有可用 GPU

```yaml
model:
  pretrained_model_path: "/path/to/model"
  device: "cuda:0"
  dtype: "bfloat16"
  use_multi_gpu: true
  device_ids: null  # 或者不写这一行，会自动使用所有 GPU
```

### 示例 3：单 GPU 模式（默认）

```yaml
model:
  pretrained_model_path: "/path/to/model"
  device: "cuda:4"
  dtype: "bfloat16"
  use_multi_gpu: false  # 或者不写这一行
```

## 工作原理

1. **DataParallel 模式**：使用 PyTorch 的 `torch.nn.DataParallel` 实现多 GPU 训练
2. **自动分配**：在 `update_policy` 阶段，batch 会自动分配到多个 GPU 上并行计算
3. **Rollout 阶段**：rollout 阶段仍然在单 GPU 上运行（因为生成过程是串行的）

## 注意事项

1. **内存使用**：多 GPU 训练会复制模型到每个 GPU，但 batch 会被分割，所以总内存使用量会减少
2. **性能**：多 GPU 训练可以加速 `update_policy` 阶段，但 rollout 阶段仍然是单 GPU
3. **batch_size**：可以适当增大 `batch_size` 来充分利用多 GPU
4. **设备选择**：确保指定的 `device_ids` 中的 GPU 都是可用的

## 解决 OOM 问题

如果遇到 OOM 错误，可以：

1. **减少 batch_size**：
```yaml
training:
  batch_size: 64  # 从 128 减少到 64
  num_questions_per_batch: 16  # 相应减少
```

2. **减少 micro_batch_size**：
```yaml
training:
  micro_batch_size: 4  # 从 8 减少到 4
```

3. **减少 max_gen_len**：
```yaml
training:
  max_gen_len: 512  # 从 1024 减少到 512
```

4. **启用内存高效的优化器**：
```yaml
training:
  memory_efficient_adamw: true
```

5. **使用多 GPU**：多 GPU 可以分担内存压力
```yaml
model:
  use_multi_gpu: true
  device_ids: [0, 1, 2, 3]
```

