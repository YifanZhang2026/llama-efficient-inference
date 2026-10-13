# Experiment Log

每条实验先写**预测**，再测，再解释。所有数字都附上命令和 commit，保证可复现。

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB（HBM2 ~1555 GB/s，FP16 tensor core ~312 TFLOPS） |
| Driver / CUDA | 595.71.05 / 13.2（torch 自带 CUDA 12.8） |
| PyTorch | 2.8.0+cu128 |
| 模型 | Llama 3.2-1B（Meta `original/consolidated.00.pth`） |
| 数据 | `prepare_data.py` 生成的 alpaca_500.json（前 200 训练 / 200–299 评估）、mt_bench_prompts.json |

## 测量方法约定

> 把最终采用的计时方法写在这里，后面所有表格都按这个口径测。

- warmup：
- prefill / decode 拆分方式：
- 采样设置（temperature / 是否固定输出长度）：
- 重复次数与统计量（均值 / 中位数）：
- 峰值显存口径（`max_memory_allocated` 何时 reset）：

---

## Part 1：INT4 量化

### E1 模型大小

| | 预测 (MB) | 实测 (MB) |
|---|---:|---:|
| FP16 | | |
| INT4（params + buffers） | | |
| 减少比例 | | |

推导：

### E2 FP16 vs INT4 性能（input_len=256, output_len=32）

| | bs=1 | bs=8 | bs=16 |
|---|---:|---:|---:|
| FP16 peak mem (MB) | | | |
| FP16 prefill (ms) | | | |
| FP16 decode (ms/token) | | | |
| INT4 peak mem (MB) | | | |
| INT4 prefill (ms) | | | |
| INT4 decode (ms/token) | | | |

预测：

观察：

### E3 INT4 峰值显存来源

- 预测：
- 测量方法：
- 结果（哪一层 / 哪个 tensor / 多大）：
- 对照实验：

| 改动 | bs=1 peak mem (MB) |
|---|---:|
| 原始实现 | |
| | |
| | |

### E4 INT4 时间花在哪（profiler）

| kernel / op | 时间占比 | 备注 |
|---|---:|---|
| | | |

结论：

### E5 Roofline：decode 离带宽上限多远

| | 每 token 读取的权重字节 | 理论下限 (ms/token) | 实测 (ms/token) | 带宽利用率 |
|---|---:|---:|---:|---:|
| FP16 bs=1 | | | | |
| INT4 bs=1 | | | | |

---

## Part 2：MoE

### E6 参数、显存与速度（N=4, K=2, LoRA r=8）

| | Dense | MoE-slice | MoE-LoRA |
|---|---:|---:|---:|
| Total params | | | |
| Trainable params | | | |
| Peak mem (MB) | | | |
| 推理时间 (s)，bs=1, in=256, out=32 | | | |

### E7 训练前的 PPL（刚转换完）

| | Dense | MoE-slice（未训练） | MoE-LoRA（未训练） |
|---|---:|---:|---:|
| 预测 PPL | | | |
| 实测 PPL | | | |

推导 / 解释：

### E8 训练与训练后评估

| | Epoch 1 loss | Epoch 2 loss | PPL | Next-token acc |
|---|---:|---:|---:|---:|
| Dense | – | – | | |
| Slice | | | | |
| LoRA | | | | |

loss 曲线图：

### E9 LoRA 有没有在学

| step | router grad norm | `lora_B` weight norm | loss |
|---:|---:|---:|---:|
| | | | |

- B=0 时 router 梯度的预测：
- PPL 差异是否超过噪声（不同 seed / eval 子集）：

### E10 MoE 慢在哪

- 预测同步次数（prefill 256 tokens）：
- profiler 实测同步次数 / 占时：
- 结论：

### E11 Expert 负载分布

图：

观察：

---

## 问题记录

> 每个问题一条。面试讲"遇到了什么问题"就照这个讲。

### P1 标题

- **现象**（带数字）：
- **最初假设**：
- **验证实验**：
- **结论**（带数字）：
- **假设错在哪 / 学到什么**：
