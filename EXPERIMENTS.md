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

## E0 原始 benchmark 复现（`run_benchmark.py`，未修改）

`logs/benchmark_orig.log`，2026-10-13。计时为 `generate()` 端到端（prefill + 32 步 decode + 采样），无 warmup，单次运行。

| input_len=256, output_len=32 | bs=1 | bs=8 | bs=16 |
|---|---:|---:|---:|
| FP16 runtime (s) | 1.10 | 0.68 | 0.72 |
| FP16 peak mem (MB) | 3072 | 4495 | 6134 |
| INT4 runtime (s) | 1.55 | 1.67 | 1.78 |
| INT4 peak mem (MB) | 4540 | 5508 | 6621 |

| bs=1, in=256, out=32 | Dense | MoE-slice | MoE-LoRA |
|---|---:|---:|---:|
| runtime (s) | 0.56 | 2.71 | 2.83 |
| peak mem (MB) | 3074 | 3075 | 3082 |

观察：
- **FP16 bs=1 (1.10 s) 比 bs=8 (0.68 s) 还慢**；同一配置在 Phase 3 的 Dense 行只要 0.56 s。差别只在于 FP16 bs=1 是整个进程的第一次 `generate()` → 首轮额外 ~0.54 s 是 warmup 开销（见 P1）。
- 峰值显存与旧报告 P100 上的数字**逐 MB 一致**：显存只由张量形状决定，与硬件无关。
- 去掉 warmup 后 FP16 bs=1/8/16 ≈ 0.56/0.68/0.72 s，bs 增大 16× 时间只多 ~30%。
- bs=1 FP16 约 0.56 s / 32 ≈ 17 ms/step（含 prefill 摊入），而权重 ~2.5 GB / 1555 GB/s ≈ 1.6 ms → 离带宽下限差一个数量级，原因待 profiler 验证（E4/E5）。
- INT4 比 FP16 慢 2.5–2.8×；MoE 比 Dense 慢 ~5×。

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

### P1 FP16 bs=1 比 bs=8 还慢

- **现象**：原始 benchmark 中 FP16 bs=1 = 1.10 s，bs=8 = 0.68 s；同配置的 Dense 行（后跑）= 0.56 s。
- **最初假设**：
- **验证实验**：
- **结论**：
- **学到什么**：

### P2 数据下载失败（环境）

- **现象**：`load_dataset("tatsu-lab/alpaca")` 报 `TypeError: process() takes no keyword arguments`。
- **定位**：traceback 最底层是 httpx2 解压响应；镜像自带 Brotli 1.0.9 不支持新参数。
- **解决**：`pip install -U brotli`（→ 1.2.0）。
- **学到什么**：看 traceback 最底层的库，而不是最上层的调用。

### P3 RoPE 缩放缺失

- **现象**：`ckpt/params.json` 中 `use_scaled_rope: true`，`llama/model.py` 未实现 Llama 3.1+ 的 `apply_scaling`。
- **影响**：短序列（≤320）下低频分量转角很小，输出仍通顺；但 dense baseline 与官方模型不完全一致。
- **待办**（改进阶段）：实现后对比 PPL 变化。
