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
| Total params | 1,498,482,688 | 1,498,613,760 | 1,500,710,912 |
| Trainable params | 0 | 131,072 | 2,228,224 |
| Peak mem (MB) | 3074 | 3075 | 3082 |
| 推理时间 (s)，bs=1, in=256, out=32（E0） | 0.56 | 2.71 | 2.83 |
| 生成速度 (tok/s)，eval_moe，10 条短 prompt，out=32 | 53.5–55.1 | 23.7–31.8 | 28.7–28.9 |

观察：benchmark（prompt 256）里 MoE 比 Dense 慢 ~5×，eval_moe（短 prompt）里只慢 ~2×。

### E7 训练前的 PPL（刚转换完）

`eval_moe.py` 的 [2/3] 步就是转换后、训练前的评估。held-out = Alpaca[200:300]，共 **5711 tokens**。

| | Dense | MoE-slice（未训练） | MoE-LoRA（未训练） |
|---|---:|---:|---:|
| 预测 PPL | | | |
| 实测 PPL | 9.66 | **326,843.71** | 9.66 |
| 实测 next-token acc | 51.37% | **0.00%** | 51.37% |

观察：
- LoRA 未训练 = Dense，逐位一致 → zero-init identity 成立。
- slice 刚转换完 PPL 就是 32 万，**比均匀随机猜（PPL = vocab = 128,256）还差**，acc 0% → 模型在转换那一刻就坏了，不是"训练预算不够"。

推导 / 解释：

### E8 训练与训练后评估

`logs/train_*.log`、`logs/eval_*.log`。默认配置：200 条训练样本，2 epochs，SGD momentum 0.9，lr_router 1e-3 / lr_expert 1e-4，`torch.manual_seed(42)`。每 epoch ~225–235 s。

| | Epoch 1 loss | Epoch 2 loss | PPL | Next-token acc |
|---|---:|---:|---:|---:|
| Dense | – | – | 9.66 | 51.37% |
| Slice | 10.4605 | 8.2425 | 3430.71 | 4.08% |
| LoRA | 2.3773 | 2.3723 | 9.59 | 51.53% |

与旧报告（A40）对比：
- Dense acc 51.37% vs 51.30%：不同 GPU 上 FP16 数值差异。
- LoRA loss / PPL / acc **与旧报告完全一致**（2.3773 / 2.3723 / 9.59 / 51.53%）。
- Slice loss 不一致（旧 10.8595 / 8.3795，PPL 3414.03 / acc 4.57%）。
- slice 训练把 PPL 从 32 万降到 3431（降了ㄧ百倍），但仍离 dense 9.66 很远。
- LoRA acc +0.16 个百分点，在 5711 个 token 上 ≈ 多猜对 9 个 token —— 是否超过噪声见 E9。

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

`eval_moe.py`，训练后，10 条 MT-bench 风格 prompt，前 4 层：

| Layer | slice E0 | E1 | E2 | E3 | | LoRA E0 | E1 | E2 | E3 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 14% | 15% | 22% | 49% | | 25% | 21% | 34% | 21% |
| 1 | 50% | 34% | 7% | 9% | | 30% | 11% | 21% | 38% |
| 2 | 24% | 50% | 13% | 14% | | 17% | 33% | 20% | 30% |
| 3 | 22% | 50% | 14% | 14% | | 31% | 28% | 31% | 10% |

图：

观察：
- top-2 of 4 下每个 token 选两个**不同**的 expert，所以单个 expert 占比上限是 50%。slice 的 Layer 1–3 都有 expert 到了 50% —— 即**每个 token 都选了它**。
- LoRA 分布更分散，但也不均匀（10%–38%）。

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
