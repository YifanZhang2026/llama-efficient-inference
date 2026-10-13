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

工具：`bench.py`（Part 1 / Part 2 的性能数字都用它；`run_benchmark.py` 只作 E0 对照）。

- 输入：`torch.randint` 随机 token，不经 tokenizer；输出 = 1 次 prefill + (output_len−1) 步 decode，与 `generate(max_gen_len=output_len)` 工作量相同。
- 采样：GPU 上 greedy argmax，循环内无 host 同步；输出长度固定，不会因 EOS 提前停。
- warmup：默认 3 轮；第 1 轮的端到端时间记为 `cold_e2e_ms`（只有进程内第一个配置是真正的冷启动）。
- 计时：CUDA event 记录在 prefill 前后和每步 decode 之后；event 间隔包含 GPU 等待 CPU 发 kernel 的空闲时间。
- 统计：默认 10 轮；prefill / e2e 取中位数，decode 取所有步的中位数并报 p10 / p90。
- 峰值显存：warmup 后 `reset_peak_memory_stats()`，再取计时阶段的 `max_memory_allocated()`；`static_mb` = 建好（并量化 / 转换）模型后的常驻显存，含预分配 KV cache。
- 每个 bs 单独建一个模型，KV cache 按该 bs 预分配。
- 结果追加到 `results/bench.csv`，含 GPU、torch 版本和时间戳。
- profiling：`--profile` 只跑一轮带 NVTX 的 rep，配合 `nsys --capture-range=cudaProfilerApi` 只抓这一段。

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

`bench.py`，warmup 3，iters 10，2026-10-13。

| | bs=1 | bs=8 | bs=16 |
|---|---:|---:|---:|
| FP16 prefill (ms) | 17.71 | 40.91 | 87.06 |
| FP16 decode (ms/step, median) | 14.96 | 16.33 | 15.94 |
| FP16 e2e (ms) / cold | 482.8 / 982.6 | 529.5 / 561.1 | 562.4 / 567.0 |
| FP16 static / peak mem (MB) | 2869 / 3067 | 2941 / 4454 | 3027 / 6050 |
| INT4 prefill (ms) | 46.18 | 78.78 | 124.31 |
| INT4 decode (ms/step, median) | 40.46 | 43.53 | 45.59 |
| INT4 e2e (ms) / cold | 1300.6 / 1569.4 | 1422.4 / 1424.9 | 1532.8 / 1532.5 |
| INT4 static / peak mem (MB) | 1137 / 4536 | 1210 / 5476 | 1290 / 6558 |

预测：

观察（数字层面，解释待写）：
- **decode 几乎不随 bs 变**：FP16 15.0 → 16.3 → 15.9 ms，INT4 40.5 → 43.5 → 45.6 ms。
- **FP16 decode 15 ms/step，而只读权重的下限 ≈ 2.5 GB / 1555 GB/s ≈ 1.6 ms**，差 ~9×；且 bs=1 prefill（256 token）17.7 ms ≈ 1 步 decode 的耗时。
- **FP16 prefill 吞吐**：bs=16（4096 token）87 ms；按 2 × 1.5 B 参数 × 4096 ≈ 12.3 TFLOP 算 ≈ 141 TFLOPS，接近 GEMM 冒烟测试测到的 ~149 TFLOPS。bs=1 只有 ~43 TFLOPS。
- **INT4 每次 forward 多出一个近似固定的开销**：decode 多 ~25.5 ms，prefill 多 ~28 ms，与 token 数基本无关。
- **static**：INT4 1137 MB vs FP16 2869 MB（−60%）。bs 每 +1，static 涨 ~9.4 MB = 该序列的 KV cache。
- **peak − static**：FP16 198 / 1513 / 3023 MB ≈ 每条序列 ~190 MB；INT4 bs=1 是 3399 MB。
- **cold**：FP16 进程内首轮比稳态多 ~500 ms，与 E0 中的 0.54 s 吻合（P1）。
- `generate()` 端到端（E0 Dense 0.56 s）比 `bench.py` 的 483 ms 多 ~80 ms：top-p 采样、tokenizer 等 forward 以外的开销。

### E3 INT4 峰值显存来源

- 预测：
- 测量方法：`mem_snapshot.py`（PyTorch memory snapshot，warmup 后记录 1 次 prefill + 3 步 decode，回放 alloc/free 找峰值时刻仍存活的块；回放峰值与 `max_memory_allocated` 误差 ≤ 1 MB）。快照：`prof/mem_{int4_bs1,fp16_bs1,fp16_bs16}.pickle`。
- 结果：

| 配置 | static | peak | 峰值时多出的块（大小 · 分配位置） |
|---|---:|---:|---|
| INT4 bs=1 | 1145 | 4535 | 1002 MB · `quantize.py:101` `w_int4.float()`；1002 + 1002 MB · `quantize.py:102` `(w_int4 − zp)` 的临时结果和 `× scale` 的结果；125 + 125 MB · `quantize.py:95/96` `w_even` / `w_odd`；125 MB · `model.py:455` 上一步的 fp32 logits |
| FP16 bs=1 | 2877 | 3066 | 125 MB · `model.py:455` fp32 logits；63 MB · 同行 fp16 logits（`.float()` 之前） |
| FP16 bs=16 | 2998 | 6021 | 2004 MB + 1002 MB · `model.py:455` 同上，×16 |

- 1002 MB = 128256 × 2048 × 4 B，125 MB = 128256 × 1024 × 1 B → INT4 峰值完全来自**形状最大的那一层（lm_head）的 dequant 临时张量**：同时存活 3 份 fp32 全尺寸张量 + 2 份 uint8 半尺寸 nibble。不是"整个模型物化了一份 FP16 权重"（旧报告的解释）。
- `w_even` / `w_odd` 在函数返回前一直被局部变量引用，`w_int4.float()` 的结果在算出 `w_fp16` 后也还被引用。
- FP16 的 peak − static 全部是 prefill 对**全部 256 个位置**算出的 logits（fp16 一份 + `.float()` 后 fp32 一份），而生成只用最后一个位置；每条序列 256 × 128256 × (2 + 4) B ≈ 188 MB，与 E2 中"每 +1 bs 涨 ~190 MB"吻合。
- 对照实验（改进阶段）：

| 改动 | bs=1 peak mem (MB) |
|---|---:|
| 原始实现 | |
| | |
| | |

### E4 INT4 时间花在哪（profiler）

nsys：`prof/int4_bs1.nsys-rep`（bs=1，1 次 prefill + 31 步 decode，warmup 之后）。GPU kernel 总时间 ≈ 32.0 ms / 2.5% ≈ **1.27 s**（FP16 同配置 ≈ 0.20 s）。

| kernel（推断对应的 dequant 步骤） | 时间占比 | 次数 | 次数 / forward |
|---|---:|---:|---:|
| `elementwise_kernel<128,2>` nocast（减 zero_point / 乘 scale） | 19.3% | 4672 | 146 |
| `elementwise_kernel<128,2>` nocast | 19.2% | 3616 | 113 |
| `direct_copy`（交错写回 `w_int4[:, 0::2]` / `.float()`） | 19.1% | 13456 | 420.5 |
| `float16_copy`（`.half()` 等） | 13.8% | 6208 | 194 |
| `elementwise_kernel<128,4>` | 12.7% | 7232 | 226 |
| `AUnaryFunctor<uchar>`（`& 0x0F`） | 4.6% | 7232 | 226 |
| `BUnaryFunctor<uchar>`（`>> 4`） | 2.1% | 3616 | 113 |
| **以上 dequant 相关合计** | **~90.8%** | | |
| GEMM / GEMV（含 lm_head） | ~6.2% | | |

- 量化的 Linear 共 16 × 7 + 1（lm_head）= **113** 个；多个 kernel 的次数 / forward 正好是 113 的倍数 → 每个 Linear 每次 forward 都完整 dequant 一遍。
- 单个 kernel 最长 ~1.57 ms：形状最大的 lm_head（128256 × 2048）。
- GPU 忙碌 ≈ 1.27 s / 32 ≈ **39.6 ms / forward**，与 decode 实测 40.5 ms/step 几乎相等。
- `cudaLaunchKernel`：INT4 61,378 次（~1,920 / forward）vs FP16 21,602 次（~680 / forward）。
- 计时结束时的 `cudaDeviceSynchronize`（CPU 发完所有 kernel 后等 GPU）：**INT4 7.6 ms vs FP16 0.31 ms** → INT4 是 CPU 跑在前面等 GPU（GPU-bound），FP16 是 GPU 一直追着 CPU（launch-bound）。

结论：INT4 时间 ~91% 花在 dequant 的逐元素 kernel 上，矩阵乘本身只占 ~6%。且 **INT4 decode 是 GPU-bound**（GPU 几乎不空闲），与 FP16 的 launch-bound 相反：dequant 把每步 GPU 工作从 ~6 ms 撑到 ~40 ms，远超 CPU 发 kernel 的 ~15 ms。

### E5 Roofline：decode 离带宽上限多远

| | 每 token 读取的权重字节 | 理论下限 (ms/token) | 实测 (ms/token) | 带宽利用率 |
|---|---:|---:|---:|---:|
| FP16 bs=1 | | | | |
| INT4 bs=1 | | | | |

**nsys：FP16 bs=1**（`prof/fp16_bs1.nsys-rep`，1 次 prefill + 31 步 decode，warmup 之后）

| 指标 | 值 | 来源 |
|---|---:|---|
| kernel launch / forward | ~680 | (21602 `cudaLaunchKernel` + 272 `cuLaunchKernel` + 32 `ExC`) / 32 |
| GPU kernel 总时间 | ~199 ms | GEMM 37.2 ms 占 18.7% |
| GPU 忙碌 / decode 步 | ~6 ms | (199 − prefill ~15) / 31 |
| NVTX decode 区间（CPU） | ~23 ms | nsys 下；不开 nsys 时 15 ms（每 launch 追踪开销 ~12 µs） |
| copy / cast 小 kernel | 5200+ 次，~27 ms（13.6%） | `direct_copy`、`float16_copy` |
| decode lm_head GEMM | 406 µs × 31 | 525 MB 权重 → 理论 338 µs → ~83% 带宽 |

结论：bs=1 decode 每步 15 ms 中 GPU 只忙 ~6 ms，其余在等 CPU 发 kernel（~680 个 × ~20 µs）→ **launch-bound**，解释了 decode 不随 bs 变、bs=1 prefill ≈ 1 步 decode。GPU 侧 6 ms 仍高于 1.6 ms 带宽下限，主要来自大量小 elementwise / copy kernel；单个大 kernel（lm_head）已达 ~83% 带宽。

待验证（改进阶段）：CUDA Graphs / `torch.compile(mode="reduce-overhead")` 应把 decode 从 ~15 ms 降到 ~6 ms。

---

## Part 2：MoE

### E6 参数、显存与速度（N=4, K=2, LoRA r=8）

| | Dense | MoE-slice | MoE-LoRA |
|---|---:|---:|---:|
| Total params | 1,498,482,688 | 1,498,613,760 | 1,500,710,912 |
| Trainable params | 0 | 131,072 | 2,228,224 |
| Peak mem (MB) | 3074 | 3075 | 3082 |
| 推理时间 (s)，bs=1, in=256, out=32（E0） | 0.56 | 2.71 | 2.83 |
| prefill (ms)（`bench.py`） | 17.71 | 1695.58 | 1695.96 |
| decode (ms/step)（`bench.py`） | 14.96 | 24.24 | 27.05 |
| 生成速度 (tok/s)，eval_moe，10 条短 prompt，out=32 | 53.5–55.1 | 23.7–31.8 | 28.7–28.9 |

观察：benchmark（prompt 256）里 MoE 比 Dense 慢 ~5×，eval_moe（短 prompt）里只慢 ~2×。`bench.py` 拆开后：**prefill 慢 ~96×（17.7 ms → 1696 ms），decode 只慢 1.6–1.8×**。

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
- slice 训练把 PPL 从 32 万降到 3431（降了约一百倍），但仍离 dense 9.66 很远。
- LoRA acc +0.16 个百分点，在 5711 个 token 上 ≈ 多猜对 9 个 token —— 是否超过噪声见 E9。

loss 曲线图：

#### E8b 训练扫描（`scripts/run_sweep.sh`，2026-10-13 夜）

都是 200 条训练样本、SGD momentum 0.9、seed 42；评估同一 held-out（5711 tokens）。

| run | 可训练参数 | 每 epoch loss | PPL | acc |
|---|---:|---|---:|---:|
| slice（默认 2 ep） | 131,072 | 10.4605 → 8.2425 | 3430.71 | 4.08% |
| slice_ep6 | 131,072 | 10.46 → 8.24 → 7.88 → 7.65 → 7.56 → 7.49 | 2174.31 | 4.80% |
| slice_unfreeze（2 ep，expert 权重也训，FP16） | 805,437,440 | 9.2276 → **9.3531** | 6713.77 | 2.85% |
| lora（默认 2 ep，lr_expert 1e-4） | 2,228,224 | 2.3773 → 2.3723 | 9.59 | 51.53% |
| lora_lr1e-3（2 ep） | 2,228,224 | 2.3542 → 2.2942 | **10.56** | 51.53% |
| lora_ep6 | 2,228,224 | 2.3773 → 2.3723 → 2.3669 → 2.3607 → 2.3532 → 2.3439 | 9.36 | 52.20% |

观察（数字层面）：
- slice 训练 3× 久：loss 继续降但越降越慢（7.49，dense 量级 ~2.4），PPL 3431 → 2174，仍是 dense 的 225×。
- slice 放开 expert 权重（805 M 参数、FP16、SGD）：epoch 2 loss **上升**，PPL 比只训 router 还差；没出现 NaN。
- LoRA lr ×10：训练 loss 降得更多，**held-out PPL 反而从 9.66 升到 10.56**，acc 不变。
- LoRA 训练 3× 久：loss 单调慢降，PPL 9.36、acc 52.20%（比 dense +0.83 个百分点）。
- slice_ep6 前两个 epoch 与默认 slice run 逐位一致（同机同 seed 可复现）；与旧报告 A40 的 slice 数字不一致。

### E9 LoRA 有没有在学

| step | router grad norm | `lora_B` weight norm | loss |
|---:|---:|---:|---:|
| | | | |

- B=0 时 router 梯度的预测：
- PPL 差异是否超过噪声（不同 seed / eval 子集）：

### E10 MoE 慢在哪

- 预测同步次数（prefill 256 tokens）：
- nsys 实测（`prof/moelora_bs1.nsys-rep`，MoE-LoRA bs=1，1 次 prefill + 31 步 decode）：

| CUDA API | 次数 | 总时间 | 对照 FP16 |
|---|---:|---:|---:|
| `cudaLaunchKernel` | 109,890 | 761 ms | 21,602 |
| `cudaMemcpyAsync` | 19,392 | 198 ms | 1,024 |
| `cudaStreamSynchronize` | **9,184** | 105 ms | 0 |

- 9,184 = 256 × 2 × 16（prefill：token × top-K × 层）+ 31 × 1 × 2 × 16（decode）= 8,192 + 992，**精确吻合**。
- 19,392 = 2 × 9,184 + 1,024（1,024 与 FP16 相同，是 KV cache 写入）。
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
