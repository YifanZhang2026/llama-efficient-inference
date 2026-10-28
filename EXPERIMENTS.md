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

`logs/benchmark_orig.log`，2026-10-13（原始日志已被重跑覆盖，见下方"复现"）。计时为 `generate()` 端到端（prefill + 32 步 decode + 采样），无 warmup，单次运行。

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
- **复现**：2026-10-14 02:31 独立重跑一次（`results/logs/benchmark_orig_rerun_1014.log`）：FP16 1.11 / 0.70 / 0.73 s，INT4 1.56 / 1.66 / 1.73 s，Dense 0.57 s，MoE 2.79 / 2.81 s；峰值显存逐 MB 相同。冷启动现象（FP16 bs=1 ≈ 2× Dense）再次出现。

---

## Part 1：INT4 量化

### E1 模型大小

| | 预测 (MB) | 实测 (MB) |
|---|---:|---:|
| FP16 | | 2858.1 |
| INT4（params + buffers） | | 1127.2 |
| 减少比例 | | 60.6% |

实测：`eval_ppl.py` 的 `params + buffers`（`logs/eval_ppl_1014.log`），与 `run_benchmark.py` 的 `print_model_size` 一致（INT4：parameters 501.13 MB + buffers 626.08 MB）。不含 KV cache。其他 group size：g64 1164.0 MB，g32 1237.7 MB。

推导：

### E1b INT4 质量（PPL / next-token acc）

`eval_ppl.py --group-sizes 32 64 128 --ctx 2048 1024`，`logs/eval_ppl_1014.log`，`results/ppl.csv`，2026-10-14。INT4 = 本仓库的 RTN（每组 min/max 非对称，scale / zero_point 存 FP16），不做校准。所有模型在同一批片段上逐段对比；agree = INT4 与 FP16 argmax 相同的比例，KL = 每 token 平均 KL(P_fp16 ‖ P_int4)（nats）。

口径：
- **Alpaca[200:300]**：与 `eval_moe.py` 相同（每条样本单独，bos+eos，截断 128 token），5711 tokens。FP16 结果 9.6602 / 51.37% 与 E7 的 Dense 行一致。
- **WikiText-2 test**（raw）：`"\n\n".join(test["text"])` 整体 tokenize（只在最开头加 BOS），共 289,082 tokens，切成不重叠的 ctx 窗口，丢弃尾部，每个窗口预测后 ctx−1 个 token，PPL = exp(总 NLL / 预测 token 数)。GPTQ / AWQ 用的是同一种切法，但除以 ctx（差 <0.1%）。
- 模型代码没有 Llama 3.1+ 的 RoPE scaling（P3），所以 WikiText 的绝对 PPL 可能与 HF 参考值略有不同；FP16 与 INT4 共用同一份模型代码，差值不受影响。

| | Alpaca PPL | acc | agree | KL | WikiText-2 ctx=2048 PPL | acc | agree | KL | ctx=1024 PPL | acc | agree | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 9.66 | 51.37% | – | – | 9.89 | 51.71% | – | – | 11.08 | 50.36% | – | – |
| INT4 g32 | 10.30 | 50.27% | 83.94% | 0.088 | 11.03 | 50.09% | 83.06% | 0.110 | 12.38 | 48.72% | 82.65% | 0.112 |
| INT4 g64 | 10.55 | 49.92% | 79.92% | 0.119 | 11.52 | 49.45% | 80.56% | 0.154 | 12.90 | 48.12% | 80.40% | 0.153 |
| INT4 g128（默认） | 11.03 | 49.87% | 80.00% | 0.157 | 12.29 | 48.51% | 77.27% | 0.217 | 13.71 | 47.19% | 77.23% | 0.214 |

预测：

观察（数字层面，解释待写）：
- 默认 g128 在 WikiText-2 ctx=2048 上 PPL 9.89 → 12.29（**+2.40，+24%**），acc −3.2 个百分点；Alpaca 上 9.66 → 11.03（+14%）。
- group 越小越好：WikiText ctx=2048 时 g128 / g64 / g32 的 ΔPPL 分别是 +2.40 / +1.63 / +1.14。KL 随 group size 单调增大（g32 → g128 约翻倍）；agree 在 Alpaca 上 g64 与 g128 基本相同（79.92% vs 80.00%）。
- 即使是 g32，也有 **~17% 的位置 top-1 预测与 FP16 不同**。
- ctx 1024 → 2048：FP16 PPL 11.08 → 9.89（上下文更长）；INT4 的 ΔPPL 两种 ctx 下相近（g128：+2.63 vs +2.40），KL 几乎不变。

解释：

### E1c 对称 vs 非对称 INT4，以及权重分布

`quant_sym.py`（`logs/quant_sym_1015.log`，`results/quant_sym*.csv`，2026-10-15）和 `weight_stats.py`（`logs/weight_stats_1028.log`，2026-10-28）。g128，113 个 Linear，所有方案的 scale 都存 FP16。

- **asym**：本仓库的实现，scale = (max − min)/15，zp ∈ [0, 15]。
- **sym7**：zp = 0，scale = absmax/7，q = clamp(round(w/s), −8, 7)（−8 实际用不到，共 15 级）。
- **sym8**：zp = 0，scale = absmax/8，同样 clamp 到 [−8, 7]（16 级全用，最大的正权重被截到 7·s）。

**重建误差**（‖W − Ŵ‖² / ‖W‖²，按层类汇总，×1e-3）

| 层类 | asym | sym7 | sym8 | sym7 / asym | sym8 / asym |
|---|---:|---:|---:|---:|---:|
| wq | 11.124 | 15.722 | 12.713 | 1.413 | 1.143 |
| wk | 11.756 | 17.253 | 13.976 | 1.468 | 1.189 |
| wv | 13.016 | 19.438 | 15.692 | 1.493 | 1.206 |
| wo | 11.200 | 15.578 | 12.595 | 1.391 | 1.125 |
| w1 | 10.635 | 14.769 | 11.947 | 1.389 | 1.123 |
| w3 | 10.589 | 14.648 | 11.850 | 1.383 | 1.119 |
| w2 | 10.958 | 15.440 | 12.488 | 1.409 | 1.140 |
| output | 11.259 | 16.048 | 12.941 | 1.425 | 1.149 |
| **全部** | **10.969** | **15.441** | **12.481** | **1.408** | **1.138** |

sym7、sym8 在 **113 / 113** 层上都比 asym 差。

**PPL**（与 E1b 同口径；asym 逐位复现 E1b 的 int4_g128）

| | Alpaca PPL / acc | agree / KL | WikiText-2 ctx=2048 PPL / acc | agree / KL |
|---|---|---|---|---|
| FP16 | 9.66 / 51.37% | – | 9.89 / 51.71% | – |
| asym | 11.03 / 49.87% | 80.00% / 0.157 | 12.29 / 48.51% | 77.27% / 0.217 |
| sym7 | 11.41 / 48.82% | 76.29% / 0.213 | 13.43 / 47.36% | 73.73% / 0.301 |
| sym8 | 11.29 / 48.94% | 76.19% / 0.184 | 12.79 / 48.01% | 75.82% / 0.252 |

**权重分布**（a：各层类合并统计）

| 层类 | 元素数 | 均值 | 标准差 | 均值 / 标准差 | 超额峰度 | max \|w\| | max \|w\| / 标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wq | 67.1 M | −2.70e-6 | 0.02761 | −0.0001 | 3.17 | 0.801 | 29.0 |
| wk | 16.8 M | 1.22e-5 | 0.03719 | 0.0003 | 2.67 | 1.133 | 30.5 |
| wv | 16.8 M | −7.98e-7 | 0.01450 | −0.0001 | 4.46 | 0.303 | 20.9 |
| wo | 67.1 M | −3.50e-7 | 0.01530 | −0.0000 | 4.26 | 0.734 | 48.0 |
| w1 | 268.4 M | −5.36e-5 | 0.02174 | −0.0025 | 2.64 | 0.996 | 45.8 |
| w3 | 268.4 M | −3.25e-7 | 0.01728 | −0.0000 | 2.66 | 1.234 | 71.4 |
| w2 | 268.4 M | −2.65e-7 | 0.01685 | −0.0000 | 1.04 | 0.867 | 51.5 |
| output | 262.7 M | −1.07e-4 | 0.02191 | −0.0049 | 0.43 | 0.346 | 15.8 |
| tok_embeddings（不量化） | 262.7 M | −1.07e-4 | 0.02191 | −0.0049 | 0.43 | 0.346 | 15.8 |

- 所有层类的均值 / 标准差都在 ±0.005 以内。
- `output` 与 `tok_embeddings` 的所有统计量（包括 (c) 的离群列下标）完全相同；`torch.equal(output.weight, tok_embeddings.weight)` = **True**（`logs/weight_stats_1028b.log`，2026-10-29）：checkpoint 中两者是同一份权重（tied），只是本仓库的模型代码把它们加载成两个独立张量。按不重复计数，参数量是 1,498,482,688 − 262,668,288 = 1,235,814,400。

（b）**zero_point 直方图**（各层类 zp = 7 和 zp = 8 各占 34–38%，zp ∈ [6, 9] 合计 95–97%；平均 zp 7.50，output 7.54；对称的组 zp ≈ 7.5）与**每组偏斜度** s = (max + min)/(max − min)：

| 层类 | zp 6–9 合计 | \|s\| < 0.05 | 0.05 ≤ \|s\| < 0.25 | \|s\| ≥ 0.25 | s 的 p5 / p50 / p95 |
|---|---:|---:|---:|---:|---|
| wq | 95.7% | 31.5% | 63.0% | 5.6% | −0.210 / 0.000 / 0.210 |
| wk | 92.9% | 30.1% | 61.5% | 8.3% | −0.234 / 0.000 / 0.234 |
| wv | 93.7% | 29.9% | 62.4% | 7.8% | −0.230 / 0.000 / 0.230 |
| wo | 97.1% | 32.0% | 64.0% | 4.0% | −0.200 / 0.000 / 0.199 |
| w1 | 97.2% | 33.1% | 63.1% | 3.8% | −0.195 / 0.000 / 0.193 |
| w3 | 97.5% | 33.2% | 63.4% | 3.4% | −0.193 / 0.000 / 0.193 |
| w2 | 96.2% | 31.6% | 63.4% | 5.0% | −0.207 / 0.000 / 0.207 |
| output | 94.7% | 29.2% | 63.8% | 6.9% | −0.240 / −0.002 / 0.214 |

（c）**离群输入通道**：每层按输入列取 max |w|，最大列 / 中位列：

| 层类 | 16 层中的 min / median / max | 在多层中重复进入 top-5 的列（出现层数） |
|---|---|---|
| wq | 1.76 / 2.61 / 6.07 | 524 (4)，72 (4) |
| wk | 1.93 / 3.13 / 8.06 | 1618 (6)，781 (5)，435 (4)，674 (4) |
| wv | 1.90 / 2.66 / 4.16 | 917 (4)，1063 (4) |
| wo | 4.20 / 6.78 / 10.77 | 无重复 |
| w1 | 2.57 / 3.82 / 7.08 | 352 (5)，1107 (3)，1809 (3)，390 (3) |
| w3 | 2.95 / 5.04 / 10.62 | **1107 (8)**，1504 (4)，923 (4)，352 (4)，1634 (4) |
| w2 | 5.92 / 8.37 / 12.16 | 2070 (2)（w2 的输入是 8192 维 FFN 隐层） |
| output | 3.45（1 层） | – |

wq / wk / wv / w1 / w3 的输入都是 attention_norm / ffn_norm 之后的 2048 维隐状态，同一组下标在不同层类中重复出现（如 1107、352）。

预测：

解释：

### E1d INT4 精度损失来自哪些层

`int4_sensitivity.py --per-layer`（`logs/int4_sensitivity_1028.log`，`results/int4_sensitivity.csv`，2026-10-28）。g128 asym；每个配置只量化一部分 Linear，其余保持 FP16（伪量化：把反量化后的 INT4 权重写进 fp16 Linear，数值上与 `QuantizedLinear` v1 相同）。口径同 E1b；ΔPPL、KL 均相对 FP16（Alpaca 9.66，WikiText-2 9.89）。`all` 逐位复现 E1b。

| 配置 | 量化的 Linear | 量化参数 | Alpaca ΔPPL | WikiText-2 ΔPPL | WikiText-2 KL | WikiText-2 acc |
|---|---:|---:|---:|---:|---:|---:|
| all | 113 | 1235.7 M | +1.367 | **+2.402** | 0.2174 | 48.51% |
| no_output（a） | 112 | 973.1 M | +1.238 | +2.208 | 0.2017 | 48.76% |
| only_wq | 16 | 67.1 M | +0.016 | +0.066 | 0.0062 | 51.61% |
| only_wk | 16 | 16.8 M | +0.014 | +0.047 | 0.0046 | 51.66% |
| only_wv | 16 | 16.8 M | +0.140 | +0.163 | 0.0151 | 51.50% |
| only_wo | 16 | 67.1 M | +0.152 | +0.157 | 0.0161 | 51.51% |
| only_w1 | 16 | 268.4 M | +0.094 | +0.149 | 0.0136 | 51.52% |
| only_w3 | 16 | 268.4 M | +0.115 | +0.245 | 0.0234 | 51.35% |
| **only_w2** | 16 | 268.4 M | **+0.449** | **+0.943** | **0.0961** | 50.43% |
| only_output | 1 | 262.7 M | +0.104 | +0.142 | 0.0141 | 51.49% |

- 不量化 lm_head：WikiText ΔPPL 2.402 → 2.208（恢复 0.194，即 8%）；需要额外存 262.7 M 个 FP16 参数（501 MB）。
- 8 个"只量化一类"的 WikiText ΔPPL 之和 = 1.912，小于 `all` 的 2.402。
- w2 占量化参数的 21.7%，单独造成 +0.943（`all` 的 39%）；wq + wk 合计 +0.113。

（c）每次只量化一个 transformer block（7 个 Linear，60.8 M 参数），WikiText-2 ΔPPL：

| block | 0 | **1** | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ΔPPL | 0.154 | **0.556** | 0.069 | 0.054 | 0.054 | 0.068 | 0.063 | 0.090 | 0.070 | 0.087 | 0.078 | 0.069 | 0.065 | 0.053 | 0.071 | 0.115 |
| KL | 0.016 | **0.059** | 0.005 | 0.006 | 0.006 | 0.006 | 0.006 | 0.007 | 0.008 | 0.008 | 0.008 | 0.007 | 0.006 | 0.006 | 0.007 | 0.012 |

- block 1 最敏感（+0.556），其次 block 0（+0.154）和 block 15（+0.115）；其余 13 个在 +0.05 – +0.09。
- Alpaca 上 block 1 只有 +0.130、block 5 为 −0.006：Alpaca（5711 token）上的逐层差异较噪，以 WikiText-2（288,627 token）为准。

**生成样例**：`results/int4_generations.md`（6 条 MT-bench 风格 prompt，greedy，最多 64 个新 token，FP16 与 INT4 g128 并排）。注意 Llama 3.2-1B 是**基座模型**（未指令微调），FP16 自身也经常复述问题、重复句子；第 4 条（火车距离题）INT4 输出退化为 "1. 2. 3. …" 的序号列表。

预测：

解释：

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
- **FP16 prefill 吞吐**：bs=16（4096 token）87 ms；按 2 × 1.5 B 参数 × 4096 ≈ 12.3 TFLOP 算 ≈ 141 TFLOPS，接近 GEMM 冒烟测试测到的 ~149 TFLOPS（2026-10-15 重测为 200–203 TFLOPS，见 E5b；149 无原始记录）。bs=1 只有 ~43 TFLOPS。
- **INT4 每次 forward 多出一个近似固定的开销**：decode 多 ~25.5 ms，prefill 多 ~28 ms，与 token 数基本无关。
- **static**：INT4 1137 MB vs FP16 2869 MB（−60%）。bs 每 +1，static 涨 ~9.4 MB = 该序列的 KV cache。
- **peak − static**：FP16 198 / 1513 / 3023 MB ≈ 每条序列 ~190 MB；INT4 bs=1 是 3399 MB。
- **cold**：FP16 进程内首轮比稳态多 ~500 ms，与 E0 中的 0.54 s 吻合（P1）。
- `generate()` 端到端（E0 Dense 0.56 s）比 `bench.py` 的 483 ms 多 ~80 ms：top-p 采样、tokenizer 等 forward 以外的开销。

### E2c FP16 decode：bs=16 比 bs=1 慢 0.9 ms，以及 bs=16 前几步特别快

`decode_steps.py --mode fp16 --bs 1 16 --iters 30`（`logs/decode_steps_1028.log`，`results/decode_steps.csv`，每步耗时全部保留），nsys 各采 1 轮（`prof/fp16_bs{1,16}_1028.nsys-rep`，`bench.py --profile`），2026-10-28。

| | bs=1 | bs=16 |
|---|---:|---:|
| decode 中位数 (ms) | 15.63 | 16.51 |
| p10 / p90 (ms) | 15.32 / 16.79 | **14.11** / 16.88 |
| 低于 0.9 × 中位数的步数 | 0 / 930 | **101 / 930** |
| 按步序号的中位数：第 0 / 1 / 2 / 3 / 4 步 (ms) | 14.98 / 15.81 / 15.54 / 15.69 / 15.63 | **10.16 / 10.14 / 9.80** / 15.33 / 16.51 |

**nsys：每个 NVTX 区间内发射的 kernel**（nsys 下 CPU 时间整体偏大）

| | bs=1 | bs=16 |
|---|---:|---:|
| prefill：CPU 区间 / GPU 忙碌 (ms) | 25.9 / 11.5 | 27.7 / **86.5** |
| prefill 的最后一个 kernel 结束 − CPU 区间结束 (ms) | 1.0 | **61.3** |
| decode_0：kernel 开始 − CPU 开始发射 (ms) | 0.9 | **61.1** |
| decode_0 / decode_1：GPU 跨度（首 kernel 开始 → 末 kernel 结束）(ms) | 22.5 / 23.2 | **10.4 / 10.3** |
| decode_3 起：CPU 区间中位数 (ms) | 23.1 | 24.6 |
| decode_3 起：GPU 忙碌中位数 (ms) | 5.4 | 8.6 |
| decode 每步 kernel 数 | 684 | **780**（+96） |

稳态一步（decode_10）的 kernel 差异：

| kernel | bs=1 个数 / GPU µs | bs=16 个数 / GPU µs |
|---|---|---|
| `direct_copy`（`elementwise_kernel<128,4>`） | 32 / 239 | **96 / 3448** |
| 新增 Half 乘法 `BinaryFunctor<Half,…,Mul>` | 0 / 0 | 33 / 137 |
| GEMM / GEMV | bs=1 用 gemv（`gemv2T`、`gemvx`）；bs=16 换成 tensor-core GEMM（`ampere_*s16816gemm*`、`sm80_xmma_gemm*`）+ `splitKreduce` | |

- bs=16 的 prefill 在 GPU 上要 86.5 ms，而 CPU 27.7 ms 就发射完了；随后 decode_0、decode_1 的 kernel 排在 prefill 之后，GPU 跨度（10.4 ms）≈ GPU 忙碌时间（9.3 ms）。从 decode_2 / decode_3 起，GPU 跨度 ≈ CPU 区间。bs=1 的 prefill GPU 只忙 11.5 ms，没有这一段。
- decode 每步多 96 个 kernel；nsys 下 CPU 每步多 1.5 ms（24.6 vs 23.1），不开 nsys 时 decode 中位数多 0.9 ms（16.51 vs 15.63）。
- `direct_copy` 每层从 2 次变成 6 次，GPU 时间 0.24 → 3.45 ms / 步；交接文档中的假设（`repeat_kv` 展开 KV，约 1.3 GB / 步 ≈ 0.85 ms）只对应其中一部分；稳态下 GPU 忙碌 8.6 ms 仍小于 CPU 24.6 ms。

预测：

解释：

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

### E4b dequant 的 HBM 搬运量

`dequant_traffic.py`，2026-10-15。推算（每个权重）：`& 0x0F` 1 + `>> 4` 1 + `& 0x0F` 1 + 两次跨步写 1 + 1 + `.float()` 5 + `- zp` 8 + `* scale` 8 + `.half()` 6 = 32 B，再加 `F.linear` 读 fp16 权重 2 B = 34 B；FP16 模型只读 2 B。

- **A**：`dequantize_packed` 拆成单个 kernel 逐个计时（CUDA event，warmup 5，取 50 次中位数），用推算字节数算有效带宽。
- **B**：1 GiB uint8 `copy_()` 的带宽（读 + 写）作为本卡可达带宽，预测整段 dequant 时间 = 32 B × N / 该带宽。
- **C**：Nsight Compute 读每个 kernel 的 DRAM / L2 字节数。**A100 容器报 `ERR_NVGPUCTRPERM`（无性能计数器权限），C 改在本地 RTX 5070 Laptop 上做**。

#### A + B：A100（`logs/dequant_traffic_1015.log`，真实权重）

1 GiB copy：`clone()` 1373 GB/s，`copy_()` 1378 GB/s（标称 1555 GB/s 的 89%）。下表"占 copy"以 1378 GB/s 为 100%。

| 步骤 | lm_head [128256, 2048] ms | 推算 B / 权重 | 有效带宽 GB/s | 占 copy | wq [2048, 2048] ms | 占 copy |
|---|---:|---:|---:|---:|---:|---:|
| `& 0x0F`（even） | 0.207 | 1 | 1270 | 92% | 0.027 | 11% |
| `>> 4` | 0.209 | 1 | 1257 | 91% | 0.027 | 11% |
| `& 0x0F`（odd） | 0.207 | 1 | 1270 | 92% | 0.026 | 12% |
| `w_int4[:, 0::2] =` | **0.508** | 1 | 517 | **37.5%** | 0.027 | 11% |
| `w_int4[:, 1::2] =` | **0.508** | 1 | 517 | **37.5%** | 0.026 | 12% |
| `.float()` | 1.282 | 5 | 1024 | 74% | 0.028 | 55% |
| `zp` / `scale` `.float()` | 0.035 | ≈0 | – | – | 0.033 | – |
| `- zp` | 1.572 | 8 | 1337 | 97% | 0.038 | 64% |
| `* scale` | 1.572 | 8 | 1337 | 97% | 0.038 | 64% |
| `.half()` | 1.193 | 6 | 1321 | 96% | 0.029 | 64% |
| **整段 `dequantize_packed`** | **7.195** | 32 | 1168 | 85% | 0.206 | 47% |
| `F.linear` fp16，1 token | 0.398 | 2 | 1319 | 96% | 0.039 | 16% |

- lm_head：各步之和 7.292 ms；按 32 B × N / copy 带宽预测 **6.100 ms**，实测 7.195 ms。
- `.float() : - zp : * scale : .half()` 耗时比 = 1.282 : 1.572 : 1.572 : 1.193 ≈ 6.5 : 8 : 8 : 6（推算 5 : 8 : 8 : 6）。
- wq（N = 4.2 M）：每步 26–38 µs，与推算字节数基本无关；整段 0.206 ms vs 预测 0.097 ms。
- lm_head 整段 dequant（7.2 ms）是同一权重 fp16 `F.linear`（0.40 ms）的 18 倍。

#### C：本地 RTX 5070 Laptop（`results/ncu_dequant_local_1015.csv`）

GPU：GeForce RTX 5070 Laptop（Blackwell，CC 12.0，GDDR7，无 ECC，L2 32 MiB），torch 2.11.0+cu128，ncu 2025（CUDA 13.3）。形状 [65536, 2048]（N = 134 M，随机 fp16 权重；8 GB 显存放不下 lm_head 的中间张量）。

```
ncu --profile-from-start off --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,gpu__time_duration.sum \
    --csv --log-file results/ncu_dequant_local_1015.csv python dequant_traffic.py --ncu --shape big=65536x2048
```

本卡 A + B（`results/logs/dequant_traffic_local_1015.log`）：copy 332 GB/s；跨步写 1.05–1.12 ms = copy 的 36–38%；整段 dequant 14.36 ms vs 预测 12.93 ms —— 与 A100 上的比例相同。

| 步骤 | 推算 B / 权重 | DRAM 读 | DRAM 写 | DRAM 合计 | L2 请求（`lts__t_bytes`） | kernel 时间 (µs) |
|---|---:|---:|---:|---:|---:|---:|
| `& 0x0F`（even） | 1（0.5 + 0.5） | 0.500 | 0.346 | 0.846 | 1.000 | 349 |
| `>> 4` | 1（0.5 + 0.5） | 0.575 | 0.355 | 0.930 | 1.075 | 396 |
| `& 0x0F`（odd） | 1（0.5 + 0.5） | 0.507 | 0.347 | 0.854 | 1.007 | 353 |
| `w_int4[:, 0::2] =` | 1（0.5 + 0.5） | **1.455** | **0.917** | **2.372** | 2.201 | 1060 |
| `w_int4[:, 1::2] =` | 1（0.5 + 0.5） | **1.620** | **0.915** | **2.535** | 2.415 | 1142 |
| `.float()` | 5（1 + 4） | 1.002 | 3.900 | 4.902 | 5.002 | 2336 |
| `zp` / `scale` `.float()` | ≈0 | 0.033 | 0.000 | 0.033 | 0.097 | 37 |
| `- zp` | 8（4 + 4） | 4.045 | 3.871 | 7.916 | 8.137 | 3244 |
| `* scale` | 8（4 + 4） | 4.042 | 3.866 | 7.908 | 8.135 | 3237 |
| `.half()` | 6（4 + 2） | 4.168 | 1.879 | 6.047 | 6.169 | 2509 |
| **合计** | **32** | | | **34.34** | **35.24** | |

单位：字节 / 权重（= 计数器读数 / N）。

- 两次跨步写：DRAM 读 1.46 / 1.62 B（推算 0.5），写 0.92 B（推算 0.5），L2 请求 2.2 / 2.4 B（推算 1）。作为参照，`w_int4` 整个数组是 1 B / 权重。
- 其余 kernel 的 L2 请求字节与推算一致（1.00 / 1.08 / 1.01 / 5.00 / 8.14 / 8.14 / 6.17）。
- `.float()` 的字节数与推算一致（L2 5.00 B），但耗时占比高于推算（本卡 copy 的 90%，A100 上 74%）。
- 非跨步的 kernel，DRAM 写都比推算的写字节少 0.10–0.15 B / 权重（≈ 13–21 MB，小于 32 MiB L2）；可能是 kernel 结束时仍留在 L2、尚未写回的脏数据不计入 DRAM 写，所以 DRAM 写是下界。

口径与局限：
- ncu 默认在每个 kernel 前清空缓存（`--cache-control all`），读都是冷的；实际推理中上一步的输出可能部分在 L2 中。
- 本卡是 GDDR7、无 ECC、L2 32 MiB；A100 是 HBM2、ECC 开启、L2 40 MB。L2 sector（32 B）粒度两者相同，DRAM 侧对部分写的处理可能不同，A100 上的 DRAM 字节数没有直接测到。

预测：

解释：

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

### E5b GEMM 吞吐与降频（功耗墙）

`gemm_smoke.py` + `scripts/run_throttle.sh`（`logs/{dmon,smi,smi_q,throttle}_*`，2026-10-15）。四个阶段：空闲 10 s、4096³ FP16 GEMM 30 s、`bench.py --mode fp16 --bs 1 --iters 100`、`bench.py --mode int4 --bs 1 --iters 100`。每 500 ms 用 `nvidia-smi --query-gpu` 采一次功耗 / 温度 / 频率 / 各降频原因。下表的"忙"= GPU 利用率 ≥ 50% 的采样点。

| 阶段 | 功耗中位数（最大） | SM 频率中位数（最小） | 显存频率 | GPU / 显存最高温度 | 利用率中位数 | 忙时处于 Active 的降频原因 |
|---|---|---|---:|---|---:|---|
| 空闲 | 67 W（72） | 1410 MHz | 1215 MHz | 45 / 44 °C | 0% | – |
| GEMM 4096³ | **248 W**（257） | **1050 MHz**（1020） | 1215 MHz | 52 / 49 °C | 100% | **SW Power Cap：100% 的忙采样点** |
| FP16 bench bs=1 | 65 W（88） | 1125 MHz（210） | 1215 MHz | 41 / 40 °C | 38% | 无（Idle 7%） |
| INT4 bench bs=1 | 211 W（231） | 1410 MHz | 1215 MHz | 49 / 52 °C | 100% | 无 |

- 功耗上限（power limit）250 W，最高 SM 频率 1410 MHz；温度远低于降频阈值（GPU Slowdown 92 °C，显存最高 95 °C）。
- **GEMM 实测 200.0 – 203.3 TFLOPS**（35 个约 0.9 s 的窗口，中位数 201.0）。按 312 TFLOPS × 1050 / 1410 ≈ 232 TFLOPS 算，实测为其 87%。E2 中引用的"约 149 TFLOPS"没有留下脚本和日志，无法复核。
- `nvidia-smi -q` 中 `SW Power Capping` 的累计计数器在所有阶段都停在 566,796 µs 不变，不可用；以 500 ms 采样为准。
- 同期 bench：FP16 decode 14.90 ms/步，INT4 40.45 ms/步，与 E2 一致。

预测：

解释：

## Part 2：MoE 推理性能

> 范围（2026-10-16 决定）：MoE 只作为**推理性能**案例保留（为什么慢：逐 token 循环 + `.item()` 同步）。slice / LoRA 的训练与效果分析（E7–E9a、E11）移到文末**附录 A**，不作为主线。

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

---

## 现阶段待做（2026-10-28 整理）

> 主线：INT4 dequant 优化（Part 3）。每项先写预测再测量，结果写回对应小节。"A100" 指 AutoDL 服务器，"本机" 指 RTX 5070 Laptop（conda env `dp`，可以用 ncu，没有 ckpt）。

### A. Part 3 主线（按顺序做）

| # | 实验 | 在哪 | 命令 / 做法 | 预测 | 写回 |
|---|---|---|---|---|---|
| T1 | **v1–v4 端到端** | A100 | `bench.py --mode int4 --impl {v1,v2,v3,v4} --bs 1 8 16`，再加一组 `--mode fp16` 作为同一时段的基线 | bs=1 decode：v1 40.5 → v2 约 22–25 → v3 Yifan 预测 → v4 约 15 ms（launch-bound，**不会比 FP16 快**） | I1–I3 的"结果" |
| T2 | **瓶颈是否转移**（nsys） | A100 | 每个版本 bs=1 `--profile` 一次，看：每次 forward 的 kernel 数、GPU 忙碌时间、最后一次同步的等待时间、v4 每个 Linear 在 CPU 上花多少时间（Triton 从 Python 发射的开销） | 最后一次同步的等待时间：v1 为 7.6 ms，v2 及以后接近 0（变回 launch-bound）；v4 的 GPU 时间小于 FP16 | I1–I3 |
| T3 | **峰值显存** | A100 | `mem_snapshot.py --mode int4 --impl {v1..v4} --bs 1` | 峰值出现在 prefill 里，而 v4 的 prefill 走的是 v3，所以 v3 和 v4 的峰值应该相同。多出的部分主要是 lm_head 的 fp16 权重（525 MB）加上 logits；v2 约 1.1 GB；v1 为 3.4 GB | I1–I3 |
| T4 | **v4 调参：split-K 还是小 BLOCK_N** | A100 | 单层：6 种形状 × M ∈ {1, 8, 16}，扫 BLOCK_N × num_warps × split；端到端：分别用 (a) 当前的 split-K 和 (b) `split_k=1` 加按 N 选择 BLOCK_N，跑 `bench.py --impl v4` | (b) 的端到端更快：每个 Linear 只发射 1 个 kernel，(a) 要发射 3 个 | I3 |
| T5 | **v4 的数值影响** | A100 | ⚠️ `eval_ppl.py` 是整段 teacher forcing（M = ctx ≥ 128），**根本走不到 v4**，测出来的 PPL 必然和 v1 一样。要改用逐 token decode（M=1，使用 KV cache）：在 Alpaca 子集上算 NLL，同时统计 v1 和 v4 的 greedy 生成逐 token 一致率 | NLL 差异 < 1e-4，一致率接近 100%；少数分叉出现在 logits 前两名非常接近的位置 | I3 |
| T6 | **CUDA Graph**（I4） | A100 | 对 decode 的一步做 capture（KV cache 已经预分配，输入 token 和 start_pos 要放进固定的 buffer），FP16 和 INT4 v4 都做 | FP16 约 6 ms（等于 GPU 忙碌时间），v4 约 4–5 ms，**到这一步 INT4 才首次比 FP16 快** | I4 |
| T7 | 和现成 kernel 对比（可选） | A100 | `torch.ops.aten._weight_int4pack_mm`（tinygemm），需要先转换打包格式；只比单层耗时 | tinygemm 比 v4 快一些 | I3 |
| T8 | vLLM 作为上限（可选） | A100 | FP16：vLLM 跑 Llama 3.2-1B，测 bs=1 decode；INT4：先确认 RTN 权重能不能转成 GPTQ 格式（zp 的 −1 偏移、打包顺序），能转再测 Marlin | — | 新小节 |
| T9 | v2–v4 的 HBM 字节数 | 本机 | 用 ncu 读每个 kernel 的 DRAM 读写字节 | v2 约 15 B / 权重，v3 约 2.5 B（加上 matmul 读取共 4.5 B），v4 约 0.53 B | I1–I3 |

需要先补的工具（Claude 写）：`dequant_traffic.py --impl`（T9）；把本机扫参脚本整理成 `int4_gemv_bench.py`（T4）；逐 token decode 的评估脚本（T5）；`bench.py --cuda-graph`（T6）。

Yifan 待填：I2 中的三项预测（每个 Linear 的 kernel 数、dequant 的 GPU 时间、v3 decode 的耗时和瓶颈）。

### B. 实验线遗留（见 EXP_HANDOFF 第 7–11 项）

| # | 实验 | 状态 |
|---|---|---|
| 7 | 降频原因（dmon、`-q -d PERFORMANCE`） | 已完成 → E5b（5f3b142） |
| 8 | 对称 vs 非对称 INT4，权重分布和 zp 直方图 | 已完成 → E1c（权重分布的 csv 和 tied 检查待重跑 `weight_stats.py`） |
| 9 | FP16 bs=16 比 bs=1 多 0.9 ms：是不是 `repeat_kv` 造成的 | 已完成 → E2c |
| 11 | INT4 精度损失来自哪些层（不量化 lm_head、按层类型逐一量化） | 已完成 → E1d（含逐层与生成样例） |

第 10 项已经由 Part 3 的 I1、I2 代替。

---

## Part 3：dequant 优化（改进阶段）

> 目标：降低 INT4 decode 中 dequant 的 HBM 搬运量（E4 / E4b：占 GPU 时间 91%，每个权重搬运 34 B）。
> 每一步都和上一步对比，并且先写预测、再测量。代码由 Yifan 实现；对比开关和测量脚本由 Claude 添加。

### I1 v2：纯 PyTorch 改写 `dequantize_packed`（状态：已实现，`QuantizedLinear.dequantize_packed_v2`；使用 `--impl v2`）

改动（只改 `llama/quantize.py`，保留 v1 方便对照）：

| # | v1 | v2 |
|---|---|---|
| A | `(packed >> 4) & 0x0F` | `packed >> 4`（uint8 右移后结果已在 0–15 之间） |
| B | 两次跨步写 `w_int4[:, 0::2]` / `[:, 1::2]` | `torch.stack((even, odd), -1).view(...)`，连续写 |
| C | 转为 fp32 计算，最后再 `.half()` | 直接转为 fp16（(q − zp) 是很小的精确整数，只在乘 scale 时舍入一次，结果与 v1 逐位相同） |
| D | `-`、`*` 各生成一个新的 1 GB 张量 | 原地运算 `sub_` / `mul_` |

预测：

| 指标 | v1 实测 | v2 预测 |
|---|---:|---:|
| dequant 字节 / 权重 | 34（ncu） | 约 15（加上 matmul 读取共约 17） |
| 每个 Linear 的 dequant kernel 数 | 11 | 5 |
| lm_head 单层 dequant（A100） | 7.2 ms | 约 3.4 ms |
| bs=1 峰值时多出的显存 | 3.4 GB | 约 1.1 GB |
| 输出 | – | 与 v1 逐位相同（`torch.equal`） |
| INT4 decode / 步（A100, bs=1） | 40.5 ms | 约 22–25 ms |
| 瓶颈 | GPU-bound（最后一次同步等待 7.6 ms） | **预计变回 launch-bound**：GPU 约 22 ms，CPU 需要发射约 1,245 个 kernel，约 25 ms |

验证（本机 = RTX 5070 Laptop，服务器 = A100）：

1. 本机：`dequant_traffic.py --impl v1/v2` 做逐位比较，测单层耗时比值；用 ncu 测每个 kernel 的 DRAM 字节数。
2. 服务器：`check_impl.py`；`bench.py --mode int4 --bs 1 8 16`；`mem_snapshot.py --mode int4 --bs 1`；nsys 采集一次，看最后一次同步的等待时间。
3. 只要逐位相同，PPL 就不用重测。

结果：

- 正确性（本机 RTX 5070 Laptop，随机权重 × 0.02，g128；2026-10-25）：形状 2048×2048、512×2048、8192×2048、2048×8192、128256×2048 全部测试，v2 和 v3 的输出都与 v1 **`torch.equal` 为 True**，dtype 为 fp16，内存连续。

### I2 v3：`torch.compile` 融合 dequant（状态：已实现，`quantize.dequant_fn("v3")`；使用 `--impl v3`）

改动：`_dequant_compiled = torch.compile(QuantizedLinear.dequantize_packed_v2, dynamic=False)`，`forward` 改为调用编译版。**只编译 dequant**，不编译整个模型，避免 FP16 基线一起变化。`dynamic=False`：113 个 Linear 共 6 种形状，每种编译一次。

性质：算子融合由编译器（Inductor 生成 Triton kernel）完成，**不是手写**。这一步的作用是给 I3 提供基线。

| 指标 | v2 预测 | v3 预测 |
|---|---:|---:|
| dequant 字节 / 权重 | 约 15 | 2.5（读 0.5 B，写 fp16 2 B）；加上 matmul 读取共 4.5 |
| 每个 Linear 的 dequant kernel 数 | 5 | Yifan 预测： |
| 一次 forward 的 dequant GPU 时间（A100） | 约 16 ms | Yifan 预测： |
| INT4 decode / 步，瓶颈 | 约 22–25 ms，launch-bound | Yifan 预测： |
| 输出 | 与 v1 逐位相同 | 预期逐位相同（Inductor 在 fp32 下计算，只舍入一次），需用 `torch.equal` 实测 |

附带对照：把 v1 也用 compile 编译。预期速度和 v3 几乎一样，因为融合以后 fp32 中间值和跨步写都只在寄存器里。用来说明哪些优化可以交给编译器。

验证：同 I1；另外用 `TORCH_LOGS=output_code` 导出生成的 Triton kernel，确认每层只有 1 个；`cold_e2e_ms` 包含编译时间，要单独标注。

无法再优化的部分：Inductor 无法融合 cuBLAS（`F.linear` 属于外部 kernel），fp16 权重仍然要完整写回 HBM 再读一次。

结果：

### I3 v4：Triton 融合 dequant + GEMV（状态：已实现，`llama/int4_gemv.py`；使用 `--impl v4`）

目标：int4 权重从 HBM 读进来，在寄存器里反量化后直接做乘加，fp16 权重不再经过 HBM。每个权重搬 0.5 B，加上 scale / zp 共约 0.53 B。

> 代码由 Claude 应 Yifan 要求编写（2026-10-25）。面试前 Yifan 需要能逐行解释。

设计：
- 一个 program 负责 BLOCK_N 个输出，沿 K 方向每次处理一个 group（BLOCK_K = 128）。这样每行每次只需要读一个 scale 和一个 zp。
- 不做交错：y = 低 4 位 · x[0::2] + 高 4 位 · x[1::2]。
- dequant 先舍入成 fp16，保证权重和 v1 逐位相同；只有 fp32 累加的顺序和 cuBLAS 不一样。
- M = 1 时用广播乘加 `tl.sum`；1 < M ≤ 16 时把 M 补到 16，用 `tl.dot`。
- N 太小、program 不够填满 SM 时用 split-K：结果用 atomic_add 累加到 fp32 缓冲区，最后转成 fp16。代价是多 2 个 kernel。
- 只在 decode（M ≤ 16）时走 v4；prefill 退回 v3。

正确性（本机，随机权重）：5 种形状 × M ∈ {1, 8, 16}，v4 与 fp32 参考值的最大误差，和 cuBLAS（v1 权重 + `F.linear`）与参考值的误差在同一量级（约 1e-3）。`QuantizedLinear` 用 v4 时，decode 输出与 v1 的差 ≤ 1e-3，不是逐位相同；prefill 走 v3，逐位相同。**接入模型后需要重测 PPL。**

单层耗时（本机 RTX 5070 Laptop，`triton.testing.do_bench`，`results/int4_gemv_sweep_local_1025.log`；笔记本 GPU 波动较大，小形状的权重可能整块留在 L2 里，因此只看比值）：

| 形状 N×K | M | fp16 F.linear | v1 | v2 | v3 | **v4** | v4 / fp16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×2048（wq/wo） | 1 | 47 µs | 259 | 171 | 73 | **22** | 0.48× |
| 512×2048（wk/wv） | 1 | 16 | 107 | 71 | 35 | **18** | 1.16× |
| 8192×2048（w1/w3） | 1 | 121 | 2105 | 1158 | 351 | **45** | 0.37× |
| 2048×8192（w2） | 1 | 112 | 1759 | 902 | 225 | **44** | 0.40× |
| 128256×2048（lm_head） | 1 | 1861 | 37855 | 21687 | 4649 | **570** | 0.31× |
| 128256×2048（lm_head） | 16 | 1823 | 36464 | 21802 | 4865 | **577** | 0.32× |

- 字节数比是 0.53 / 2 = 0.27×。lm_head 实测 0.31×，达到了本机拷贝带宽的约 80%。
- wk/wv 这类小层每个只有约 0.5 MB，耗时主要是固定开销，所以 v4 并不比 fp16 快。
- **split-K 和 launch-bound 的冲突**（2026-10-25 读代码时发现）：按 A100 的 108 个 SM 算，bs=1 时除 lm_head 外所有层都会走 split-K，每个 Linear 要发射 3 个 kernel（`zeros` + kernel + `.half()`），112 层共 336 次，而不是 112 次。decode 本来就是 launch-bound，这会增加 CPU 时间。A100 上要对比：(a) split-K；(b) 不拆分、把 BLOCK_N 调小（本机 2048×2048 时 bn=32、不拆分最快）。比较标准是**端到端 decode 时间**，不是单个 kernel 的耗时。
- 待在 A100 上做：见"现阶段待做"中的 T1–T7。注意：PPL 评估是整段输入（M ≥ 128），走不到 v4，要用逐 token decode 才能测出 v4 的数值影响（T5）。

结果（A100）：

### I4 CUDA Graph（状态：未开始）

v2 之后 INT4 decode 预计是 launch-bound，CPU 发射时间约 15 ms 以上，比 GPU 时间长。用 CUDA Graph 消除发射开销后，才能看到 v4 在 GPU 时间上的优势。FP16 也要同样加 Graph 再对比。

结果：

---

## 问题记录

> 每个问题一条。面试讲"遇到了什么问题"就照这个讲。

### P1 FP16 bs=1 比 bs=8 还慢

- **现象**：原始 benchmark 中 FP16 bs=1 = 1.10 s，bs=8 = 0.68 s；同配置的 Dense 行（后跑）= 0.56 s。
- **最初假设**：
- **验证实验**：`prof/fp16_bs1_full.nsys-rep`（`scripts/run_sweep.sh`，`nsys` 不带 capture range，抓整个进程；`bench.py --mode fp16 --bs 1 --iters 2` = warmup 3 + 计时 2，共 5 轮）。用 `nsys stats` 导出的 sqlite，按每轮 33 个 `cudaEventRecord` 切分各轮，逐轮比较（`logs/nsys_stats_fp16_full.txt` 是汇总表）。nsys 下各轮都变慢（每次 launch 的追踪开销），但首轮多出的量与不开 nsys 时一致：

| | rep 0（cold） | rep 1 | rep 4 |
|---|---:|---:|---:|
| 整轮 CPU 时间 (ms) | 1312.8 | 800.0 | 751.3 |
| prefill (ms) | **429.0** | 26.7 | 26.7 |
| decode 第 1 步 (ms) | **141.9** | 24.4 | 24.2 |
| decode 第 2 步 (ms) | 26.5 | 24.7 | 24.3 |
| kernel 数 / GPU kernel 总时间 | 21,906 / 203.5 ms | 21,906 / 203.5 ms | 21,906 / 203.4 ms |
| CUDA API 时间（主线程） | 443.3 ms | 179.6 ms | |
| API 之外的 CPU 时间 | 869.5 ms | 620.4 ms | |

rep 0 − rep 1 = +513 ms（不开 nsys 时 `bench.py` 的 cold − e2e ≈ 983 − 483 = 500 ms），**全部在 prefill（+402 ms）和 decode 第 1 步（+117 ms）**；从第 2 步起与稳态相同。GPU 侧工作完全一样（kernel 数、kernel 名称、GPU 时间都相同）。

多出的 513 ms 的组成：
- `cudaLaunchKernel` **+199 ms**，调用次数相同：几乎都来自 10 次"该 kernel 的第一次 launch"（比 rep 1 同位置慢 >0.2 ms 的只有这 10 次，合计 +195 ms）。prefill 里：`vectorized_elementwise_kernel` ×3（26.4 / 17.0 / 8.1 ms）、`softmax_warp_forward` 24.4 ms、`splitKreduce_kernel` 16.8 ms、`reduce_kernel` ×2（12.9 / 8.3 ms）、`triu_tril_kernel` 5.2 ms；decode 第 1 步：`indexSelectSmallIndex`（embedding 查表，单 token 时才用的 kernel）**75.6 ms**。
- `cuLibraryLoadData` ×14，**+48.7 ms**，全在 prefill，集中在首轮 +210 → +318 ms（第一个 GEMM 前后）。rep 1 起不再出现。
- `cuKernelGetFunction` +8.6 ms；`cudaMalloc` ×12 + `cudaFree` ×1，+6.5 ms（caching allocator 首次扩容）；`cuGetProcAddress` ×838、`cuInit` ×4、`cudaGetDriverEntryPoint` ×2 只在首轮出现，耗时可忽略。
- **API 之外的 CPU 时间 +249 ms**（prefill +219，decode 第 1 步 +42）。最大的一段空隙 **115.6 ms** 在首轮 +95 ms 处：最后一次 `cuInit` 之后、第一次 `cuLibraryLoadData` 之前，期间没有任何 CUDA API 调用；另一段 16.7 ms 在 `cudaGetDriverEntryPoint` 与 `cuInit` 之间。采集时 `--sample=none`，没有 CPU 调用栈，无法直接看到这段时间里 CPU 在执行什么。
- 首轮之前的 setup（建模型、`load_state_dict`）另有 1.7 s，其中 `cudaMemcpyAsync` ×147 共 630 ms（权重 H2D），不计入 cold_e2e。
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

---

## 附录 A：MoE 训练与效果（课程部分，不作为主线）

以下小节保留原编号，数据与结论不变，只是移出主线。

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

#### E7b slice 输出量级（未训练）

`slice_scale.py`（seed 42，top-2 of 4），`logs/slice_scale_1014.log`，`results/slice_scale.csv`，2026-10-14。输入 = Alpaca[200:300] 的 5711 个 token（与 E7 相同）。每个 token 算 ratio = ‖v‖ / ‖dense_ffn(x)‖ 和 cos(v, dense_ffn(x))，表中为 token 平均。

- **teacher-forced**：每层 FFN 的输入取自 dense 模型，各层互不影响。v 有三档：
  - `all_w1`：4 个 expert 全加、权重 1（应精确等于 dense）；
  - `topk_w1`：router 选中的 top-2、权重 1；
  - `actual`：top-2 × 归一化后的 router 权重（即 `MoEFeedForward.forward`；与其输出的最大差 ≤ 2.5e-2）。
- **free-running**：转换后的 MoE 模型自己跑，比较每层之后的残差流 h 与 dense 模型的。

| layer | all_w1 ratio / cos | topk_w1 ratio | topk_w1 cos | actual ratio | actual cos | 单个 expert ratio | top-1 权重 | free-run ‖h‖ 比 | free-run cos |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 / 1.0000 | 0.640 | 0.769 | 0.321 | 0.768 | 0.429 | 0.519 | 0.705 | 0.771 |
| 1 | 1.0000 / 1.0000 | 0.681 | 0.718 | 0.341 | 0.717 | 0.475 | 0.520 | 0.731 | 0.531 |
| 2 | 1.0000 / 1.0000 | 0.689 | 0.729 | 0.345 | 0.728 | 0.474 | 0.527 | 1.318 | 0.256 |
| 3 | 1.0000 / 1.0000 | 0.687 | 0.695 | 0.344 | 0.692 | 0.493 | 0.535 | 1.673 | 0.126 |
| 4 | 1.0000 / 1.0000 | 0.702 | 0.715 | 0.352 | 0.713 | 0.492 | 0.531 | 1.930 | 0.065 |
| 5 | 1.0000 / 1.0000 | 0.692 | 0.695 | 0.347 | 0.693 | 0.497 | 0.530 | 2.077 | 0.032 |
| 6 | 1.0000 / 1.0000 | 0.716 | 0.711 | 0.360 | 0.708 | 0.502 | 0.531 | 2.107 | 0.012 |
| 7 | 1.0000 / 1.0000 | 0.708 | 0.700 | 0.355 | 0.698 | 0.503 | 0.531 | 2.206 | 0.003 |
| 8 | 1.0000 / 1.0000 | 0.721 | 0.706 | 0.363 | 0.703 | 0.506 | 0.536 | 2.128 | 0.006 |
| 9 | 1.0000 / 1.0000 | 0.711 | 0.698 | 0.357 | 0.695 | 0.506 | 0.539 | 1.938 | −0.017 |
| 10 | 1.0000 / 1.0000 | 0.707 | 0.705 | 0.355 | 0.702 | 0.499 | 0.536 | 1.828 | −0.017 |
| 11 | 1.0000 / 1.0000 | 0.714 | 0.716 | 0.359 | 0.713 | 0.496 | 0.540 | 1.651 | 0.006 |
| 12 | 1.0000 / 1.0000 | 0.698 | 0.712 | 0.351 | 0.708 | 0.490 | 0.541 | 1.516 | 0.014 |
| 13 | 1.0000 / 1.0000 | 0.684 | 0.724 | 0.343 | 0.721 | 0.475 | 0.539 | 1.475 | −0.020 |
| 14 | 1.0000 / 1.0000 | 0.644 | 0.792 | 0.324 | 0.789 | 0.419 | 0.539 | 1.347 | −0.003 |
| 15 | 1.0000 / 1.0000 | 0.582 | 0.869 | 0.293 | 0.867 | 0.351 | 0.542 | 1.052 | 0.025 |

预测：

观察（数字层面，解释待写）：
- `all_w1` 每层 ratio / cos 都是 1.0000（最大绝对误差 ≤ 0.36，在 layer 1；其余层 ≤ 6e-2）→ 切分本身无损。
- `actual` 的范数只有 dense 的 **0.29–0.36**，约为 `topk_w1`（0.58–0.72）的一半；top-1 归一化权重 0.52–0.54，即两个被选 expert 的权重接近 0.5 / 0.5。
- 方向：top-2 输出与 dense 的 cos 0.69–0.87，乘不乘权重几乎不变。
- 单个 expert 的范数约为 dense 的 0.35–0.51。
- free-running：残差流 cos 从 layer 0 的 0.77 逐层下降，**layer 7 起 ≈ 0**；范数比在 layer 7 达到 2.2，到 layer 15 回到 1.05。即到中间层，MoE 模型的隐藏状态已与 dense 模型无关，与 E7 中 PPL 32 万、acc 0% 一致。

解释：

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

#### E8c 优化器更新被舍入掉了多少

`update_rounding.py --steps 20`，`logs/update_rounding_1014.log`（slice / slice-unfreeze）、`logs/update_rounding_lora_1014.log`（lora，A / B 分开统计），`results/update_rounding.csv`，2026-10-14。按 `train_moe.py` 的设置跑前 20 个训练样本：SGD momentum 0.9，全局 `clip_grad_norm_(1.0)`，lr_router 1e-3 / lr_expert 1e-4，max_seq_len 64。router 初始化与 E8 / E8b 的 run 不同（`train_moe.py` 在转换前消耗了 RNG）。

每步、每组参数统计：
- `grad=0`：clip 后梯度恰为 0 的元素比例（未被路由到的 expert 梯度为 None，SGD 跳过，也计入）；
- `lost`：预期更新 Δ = lr × (0.9·buf + grad)（fp32 计算）满足 |Δ| < ulp(w)/2 的比例 → 舍入回原值；
- `unch`：`optimizer.step()` 后权重逐位不变的比例（实测）；`unch|Δ≠0` 只在 Δ≠0 的元素里统计；
- `Δ/ulp`：|Δ| / ulp(w) 的分位数（随机子样本）。

| config | 组（dtype） | step | clip 前 grad norm | grad=0 | lost | unch | unch｜Δ≠0 | Δ/ulp p50 | p90 | p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| slice | router（fp16） | 1 | 5.31 | 29.70% | 87.34% | 87.34% | 81.99% | 3.7e-2 | 0.67 | 7.3 |
| | | 2 | 8.23 | 31.26% | 77.54% | 77.53% | 69.40% | 8.9e-2 | 1.35 | 13.6 |
| | | 5 | 7.16 | 31.26% | 66.22% | 66.21% | 58.41% | 0.19 | 2.82 | 26.3 |
| | | 10 | 13.84 | 25.02% | 58.22% | 58.22% | 52.25% | 0.31 | 4.32 | 42.9 |
| | | 20 | 3.16 | 20.32% | 48.80% | 48.80% | 44.45% | 0.53 | 5.29 | 52.2 |
| slice-unfreeze | router（fp16） | 1 | 54.44 | 29.80% | 98.63% | 98.63% | 98.05% | 3.6e-3 | 6.5e-2 | 0.71 |
| | | 2 | 65.50 | 31.28% | 96.68% | 96.67% | 95.47% | 1.0e-2 | 0.16 | 1.54 |
| | | 5 | 63.88 | 28.20% | 92.90% | 92.90% | 91.42% | 2.4e-2 | 0.34 | 3.44 |
| | | 10 | 81.00 | 23.56% | 89.23% | 89.23% | 88.11% | 3.8e-2 | 0.54 | 5.27 |
| | | 20 | 51.47 | 29.73% | 87.32% | 87.32% | 86.00% | 4.4e-2 | 0.67 | 6.48 |
| slice-unfreeze | expert（fp16，805 M） | 1 | 54.44 | 31.38% | **100.00%** | 100.00% | 100.00% | 1.7e-5 | 5.6e-4 | 6.9e-3 |
| | | 2 | 65.50 | 32.13% | 100.00% | 100.00% | 99.99% | 3.3e-5 | 9.9e-4 | 1.2e-2 |
| | | 5 | 63.88 | 29.86% | 99.98% | 99.98% | 99.97% | 7.0e-5 | 2.0e-3 | 2.4e-2 |
| | | 10 | 81.00 | 26.39% | 99.97% | 99.97% | 99.96% | 1.0e-4 | 2.9e-3 | 3.5e-2 |
| | | 20 | 51.47 | 31.15% | **99.95%** | 99.95% | 99.93% | 1.2e-4 | 3.6e-3 | 4.5e-2 |
| lora | router（fp32） | 1 | 2.62 | **100.00%** | 100.00% | 100.00% | – | 0 | 0 | 0 |
| | | 2 | 3.11 | 68.75% | 99.95% | 99.95% | 99.85% | 0 | 1.2e-3 | 2.4e-2 |
| | | 5 | 2.46 | 18.75% | 99.43% | 99.43% | 99.30% | 1.7e-3 | 2.4e-2 | 0.28 |
| | | 10 | 1.93 | 18.75% | 85.03% | 85.03% | 85.03% | 6.8e-2 | 0.77 | 7.74 |
| | | 20 | 2.46 | 0.00% | 65.58% | 65.58% | 65.58% | 0.24 | 2.52 | 26.3 |
| lora | lora_A（fp32） | 1 | 2.62 | 100.00% | 100.00% | 100.00% | – | 0 | 0 | 0 |
| | | 2 | 3.11 | 0.00% | 99.80% | 99.80% | 99.80% | 8.7e-4 | 9.6e-3 | 0.10 |
| | | 5 | 2.46 | 0.00% | 98.86% | 98.86% | 98.86% | 6.2e-3 | 5.5e-2 | 0.55 |
| | | 10 | 1.93 | 0.00% | 93.25% | 93.25% | 93.25% | 3.0e-2 | 0.33 | 3.40 |
| | | 20 | 2.46 | 0.00% | 82.87% | 82.87% | 82.87% | 8.9e-2 | 0.95 | 10.2 |
| lora | lora_B（fp32，初值 0） | 1–20 | | 0–1.56% | 0–1.56% | 0–1.56% | 0.00% | ≥9e5 | | |

预测：

观察（数字层面，解释待写）：
- 每一行 `lost` 与实测 `unch` 一致（差 ≤ 0.01 个百分点）→ "|Δ| < ulp/2 就被舍入掉"的判据成立。
- **slice-unfreeze 的 expert 权重：20 步里每步 ≥ 99.95% 的更新被舍入掉**，Δ 的中位数只有 ulp 的 1e-5 – 1e-4；p99 也只到 ulp 的 ~4%。
- slice-unfreeze 的 clip 前梯度范数 46–121，而 slice（只训 router）是 3.2–27（20 步全部范围；表中只列了 5 步）；同一个 router 在 slice-unfreeze 下丢失 87–99%，在 slice 下丢失 49–87%。
- 丢失比例随步数下降（momentum buffer 累积）：slice router 87% → 49%（20 步）。
- slice router 有 19–34% 的元素梯度恰为 0（slice-unfreeze 的 expert 25–35%）。
- LoRA（fp32）：第 1 步 router 和 `lora_A` 梯度 100% 为 0（此时 `lora_B` = 0）；`lora_B` 中 Δ≠0 的更新从未丢失（`lost` 偶尔 1.56% 全部是梯度恰为 0 的元素）；**`lora_A` 虽是 fp32，前 20 步仍有 83–99.8% 的更新被舍入掉**，比例逐步下降。

解释：

### E9 LoRA 有没有在学

| step | router grad norm | `lora_B` weight norm | loss |
|---:|---:|---:|---:|
| | | | |

- B=0 时 router 梯度的预测：
- PPL 差异是否超过噪声（不同 seed / eval 子集）：

#### E9a 训练 seed 方差（LoRA，6 epochs）

`scripts/run_lora_seeds.sh`（seed 1 / 2 / 3 并行，seed 42 稍后单独启动、与它们同时运行），`logs/{train,eval}_lora_ep6_s{1,2,3,42}.log`，2026-10-14。其余配置同 E8b 的 lora_ep6。seed 影响 router 与 LoRA A 的初始化（训练数据顺序固定，不 shuffle）。

| seed | epoch 1 → 6 loss | PPL | acc | acc − Dense |
|---:|---|---:|---:|---:|
| 1 | 2.3771 → 2.3717 → 2.3657 → 2.3591 → 2.3513 → 2.3419 | 9.36 | 52.32% | +0.95 pp |
| 2 | 2.3772 → 2.3720 → 2.3664 → 2.3600 → 2.3528 → 2.3444 | 9.37 | 52.23% | +0.86 pp |
| 3 | 2.3774 → 2.3726 → 2.3677 → 2.3622 → 2.3561 → 2.3488 | 9.41 | 52.11% | +0.74 pp |
| 42 | 2.3773 → 2.3723 → 2.3669 → 2.3607 → 2.3532 → 2.3439 | 9.36 | 52.20% | +0.83 pp |
| 均值 ± 标准差（n=4） | | 9.375 ± 0.024 | 52.22% ± 0.09 pp | +0.85 pp |

Dense = 9.66 / 51.37%（每个 eval log 的 [1/3] 都重新测得同样的值）。

- **并行不改变数值**：并行跑的 seed 42 与 E8b 中单独串行跑的 lora_ep6，6 个 epoch 的 loss 全精度逐位相同（`Training losses: [2.377277950391182, …, 2.3438975872141263]`），PPL / acc 也相同。并行只影响耗时（每 epoch 342–388 s vs 串行 233–239 s）。
- 4 个 seed 的 acc 都比 Dense 高 0.74–0.95 pp；seed 间标准差 0.09 pp。
- `eval_moe.py` 只打印 2 位小数的 PPL，所以 PPL 的 seed 方差分辨率有限。
- 这里只量了**训练 seed** 的方差；评估集固定为同一组 5711 个 token，评估集抽样带来的不确定性（换一批 held-out 样本会怎样）没有量。

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
