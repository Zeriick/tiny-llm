**Day 7 split-K 性能诊断，2026-09-16**

结论：当前实现确实启用了 split-K，但在这台 Apple M4 的 Qwen3-4B、32-token projection replay 中没有显示净收益。问题不能归结为测试没有打开开关。调度仅覆盖 K/V 投影，新增的 partial buffer、归约 dispatch 和依赖同步需要计入成本。单独一个 K/V 算子的小幅收益没有转化为完整 projection replay 的收益。

环境：Apple M4，10 GPU cores，24 GB，macOS 27.0，MLX 0.32.0，mlx-lm 0.31.3，BF16，W4 group size 128，36 层。原始用户报告保留在仓库根目录的 `week2-day7-short-attribution.json`。

**调度覆盖范围**

下表用“归约长度”和“输出宽度”避免与源代码中的 N/K 命名混淆。这里行数 M=32。当前策略的 target_threadgroups=128、max_splits=8。

| 投影 | 归约长度 | 输出宽度 | 输出 tiles | 当前 splits | 临时 target=320 实验 splits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q | 2560 | 4096 | 128 | 1 | 2 |
| K | 2560 | 1024 | 32 | 4 | 8 |
| V | 2560 | 1024 | 32 | 4 | 8 |
| O | 4096 | 2560 | 80 | 1 | 4 |
| gate | 2560 | 9728 | 304 | 1 | 1 |
| up | 2560 | 9728 | 304 | 1 | 1 |
| down | 9728 | 2560 | 80 | 1 | 4 |

当前 K/V 分区边界为 0、640、1280、1920、2560，均为 128 的倍数。源码中 splits=1 直接选择 Day 5 unsplit kernel，不分配 partial buffer，不启动 reduction。Python 调用记录确认 36 层的七种投影均正确传递了 checkpoint 的开关；此记录是 Python 路由观测，并非 Metal GPU trace。

K/V 合计约占这七个层内投影的 5.19% 算术工作量。这个比例不是耗时比例，但说明 `projections` 占总 attribution 的 93.5% 并不代表其全部工作都受当前 split-K 策略影响。

每个 K/V 算子分配 4×32×1024×4 = 512 KiB FP32 partial buffer，产生一次 512 KiB 写入和一次 512 KiB 读取，以及一次新增 reduction dispatch。36 层合计新增 72 次 reduction，partial 的逻辑读写量为 72 MiB。这不是实测 DRAM 流量，因为缓存可能吸收部分读写。

**测量结果**

| 同轮比较 | unsplit 中位数 | split-K 中位数 | 延迟变化 |
| --- | ---: | ---: | ---: |
| 用户原始 projection replay，12 轮 | 83.531 ms | 83.876 ms | +0.413% |
| 当前策略，AB/BA 交替 48 轮 | 99.094 ms | 99.518 ms | +0.428% |
| 临时 target=320，AB/BA 交替 48 轮 | 140.797 ms | 152.698 ms | +8.452% |

后两项各自预热 6 轮，在同一进程中比较两个 checkpoint，使用同一组输入和同一份模型权重。每个 checkpoint 有 48 个原始样本，执行顺序每轮反转。计时包括 Python 构图和同步 `mx.eval`，与原 profiler 的计时范围一致。绝对延迟跨轮次明显变化，不能直接将 99 ms 与 140 ms 作策略比较；这里的比较对象都是每轮内交替测量的 unsplit/split-K。target=320 的 max_splits 仍为 8，未改变数学运算、FP32 partial 或 reduction kernel。

现有逐算子 benchmark 使用每个模型第一层的真实权重，12 轮预热、120 轮测量，在 simd、mlx、split-k 的六种排列中轮换。K 投影中位数为 308.458→296.604 μs，V 为 306.334→289.042 μs。使用同一个 unsplit kernel 的其他形状也出现约 1%～2% 的差异，因此不能把所有微小差异解释为 kernel 收益。

另测了将 36 层同类投影的不同真实权重一次提交的批次：当前策略 K 为 3.738→4.021 ms，V 为 3.660→4.184 ms。这个诊断批次允许独立投影并行执行，不保留实际层间依赖，不能当作模型前向性能；它说明单算子同步延迟的结果不能直接相加来预测整图效果。完整 projection replay 则保留原 profiling 脚本的投影依赖结构。两个 checkpoint 的 73 个 replay 输出均为有限值。

这些结果支持“当前 split 策略没有产生可确认的完整 replay 加速，简单提高 split 数量反而有风险”。没有单独的 GPU 时间戳或计数器，因此尚不能精确拆分 partial matmul、归约、同步与 CPU 开销，也不能声称已测得 occupancy 的变化。

**测试与 profiler 审查**

Day 7 的四个测试检查 checkpoint 配置、数值结果、边缘输出和较大 prefill 的精确 fallback，不检查运行时间。四个测试通过不能证明性能提升。源码检查确认 partial buffer、片段累加和最终归约均为 FP32；参考测试中关于“partial 为 BF16”的注释与当前实现不符。

原 profiler 确实执行了 `mx.eval(*outputs)`，不存在仅计时惰性构图而没有执行 GPU 的问题。但是，它只轮换同一个 checkpoint 内的 category 顺序，两个 checkpoint 整体顺序固定，并且仅保留中位数，未保存原始样本。这限制了对亚百分比差异的判断。

发现一个独立的元数据错误：`PREFILL_LOGITS="all"`，但 `KernelReplay.projections()` 实际使用 `hidden[:, -1:, :]` 计算 LM head。两个 checkpoint 都执行相同规则，因而此错误不能解释两者的相对性能差异；但报告不应被当成计算全部 token logits 的工作负载。此次诊断保留该行为以匹配原始测量，未修改 profiler。

**恢复与验证**

临时 target=320 修改已经按字节恢复为实验前的 C++ 文件，并重新运行 `pdm run build-ext`。随后 `pdm run test --week 2 --day 7`：4 passed in 3.62s。此次诊断未保留任何生产代码修改；原有未提交的 split-K 实现和测试改动保持原样。`restored-source.json` 保存恢复后三个扩展源文件的 SHA-256。

数据文件：`individual-operators.json`、`paired-current-policy.json`、`paired-target320-experiment.json`。paired 文件含原始样本、顺序、12 轮分块中位数、投影形状、分区边界和路由记录。初次 paired 报告没有源文件 SHA-256；后续临时实验报告与恢复记录包含这些哈希。

复测当前源码（输出路径必须不存在）：

```bash
pdm run build-ext
pdm run python benchmark_results/local-m4-day7-diagnosis-2026-09-16/diagnose.py /tmp/day7-paired-new.json
```

诊断脚本从当前 C++ 文件读取 target 和 cap，所以运行前应重建。它不修改源码、不调整 split 策略。临时 target=320 的报告是历史实验；再次复现该实验需要显式更改 target、重建、测量，之后恢复并重建。
