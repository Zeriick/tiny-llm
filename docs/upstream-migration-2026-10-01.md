# 2026-10-01 上游课程迁移记录

本次合并目标为 `skyzh/main` 的 `da3e841ce4fb70d341e6e799648cb703ca96a96e`。
迁移前的全部已跟踪学习代码保存在提交 `f10a8f2`，备份分支为
`codex/pre-upstream-20261001`。本机忽略的测试副本和当时的工作区补丁另存于
`.tiny-llm/migration-20261001/`，不纳入版本控制。

## 新旧课程对应关系

| 原有内容 | 新入口 | 迁移结果 |
| --- | --- | --- |
| KV cache | Week 2 Day 1：`kv-cache`、`capacity-cache` | 保留拼接缓存；新增有界容量、逻辑前缀、计数器、reset 和 rewind 支持 |
| W4 量化与 matvec | Week 2 Day 2：`quantized-matvec` | 复用已有 Python/C++/Metal 实现 |
| SIMD matrix prefill | Week 2 Day 3：`simd-matmul` | 复用已有 SIMD 内核，按新版顺序组合模型 |
| RMSNorm、RoPE、SwiGLU | Week 2 Day 4：`rmsnorm`、`rope`、`swiglu` | 保留已有融合算子；新增寄存器缓存 RMSNorm 留作练习 |
| 新 BF16 tiled prefill | Week 2 Day 5：`tiled-prefill`、`selected` | 新接口、校验、模型 dispatch 和 native scaffold 已接通；暂以已有 readable GQA 兼容 |
| 旧 decode-attention / Split-K 实验 | `tiny_llm.legacy_week2` | 保留旧模型及底层代码；新版 canonical selector 不再接受这些 checkpoint |
| Week 3 Day 1–3 | 原入口 | 保留连续批处理、chunked prefill、分页缓存与写入内核 |

默认 Week 2 checkpoint 现在是 `selected`。同名 `simd-matmul` 的累积功能也按新版课程调整：
它位于融合算子之前，因此不能把迁移前后的同名 checkpoint 直接当作性能等价的模型。
教材、参考实现和正式章节测试已采用上游版本。旧 Day 6/7 正式测试被移出新版目录。

## 已完成的兼容工作

- `TinyKvFullCache(capacity=None)` 同时支持原有拼接模式和有界模式。有界模式只返回逻辑前缀，
  overflow 在修改前报错，rewind/reset 保留物理容量供后续覆写。K、V 可保持各自 dtype，
  兼容原有 readable RoPE 使 K 升为 float32、V 保持 BF16 的路径。
- `create_kv_cache(capacity=...)`、生成器和单请求 benchmark 接通请求容量。
  生成器按 prompt token 数加 `max_tokens` 预留；Week 3 无参数工厂继续支持，异常清理也保留。
- 新增独立控制参数和 `implementation_backends`。单请求 benchmark JSON 记录实际实现类别，
  避免把选择了新 checkpoint 误认为已经运行新内核。
- attention 兼容层支持 GQA，以及 additive mask 下全屏蔽行输出零。
- 新 `_dense_attention_prefill_mma` 私有绑定和 `Week2DensePrefillMMA` 已迁移；
  native 入口仍明确报未实现，Python 兼容入口不会误调用它。

## 留给学习者的核心任务

### 1. 新 Week 2 Day 4：register-cached RMSNorm

目前 `FastRMSNorm` 仍调用已有的 two-pass kernel。`use_register_cached_rms_norm=True`
表达新版模型选择，但不代表这一优化已经完成；`implementation_backends["rmsnorm"]`
会显示 `course_two_pass`，`dispatch_counts["register_cached"]` 保持零。

阅读 `book/src/week2-04-fused-model-kernels.md`，在
`src/extensions/src/week2_kernels.metal` 中实现 `week2_rms_norm_register_cached`，
并在 C++ `Week2RMSNorm::eval_gpu` 中增加适用行宽的调度与宽行 fallback。
重点理解寄存器保存输入、归约、二次读内存的消除，以及寄存器占用与 occupancy 的取舍。
完成后同步更新 Python 计数和 backend 标记，不能只凭数值测试通过判断优化完成。

### 2. 新 Week 2 Day 5：BF16 tiled prefill attention

阅读 `book/src/week2-05-tiled-prefill-attention.md`。入口是
`src/tiny_llm/week2_kernels.py` 的 `dense_prefill_attention_mma`，native scaffold 在
`src/extensions/src/week2_kernels.cpp` 和 `tiny_llm_ext.h`。
当前 `DENSE_PREFILL_MMA_IMPLEMENTED = False`，因此复用我们已有的 readable GQA；
`implementation_backends["prefill_attention"]` 显示 `readable_compatibility`。

实现 BF16、D=128、L>=9 的 `week2_dense_prefill_mma_bf16_d128`，重点学习
Q/K 分块、SIMD-group matrix、online softmax、GQA head 映射、非零历史前缀的因果 mask，
以及全屏蔽行处理。完成 native wrapper、GPU 调度和 Metal 内核后，再启用上述常量，
验证真正的 native 路径并做性能对照。

旧 `flash_attention.cpp/.metal` 实验文件仍保留，但未加入当前 CMake/binding，
不能直接视作新 BF16 内核已经实现。

### 3. Week 3 的当前进度

Day 1–3 的现有实现已保留且章节检查通过。Day 2 的 chunk 边界、历史前缀 mask、
decode 在长 prompt chunk 间推进、cache materialize 和异常释放均已覆盖。
下一步仍值得自己做的是 chunk 大小与 TTFT/TPOT 的实测，以及后续直接读取页表的
paged attention / FlashAttention。此前未完成的后续 native 练习没有用参考答案补齐。

上游将旧 Week 3 性能记录标注为五天版 Week 2 改版前的历史数据。
保留原数据，新测量另存文件，并重新跑对应参考基线；本次迁移没有运行长性能 benchmark。

## 访问旧实现

```python
from tiny_llm.legacy_week2 import Qwen3ModelWeek2 as LegacyQwen3ModelWeek2

model = LegacyQwen3ModelWeek2(mlx_model, checkpoint="split-k")
```

`src/tiny_llm/legacy_week2.py` 保留迁移前的模型实现，依赖共享的兼容算子。
旧 decode-attention 的 Python 路径现在也使用 readable GQA；原 native stub 未冒充已完成内核。
`tests_legacy/week2/` 保存旧七天版测试，独立于 `pdm run test` 的默认 `tests/` 目录。
其中旧 CLI/selector 断言可能与新正式入口不再一致，整个归档不作为新版验收标准。
旧 Day 7 和原自编 Split-K kernel 测试已单独验证。

## 本次验证

- 学习扩展和参考扩展均重新构建成功。
- 新 Week 2 Day 1–5：6 + 14 + 6 + 7 + 10 = **43 passed**。
- Week 3 Day 1–3：15 + 12 + 39 = **66 passed**。
- 新增迁移回归 3 项、原 Split-K kernel 34 项、归档 Day 7 四项：**41 passed**。
  该次命令还重复运行了 Day 1 六项，共输出 47 passed。
- 总计 **150 项不同测试通过**。未运行全课程测试或长时间性能测试。

复现补充检查：

```bash
pdm run test --week 2 --day 1 -- tests/test_upstream_migration.py tests/test_week_2_split_k_kernels.py tests_legacy/week2/test_week_2_day_7.py -q
```

章节检查通过说明兼容路径的数值与接口成立，不证明新增的寄存器缓存或 tiled GPU 优化完成。
