**Qwen3-4B / Apple M4：强制 split-count 扫描**

结论：32-token K/V 的 S=4 确有约 6% 的单算子收益，当前默认策略已经选中了它。部分 down 形状有约 1%～2% 的局部收益，但按形状组合后未得到明确的完整 prefill 加速。统一增加所有层内投影的 split 数，完整 prefill 稳定变慢。此次没有找到足够强的证据去替换生产策略。

本次固定 M=32/64/128，使用 Qwen3-4B 真实 BF16/W4 权重，量化组为 128。扫描 Q、K、V、O、gate、up、down 的 S=1/2/4/8。临时 C++ 调试开关仅修改现有 primitive 的 split 选择，不改变公开 API、loader、W4 解量化、fragment、FP32 partial 或 reduction 实现。所有配置使用同一份二进制；每个计时都包含 partial kernel、归约、临时 buffer，以及 Python 构图和同步 mx.eval。未测量 GPU 时间戳或 occupancy。

**单算子结果：相对 S=1 的配对延迟变化（负值更快）**

| Tokens | 投影 | 归约长度 × 输出宽度 | S=2 | S=4 | S=8 |
| ---: | --- | --- | ---: | ---: | ---: |
| 32 | Q | 2560 × 4096 | +4.08% | +4.10% | +13.74% |
| 32 | K/V | 2560 × 1024 | -3.91% | -5.99% | -3.44% |
| 32 | O | 4096 × 2560 | +1.53% | +0.61% | +1.77% |
| 32 | gate/up | 2560 × 9728 | +2.72% | +6.37% | +15.43% |
| 32 | down | 9728 × 2560 | +1.19% | -2.35% | -1.29% |
| 64 | Q | 2560 × 4096 | +2.23% | +6.81% | +15.34% |
| 64 | K/V | 2560 × 1024 | -0.81% | +0.60% | +5.43% |
| 64 | O | 4096 × 2560 | +0.27% | +2.38% | +6.95% |
| 64 | gate/up | 2560 × 9728 | +5.79% | +10.95% | +21.21% |
| 64 | down | 9728 × 2560 | -1.41% | -1.63% | +0.86% |
| 128 | Q | 2560 × 4096 | +5.22% | +8.65% | +19.74% |
| 128 | K/V | 2560 × 1024 | +3.06% | +4.45% | +15.01% |
| 128 | O | 4096 × 2560 | +2.34% | +5.44% | +10.96% |
| 128 | gate/up | 2560 × 9728 | +6.92% | +12.20% | +22.68% |
| 128 | down | 9728 × 2560 | +0.51% | +1.44% | +4.38% |

K/V 与 gate/up 分别合并相同形状的样本。其他表格项各自对应单个投影。每个具体投影、每个长度、每个 split 配置有 96 个样本（两轮各 48），每轮预热 24 次，并遍历四种 S 的全部 24 种排列。第二轮反转投影和长度顺序；共 8,064 个计时样本。

**独立的完整 prefill 验证**

每次建立新 KV cache，batch=1，offset=0，logits_to_keep=1；包含实际 attention、归一化、激活、投影与 KV 写入。没有启用可选 decode attention。统一 S 指每层的七个矩阵投影；最后一个 token 的 LM head 仍使用原 matvec 路径。六种配置轮换正反顺序，覆盖每个测量位置；长度同样正序/倒序各一轮。每轮预热 6 次、测量 12 次，每配置每长度合计 24 个样本，共 432 个计时样本。

| Tokens | S=1 中位数 ms | 统一 S=2 | 统一 S=4 | 统一 S=8 | 原策略 auto | 按形状候选 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 100.621 | +3.62% | +5.25% | +14.83% | +0.38% | -0.02% |
| 64 | 184.429 | +3.94% | +7.88% | +17.69% | +0.75% | -0.24% |
| 128 | 368.272 | +5.10% | +9.26% | +18.25% | -0.23% | -0.60% |

百分比均先在同一测量轮内计算 100×(候选耗时/S=1 耗时−1)，再取中位数；它不是分别汇总两个延迟中位数后再相除。原始分轮数据、顺序和样本保存在 JSON 中。

候选筛选规则：按相同矩阵形状合并微基准样本，选最小配对延迟；只有整体改善超过 1%、且两轮各自均改善，才将该 split 纳入完整模型实验。这是探索性筛选，不是统计显著性证明。

- 32 tokens：K/V 选 S=4，down 选 S=4，其余 S=1。
- 64 tokens：仅 down 选 S=4，其余 S=1。
- 128 tokens：全部 S=1，作为相同 kernel 的对照。

按形状候选在 32/64 tokens 分别只有 −0.021% / −0.236% 的变化；128-token 同核对照也出现 −0.603%。这些量级不足以认定模型级优化有效。与之相比，统一 S=8 在三个长度均慢约 15%～18%，方向在两轮中一致。

**验证与恢复**

- 84 个强制配置均检查 BF16 dtype、完整输出 shape 和全部元素有限性。每个配置抽查 8 个输出列的所有行，使用现有 Python dequantize_weights 方程，再做独立 NumPy FP64 点积，遍历完整归约；容差沿用本地 Day 5 Python equation 测试的 rtol=5e-3、atol=2e-3。
- 检查每个分区非空，所有边界为 128 的倍数，起点为 0、终点为完整归约长度，相邻半开区间连续。非均匀分组（例如 20 组拆 8 份）也包含在扫描中。
- S=1 的单算子输出与 Day 5 位级一致；完整模型三个长度的 S=1 logits 也与 Day 5 位级一致。全部 18 个完整 prefill 验证配置输出有限，dtype/shape 正确，在本次合成输入上 argmax 相同；这不替代更广泛的模型精度评估。
- C++ eval_gpu 日志逐条确认 84 次单算子强制分派，并按计数核对 3,780 次完整模型强制分派；日志仅在正确性检查时启用，计时期间关闭。日志是 host dispatch 记录，不是 GPU occupancy 证据。
- 临时 C++ hook 已移除，三个扩展源文件的 SHA-256 与扫描前完全一致；已重新运行 pdm run build-ext。
- 恢复后执行 pdm run test --week 2 --day 7：4 passed in 3.04s。生成的测试文件恢复为运行前内容。

**文件与复现**

- scan.py：扫描和验证脚本。
- operators.json / prefill.json：完整元数据、原始样本、执行顺序、候选规则与汇总。
- operators.dispatch.log / prefill.dispatch.log：非计时阶段的分派记录。
- dispatch-validation.json / restoration.json：分派计数核验与恢复记录。
- benchmark-hook.patch：仅供实验的 C++ 补丁，当前生产源码没有应用它。两个 C++ snapshot 保存扫描前后内容。

复现需要先检查源码是否仍匹配 original-source.json，再应用临时补丁、重建，运行两个阶段，最后移除补丁并重建。下面输出路径必须不存在。临时规则在 primitive 求值时读取，因此脚本始终等 mx.eval 完成后才切换，不得将不同规则的未求值图一起提交或用于 mx.compile。

```bash
git apply benchmark_results/local-m4-split-count-scan-2026-09-17/benchmark-hook.patch
pdm run build-ext
pdm run python benchmark_results/local-m4-split-count-scan-2026-09-17/scan.py --stage operators --output /tmp/split-operators-new.json
pdm run python benchmark_results/local-m4-split-count-scan-2026-09-17/scan.py --stage prefill --operators /tmp/split-operators-new.json --output /tmp/split-prefill-new.json
git apply -R benchmark_results/local-m4-split-count-scan-2026-09-17/benchmark-hook.patch
pdm run build-ext
pdm run test --week 2 --day 7
```

即使扫描失败，也需要移除临时补丁并重建；不要用它作为新的生产策略。
