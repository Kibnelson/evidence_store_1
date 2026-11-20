# KV delta ratio and speedup

## 1. Count what D needs

After Blend finishes, compare each final K and V row on P with the compatible row already on the decoder D.

| Final row | What D does |
| --- | --- |
| Same as D's retained row | Reuse it |
| Different from D's retained row | Receive the complete replacement row |
| No compatible row on D | Receive the complete current row |

Sum the **bytes** in each group. The diff ratio covers rows that have a base on D. The fraction sent also includes rows D is missing.

```text
Base coverage  = (same bytes + changed bytes) / all target KV bytes
Diff ratio     = changed bytes / (same bytes + changed bytes)
Delta KV bytes = changed bytes + bytes without a base
Fraction sent  = 1 − base coverage × (1 − diff ratio)
```

With no compatible base, send the full KV. The diff ratio is undefined in that case.

## 2. Turn bytes saved into time saved

Use the completed NIXL payload bytes for both modes to include full READ padding and delta packing. This estimate assumes both modes issue a READ with the same effective bandwidth and fixed cost.

```text
Transfer time saved = (full bytes − delta bytes) / bandwidth
Net time saved      = transfer time saved − extra work time
Predicted time      = full-mode time − net time saved
Predicted speedup   = full-mode time / predicted time
```

Extra work time is the added time on the path you measure, such as fingerprinting, packing, and reconstruction. The delta is faster when the transfer time saved exceeds that extra work time. Use the same timing boundary in both modes.

## Example

Suppose the full target is **100 MB**. D has **40 MB** that matches and **35 MB** that changed. The remaining **25 MB** has no base on D.

- Base coverage is **75%**. The diff ratio is **35 ÷ 75 = 46.7%**.
- The delta sends **35 + 25 = 60 MB**, assuming the layouts add no extra bytes.
- At **1 GB/s**, avoiding **40 MB** saves **40 ms**. If packing and reconstruction add **15 ms**, the net saving is **25 ms**.
- A **110 ms** full handoff becomes **85 ms**. The predicted speedup is **110 ÷ 85 = 1.29×**.

Use a matched full and delta replay to measure bandwidth and extra work, then check predicted times on a separate set of matched requests. [Detailed model and smoke checks](vllm-legacy-full-blend-tracing.md#predict-the-speedup-and-break-even-point)
