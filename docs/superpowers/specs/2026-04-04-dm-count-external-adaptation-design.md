# DM-Count External Adaptation Design

Date: 2026-04-04

## Goal

将 `DM-Count` 作为横向对比基线接入当前 `SecondChoice` 仓库，优先复用官方/高可信实现，再针对 `GWHD_2021`、`MTC`、`URC` 做最小本地适配，最终产出可用于论文主表或补充表的统一协议结果。

## Design Choice

本设计采用“外部仓库优先接入，再做本地适配”的路线，不直接把 `DM-Count` 手搓重写进 `pack/` 训练主干。

原因有三点：

- `DM-Count` 的核心创新不只是 backbone，还包括 distribution matching 的训练目标与优化逻辑。
- 如果一开始就把方法重写进 `pack/`，很容易做成 `DM-Count-style baseline`，而不是论文意义上的 `DM-Count`。
- 先保留官方实现边界，再逐步统一数据、指标和复杂度口径，更容易解释实验结果，也更利于论文写作。

## Source of Truth

当前优先目标仓库为官方实现：

- GitHub: `https://github.com/cvlab-stonybrook/DM-Count`
- Paper: `Distribution Matching for Crowd Counting`, NeurIPS 2020

本地参考论文文件：

- `ExperimentReference/NeurIPS-2020-distribution-matching-for-crowd-counting-Paper.pdf`

## Integration Boundary

`DM-Count` 的接入边界明确如下：

- 外部代码落在 `external/baselines/dm_count/`
- 当前仓库只新增必要的接入说明、适配脚本、结果汇总逻辑
- 第一阶段不修改 `pack/train.py` 主训练循环
- 第一阶段不把 `DM-Count` 强行注册为 `pack.models.build_model()` 的一部分

换句话说，`CSRNet` 采用的是“统一框架内本地实现”路线，而 `DM-Count` 采用的是“外部实现接入 + 本地协议适配”路线。这两者的论文标注必须区分。

## Target Outcome

第一阶段目标不是“完全统一代码结构”，而是拿到下面这些可核验产物：

- 可以定位并固定官方/高可信仓库版本
- 可以说明其依赖、训练入口、数据格式和验证入口
- 可以将 `GWHD_2021`、`MTC`、`URC` 映射到其数据协议
- 可以在本地输出 `MAE / MSE / MAPE`
- 可以在统一输入尺寸下统计 `Params / FLOPs`

只有完成这些项后，`DM-Count` 才能进入“正式落表候选”。

## Adaptation Strategy

### 1. Repository Intake

先将官方仓库克隆到：

- `external/baselines/dm_count/`

接入阶段要先回答四个问题：

- 它的训练主入口在哪里
- 它的数据集类和标注格式是什么
- 它的验证/测试输出是什么
- 它对 Python、PyTorch、CUDA 的依赖边界是什么

### 2. Data Adaptation

不优先改写核心训练逻辑，而是优先适配数据输入。

适配原则：

- `GWHD_2021` 仍使用框转点后的计数语义
- `MTC` 和 `URC` 保持当前项目的点监督协议
- 若 `DM-Count` 原仓库依赖 crowd counting 常见格式，则增加数据转换脚本或 dataset wrapper，而不是让三个数据集各自绕开统一语义

### 3. Evaluation Alignment

评价口径统一到当前项目论文协议：

- `MAE`
- `MSE`
- `MAPE`

如官方仓库默认只输出部分指标，需要在外部接入层补一个结果导出与汇总脚本，最终结果表按当前论文字段输出。

### 4. Complexity Measurement

复杂度不优先引用论文值，而优先本地统一统计。

统一口径：

- 输入：`1 x 3 x 1080 x 1920`
- `eval()` 模式
- 单尺度整图前向
- 输出：`Params` 与 `Approx FLOPs`

如果官方实现存在动态行为导致 FLOPs 统计不稳定，要在论文中明确标注“近似本地口径”。

## Non-Goals

本阶段明确不做以下事情：

- 不在第一轮把 `DM-Count` 整体迁入 `pack/`
- 不在没有官方实现核对前直接实现 `DM-Count-style baseline`
- 不把论文引用结果和本地复现结果混写到同一主表

## Risks

主要风险如下：

- 官方仓库年代较早，依赖或训练脚本可能需要环境修补
- 数据格式与当前农业计数协议可能不直接兼容
- 复杂度统计可能需要额外 wrapper
- 即使能跑通，若其训练配置和当前统一协议差异较大，最终论文中需要明确标注为 `adapted with method-specific training settings`

## Success Criteria

本设计成功的最低标准是：

- 能稳定接入官方/高可信 `DM-Count` 仓库
- 能让至少一个目标数据集完成真实训练/验证闭环
- 能输出统一指标与复杂度结果
- 能明确说明哪些结果属于 `reproduced under unified local protocol`，哪些结果属于 `adapted with method-specific training settings`

## Recommendation

执行顺序建议如下：

1. 固定官方仓库与版本
2. 审查训练入口、依赖与数据协议
3. 落地数据适配
4. 打通单数据集训练/验证
5. 补统一指标导出
6. 补复杂度统计
7. 再决定是否值得继续推进 `CAN` 或 `SASNet`
