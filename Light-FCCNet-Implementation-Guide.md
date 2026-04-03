# Light-FCCNet 实现说明书

基于论文图片第 32-47 页整理，目标是把当前论文内容转化为一份可直接指导复现与落地实现的工程文档。

本文档分为两部分：
- 第 1 部分：可复现实现清单
- 第 2 部分：代码模块设计

---

## 第 1 部分：可复现实现清单

### 1. 复现目标

本项目的复现目标分为两个层级：

- 最小可复现目标：
  能够在公开数据集上完成 Light-FCCNet 的训练、验证和测试，得到稳定收敛的计数模型，并复现论文中的模块组合关系。
- 结果接近论文目标：
  在 `GWHD_2021`、`MTC`、`URC` 三个数据集上，取得与论文表格同量级的 `MAE`、`MSE`、`MAPE`，并验证轻量化和消融趋势。

### 2. 论文中已经明确的信息

以下内容可视为当前实现的已知事实：

- 任务类型：
  大田作物计数，核心输出是密度图及由密度图积分得到的作物数量。
- 模型名称：
  `Light-FCCNet`
- 核心组成：
  - 轻量级特征金字塔聚合模块
  - 多重注意力模块
  - 面向计数任务的新损失函数
- 数据增强：
  使用随机裁剪。
- 轻量化卷积模块：
  使用通道拆分、逐点卷积、逐通道卷积和残差式融合。
- 金字塔聚合：
  存在多层下采样特征，文中提到 `1/4`、`1/16`、`1/64` 尺度，再统一上采样融合。
- 多重注意力模块：
  包含空间注意力和通道注意力。
- 损失函数：
  `L_FCC = (1 - α)(L2 + L_C) + αL_S`
- 实验平台：
  `Tesla V100`、`4 核 CPU`、`100GB` 磁盘。
- 训练设置：
  - 优化器：`Adam`
  - `batch size = 16`
  - 初始学习率：`1e-5`
  - 权重衰减：`1e-4`
  - 激活函数：`ReLU`
  - 参数初始化：标准差 `0.01` 的高斯初始化
- 评价指标：
  `MAE`、`MSE`、`MAPE`
- 对比模型：
  `CMTL`、`CSRNet`、`CAN`、`DM-Count`、`M-SegNet`、`SASNet`
- 数据集：
  `GWHD_2021`、`MTC`、`URC`
- 模型规模指标：
  - 参数量：`0.73M`
  - 计算量：`125.53G`

### 3. 论文中尚未明确但实现必需的信息

这些内容没有在当前 16 页图片里被完整写清，必须作为复现中的待确认项：

- 数据集划分方式：
  训练集、验证集、测试集如何划分。
- 输入图像尺寸：
  论文明确有随机裁剪，但没有在这一章里清楚写出训练输入尺寸。
- 密度图生成细节：
  高斯核大小、sigma 计算、边界处理。
- `LDMS` 的具体实现方式。
- 损失函数中的超参数：
  - `α`
  - `K`
  - 距离系数 `f`
  - `SSIM` 的窗口大小和常数项
- 是否使用学习率调度器。
- 总训练轮次或停止条件。
- 数据归一化参数：
  均值、方差、颜色空间是否变换。
- 数据加载方式：
  全图训练还是裁块训练。
- 推理时策略：
  直接整图、滑窗、缩放后整图，还是多尺度测试。

### 4. 默认假设策略

在没有补齐原文或源码前，建议采用以下默认假设推进工程实现：

- 输入尺寸：
  默认训练输入为 `512 x 512`。
  原因是大田计数任务通常需要兼顾局部密度与显存开销，且与文中多层下采样尺度更容易对齐。
- 图像通道：
  使用 RGB 三通道。
- 训练方式：
  裁块训练，整图验证。
- 数据归一化：
  默认采用 ImageNet 归一化，直到论文原始实现能提供更准确信息。
- 学习率策略：
  先使用固定学习率 `1e-5`，如果收敛慢，再加 `CosineAnnealingLR` 作为工程增强项。
- 训练轮次：
  默认 `200-300 epochs`，配合早停或最佳验证集保存。
- 损失超参数初值：
  - `α = 0.1`
  - `K = 3`
  - `f = 1.0`
  这些值应明确标注为工程假设，不可伪装成论文原值。

### 5. 数据集准备清单

#### 必需数据集

- `gwhd_2021(小麦)\gwhd_2021`
- `Maize Tassel Counting Dataset（玉米）\Maize Tassel Counting Dataset`
- `URC（水稻）`

#### 已确认的真实目录与标注结构

##### 1. `gwhd_2021(小麦)\gwhd_2021`

- 图像目录：
  `images/`
- 划分文件：
  - `competition_train.csv`
  - `competition_val.csv`
  - `competition_test.csv`
- 样本数量：
  - train: `3657`
  - val: `1476`
  - test: `1382`
- 标注格式：
  CSV 行格式包含：
  - `image_name`
  - `BoxesString`
  - `domain`
- `BoxesString` 内容示例：
  `99 692 160 764;641 27 697 115;935 978 1012 1020`
- 解释：
  每个框是 `xmin ymin xmax ymax`，多个框用分号分隔。
- 工程结论：
  该数据集原始监督是边界框，不是点标注。进入 Light-FCCNet 前必须先把框转为点，最直接方式是取框中心点。
- 参考代码核对结论：
  已检查本地参考代码 `fccnet_study/FCCNet/data/gwhd_dataset.py` 与 `fccnet_study/论文最终材料/FCCNet_Code_For_Review/.../New_Fccnet/pack/data/gwhd_dataset.py`，两者都明确将 `xmin ymin xmax ymax` 转为 bbox 中心点 `(x1+x2)/2, (y1+y2)/2`，因此设计文档中可直接将 `GWHD_2021` 的默认框转点策略写为中心点。

##### 2. `Maize Tassel Counting Dataset（玉米）\Maize Tassel Counting Dataset`

- 划分文件：
  - `train.txt`
  - `val.txt`
  - `test.txt`
- 样本数量：
  - train: `251`
  - val: `35`
  - test: `75`
- 图像与标注配对方式：
  `train.txt` 每行给出一对路径：
  `Jalaid2015_1/Images/xxx.jpg Jalaid2015_1/Annotations/xxx.mat`
- 文件数量：
  - `.jpg`: `361`
  - `.mat`: `361`
- 标注格式：
  `MATLAB 5.0 MAT-file`
- 已确认的内部结构：
  `.mat` 中存在一个 key：`annotation`
  其结构是一个带字段的 MATLAB struct，字段包括：
  - `filename`
  - `bndbox`
- 关键事实：
  虽然字段名叫 `bndbox`，但内部存储的不是传统 `xmin/ymin/xmax/ymax` 框，而是一个 `N x 2` 的二维坐标数组。
- 样例：
  `[[2395, 684], [2588, 516], ...]`
- 工程判断：
  该数据集本质上提供的是点标注，可直接作为计数点使用，不需要再从框转点。
- 工程结论：
  该数据集可以通过 `train/val/test.txt` 直接建立样本索引，读取 `.mat['annotation']['bndbox']` 后即可得到点坐标。

##### 3. `URC（水稻）`

- 划分文件：
  - `train.txt`
  - `val.txt`
  - `test.txt`
- 样本数量：
  - train: `197`
  - val: `49`
  - test: `109`
- 图像与标注配对方式：
  每行形如：
  `train/imgs_4/xxx.jpg train/dis_data_4/xxx.h5`
- 目录结构：
  - 图像：`train/imgs_4`、`test/imgs_4`
  - 标注：`train/dis_data_4`、`test/dis_data_4`
- 标注格式：
  标准 `HDF5` 文件
- 已确认的内部结构：
  每个 `.h5` 文件包含两个 key：
  - `coordinate`
  - `dis`
- 字段含义：
  - `coordinate`：形状为 `N x 2` 的浮点坐标数组，表示计数点位置。
  - `dis`：一个标量整数，目前样例均为 `27`。
- 样例：
  - `coordinate.shape = (779, 2)`
  - `dis = 27`
- 数据抽样结论：
  已抽查多个 `.h5`，`coordinate.shape` 常见为 `246` 到 `892` 个点，而 `dis` 仅为 `26/27/49/51` 这类小整数，因此 `dis` 不是植株计数。
- 参考代码核对结论：
  本地参考实现 `fccnet_study/FCCNet/data/urc_dataset.py` 曾将 `dis` 误读为 `count`，但同目录下的 `density_map.py` 实际并未使用 `gt_count` 归一化密度图，所以这个误用主要污染评估标签，不改变密度图生成。
  论文附带的参考实现 `fccnet_study/论文最终材料/FCCNet_Code_For_Review/.../New_Fccnet/pack/data/urc_dataset.py` 已明确写明：`URC's "dis" field is not plant count`，并改为优先使用 `gt`、其次 `density.sum()`、最后 `len(points)` 作为计数语义。
- 工程结论：
  该数据集已有现成划分和一一对应的图像-标签路径，可以直接从 `.h5['coordinate']` 读取点坐标；Light-FCCNet 复现时不应把 `dis` 当作计数监督，默认计数应以点数 `len(points)` 为准，若后续发现官方 `gt` 字段再优先切换。

#### 数据准备动作

- 下载原始图像和标注文件。
- 统一整理为相同目录结构。
- 检查标注格式是否一致。
- 将所有数据转换为统一的内部样本格式。
- 为每个数据集生成：
  - 图像路径
  - 标注点坐标或可转点的中间表示
  - 图像宽高
  - 样本 ID
  - 划分标签：`train`、`val`、`test`

#### 建议目录

```text
data/
  raw/
    GWHD_2021/
    MTC/
    URC/
  processed/
    GWHD_2021/
    MTC/
    URC/
  splits/
    gwhd_2021.json
    mtc.json
    urc.json
```

### 6. 标注与密度图生成清单

#### 输入标注

Light-FCCNet 本质上依赖点级或近点级计数监督，因此需要把三种不同来源的监督统一转换为点集表示。

#### 三类适配器建议

- `GWHD2021Adapter`
  输入：CSV 中的 bbox 字符串
  输出：点列表，建议先使用 bbox 中心点。
- `MTCAdapter`
  输入：`.mat`
  输出：点列表。
- `URCAdapter`
  输入：`.h5`
  输出：点列表和附加距离参数 `dis`。

#### 必须实现的预处理

- 将原始标注转为点坐标列表。
- 检查超出边界的点。
- 对重复点、非法点进行清洗。
- 生成密度图监督标签。

#### 推荐的密度图生成流程

- 对每个标注点生成局部高斯响应。
- 所有高斯响应叠加为密度图。
- 确保密度图积分近似等于作物个数。
- 将密度图尺寸与网络输出尺寸保持一致，或在损失中做对齐。

#### 待确认项

- 高斯 sigma 是否固定。
- sigma 是否与邻近点距离相关。
- 是否使用与 `LDMS` 一致的动态框尺度信息辅助训练。
- `URC` 中 `dis` 的真实语义是否对应某种额外尺度标签，目前已确认它不是计数，但论文正文尚未给出正式定义。

### 7. 训练配置清单

#### 论文已知配置

- 优化器：`Adam`
- `batch size = 16`
- 初始学习率：`1e-5`
- 权重衰减：`1e-4`
- 激活函数：`ReLU`
- 参数初始化：高斯初始化，标准差 `0.01`

#### 工程建议补充项

- 混合精度训练：建议开启。
- 梯度裁剪：建议支持可选开关。
- 断点恢复：必须支持。
- 验证频率：每个 epoch 验证一次。
- 模型保存：
  - 最新权重
  - 最优验证集权重
  - 配置快照

#### 训练日志

至少记录：

- 总损失
- `L2`
- `L_C`
- `L_S`
- 学习率
- 验证集 `MAE`
- 验证集 `MSE`
- 验证集 `MAPE`

### 8. 评估配置清单

#### 必须实现的指标

- `MAE`
- `MSE`
- `MAPE`

#### 评估流程

- 对验证集或测试集逐张推理。
- 将预测密度图积分为计数值。
- 与真实计数进行误差统计。
- 生成按数据集拆分的汇总表。

#### 推荐额外输出

- 预测可视化图。
- 预测密度图热力图。
- 不同模型或模块组合的对比图。

### 9. 复现风险清单

#### 高风险

- 损失函数中的 `α`、`K`、`f` 未明确。
- 输入尺寸未明确。
- 密度图生成细节未明确。
- `GWHD_2021` 框转点策略会直接影响训练标签。

#### 中风险

- `URC` 的 `dis` 虽已确认不是计数，但论文正文尚未正式定义其语义。
- 若直接照搬本地参考实现 `fccnet_study/FCCNet/data/urc_dataset.py` 的旧写法，会把 `dis` 误当 `count`，导致评估标签错误。
- 轻量金字塔模块具体通道数未完全写明。
- 空间注意力和通道注意力的实现细节需根据论文公式自行补全。
- 推理时是否整图或滑窗未明确。

#### 低风险

- 优化器、学习率、weight decay、评价指标已明确。
- 主体方法结构已经足够清楚。

### 10. 最小可跑通版本

如果目标是先拿到一个可训练、可验证、结构正确的版本，建议删减为：

- 数据集先只接入一个，例如 `GWHD_2021`
- 随机裁剪 + 密度图监督
- Backbone 只实现：
  - 轻量卷积模块
  - 四层金字塔聚合
  - 空间注意力
  - 通道注意力
  - 融合头
- 损失函数先实现完整 `L_FCC`
- 验证指标只跑 `MAE`、`MSE`、`MAPE`

这个版本的目标不是论文数值，而是：

- 模型能收敛
- 预测密度图有意义
- 参数量明显低于常见大模型

### 11. 接近论文结果版本

如果目标是尽量逼近论文结果，需要补齐：

- 三个数据集统一训练和评估管线
- 严格的数据划分
- 更接近论文的密度图生成逻辑
- `LDMS` 对应的标注尺度构造
- 消融实验开关
- 计算量与参数量统计脚本
- 可视化对比图生成脚本

### 12. 推荐执行顺序

建议按以下顺序推进：

1. 统一数据格式
2. 完成密度图标签生成
3. 搭建最小版本模型
4. 跑通单数据集训练
5. 实现完整损失函数
6. 实现完整多重注意力模块
7. 跑通三数据集评估
8. 增加消融实验配置
9. 统计参数量和 FLOPs
10. 输出可视化结果

---

## 第 2 部分：代码模块设计

### 1. 设计目标

代码设计目标不是机械抄论文，而是反推出一套清晰、可维护、可消融、可复现实验的工程结构。

整体原则：

- 模块边界清晰
- 配置与实现解耦
- 训练与评估逻辑独立
- 方便做消融实验

### 2. 项目目录建议

```text
light_fccnet/
  configs/
    base.yaml
    dataset_gwhd.yaml
    dataset_mtc.yaml
    dataset_urc.yaml
    model_light_fccnet.yaml
    ablation/
      baseline.yaml
      baseline_p1.yaml
      baseline_p2.yaml
      baseline_p3.yaml
      full.yaml  # corresponds to Baseline+P1+P2+P3

  data/
    datasets.py
    transforms.py
    density_map.py
    annotations.py
    splits.py

  models/
    light_fccnet.py
    blocks/
      lightweight_conv.py
      pyramid_fusion.py
      spatial_attention.py
      channel_attention.py
      multi_attention.py
      heads.py

  losses/
    light_fcc_loss.py
    count_loss.py
    ssim_loss.py
    pixel_loss.py

  engine/
    trainer.py
    evaluator.py
    inferencer.py
    hooks.py

  metrics/
    counting_metrics.py

  utils/
    config.py
    logger.py
    checkpoint.py
    seed.py
    flops.py
    visualize.py

  scripts/
    train.py
    evaluate.py
    infer.py
    export_results.py
```

### 3. 数据流总览

整体数据流建议如下：

1. 读取原始图像与标注
2. 应用随机裁剪和基础增强
3. 将点标注转换为密度图与计数标签
4. 图像输入 `LightFCCNet`
5. Backbone 输出多尺度特征
6. 多重注意力模块进行跨尺度融合
7. Head 输出预测密度图
8. 对预测密度图积分得到预测计数
9. 使用 `L2 + L_C + L_S` 计算损失
10. 验证阶段输出 `MAE`、`MSE`、`MAPE`

### 4. 顶层模型设计

建议顶层模型为 `LightFCCNet` 类，负责组织全部子模块。

#### 结构建议

```python
class LightFCCNet(nn.Module):
    def __init__(self, ...):
        self.backbone = PyramidFeatureAggregation(...)
        self.attention = MultiAttentionFusion(...)
        self.head = DensityHead(...)

    def forward(self, x):
        pyramid_feats = self.backbone(x)
        fused = self.attention(pyramid_feats)
        density = self.head(fused)
        return density
```

### 5. `LightweightConvBlock` 设计

#### 作用

实现论文中的轻量化卷积模块，是参数量压缩的基础单元。

#### 输入输出

- 输入：`[B, C, H, W]`
- 输出：`[B, C, H, W]`

#### 内部步骤

1. 沿通道维拆分为两部分：
   `F1`、`F2`
2. `F1` 经过 `1x1` pointwise conv，得到 `F3`
3. `F3` 经过 depthwise conv，得到 `F4`
4. `F3 + F2` 得到 `F5`
5. `F4` 与 `F5` 融合
6. 输出通道恢复到原始通道数

#### 推荐实现接口

```python
class LightweightConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, act_layer=nn.ReLU):
        ...

    def forward(self, x):
        ...
```

#### 工程注意点

- 输入通道数最好为偶数，方便拆分。
- depthwise conv 要设置 `groups = channels // 2`
- 最终融合方式建议显式实现，不要藏在过深的 Sequential 里。

### 6. `PyramidFeatureAggregation` 设计

#### 作用

构建四层金字塔特征提取与聚合主干，对应论文中的轻量级特征金字塔聚合模块。

#### 结构推断

- Stage 1：
  两个轻量化卷积模块，保持较高分辨率特征。
- Stage 2：
  一个轻量化卷积模块 + 一个 stride=2 的深度可分离卷积 + 一个轻量化卷积模块 + 残差连接。
- Stage 3：
  与 Stage 2 类似，进一步下采样。
- Stage 4：
  与 Stage 3 类似，形成最深层特征。

#### 输出

返回多尺度特征列表，例如：

```python
[feat_s1, feat_s2, feat_s3, feat_s4]
```

#### 建议接口

```python
class PyramidFeatureAggregation(nn.Module):
    def __init__(self, in_channels=3, stage_channels=(32, 64, 128, 256)):
        ...

    def forward(self, x):
        return [f1, f2, f3, f4]
```

#### 工程注意点

- 所有阶段输出要记录，供后续注意力融合使用。
- 深度可分离卷积建议独立封装。
- 由于论文未给出完整通道数，`stage_channels` 必须做成配置项。

### 7. `SpatialAttentionModule` 设计

#### 作用

建模像素空间上的长距离依赖，使局部作物区域与上下文关系更稳定。

#### 输入输出

- 输入：单个尺度特征 `[B, C, H, W]`
- 输出：同形状增强特征 `[B, C, H, W]`

#### 结构推断

论文中存在 `A`、`B`、`C`、`D` 的特征构造和矩阵乘法流程，因此建议实现为经典 non-local 风格：

1. 由输入特征生成 `B` 和 `C`
2. reshape 为 `[B, C, N]`
3. 计算空间相关性矩阵
4. 通过 `Softmax` 得到注意力图
5. 与 `D` 分支结合
6. 使用可学习系数与原始输入做残差融合

#### 建议接口

```python
class SpatialAttentionModule(nn.Module):
    def __init__(self, channels):
        ...

    def forward(self, x):
        ...
```

### 8. `ChannelAttentionModule` 设计

#### 作用

建模通道间的相关性，对不同尺度或不同通道的响应做加权。

#### 输入输出

- 输入：单尺度特征 `[B, C, H, W]`
- 输出：增强后特征 `[B, C, H, W]`

#### 结构推断

1. 先通过 `3x3` 卷积降低通道到 `C/2`
2. reshape 为二维矩阵
3. 计算通道相关性
4. 通过 `Softmax` 得到通道权重
5. 与原始特征做加权和残差融合

#### 建议接口

```python
class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, reduction=2):
        ...

    def forward(self, x):
        ...
```

### 9. `MultiAttentionFusion` 设计

#### 作用

将金字塔不同尺度特征统一到同一分辨率，再通过注意力机制完成融合。

#### 推荐流程

1. 接收多尺度特征列表
2. 将所有特征上采样到最高分辨率
3. 每个尺度特征先做通道对齐
4. 对每个尺度分别应用空间注意力和通道注意力
5. 采用求和或拼接后卷积的方式融合
6. 输出统一特征图

#### 建议接口

```python
class MultiAttentionFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        ...

    def forward(self, feats):
        ...
```

#### 推荐实现选择

优先用“拼接 + 卷积融合”，因为更直观，也更适合后续做消融。

### 10. `DensityHead` 设计

#### 作用

把融合后的特征映射为单通道密度图。

#### 推荐结构

- `3x3 conv + ReLU`
- `3x3 conv + ReLU`
- `1x1 conv -> 1 channel`
- 输出密度图经过 `ReLU` 或 `Softplus` 保证非负

#### 建议接口

```python
class DensityHead(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        ...

    def forward(self, x):
        ...
```

### 11. `LightFCCLoss` 设计

#### 组成

- 像素损失 `L2`
- 计数损失 `L_C`
- 结构相似损失 `L_S`

#### 推荐拆分

不要把全部逻辑写进一个文件里，建议拆成：

- `pixel_loss.py`
- `count_loss.py`
- `ssim_loss.py`
- `light_fcc_loss.py`

#### 顶层接口

```python
class LightFCCLoss(nn.Module):
    def __init__(self, alpha=0.1):
        ...

    def forward(self, pred_density, gt_density):
        ...
```

#### 内部逻辑

- `L2`：
  逐像素均方损失。
- `L_C`：
  对密度图积分后，与真实计数做误差约束。
- `L_S`：
  对预测密度图和真实密度图计算结构相似损失。

#### 返回值建议

返回总损失和明细字典：

```python
{
    "loss": total,
    "l2": l2,
    "count": lc,
    "ssim": ls,
}
```

### 12. 数据模块设计

#### `datasets.py`

负责：

- 读取样本路径
- 读取图像
- 读取点标注
- 应用增强
- 调用密度图生成器

#### `transforms.py`

负责：

- 随机裁剪
- 随机翻转
- 尺寸对齐
- 张量化和归一化

#### `density_map.py`

负责：

- 根据点标注生成密度图
- 支持固定 sigma 和动态 sigma 两种策略

### 13. 训练器设计

#### `trainer.py` 职责

- 构建 dataloader
- 执行 forward / backward
- 记录损失
- 保存 checkpoint
- 触发验证

#### 建议流程

```python
for epoch in range(num_epochs):
    train_one_epoch(...)
    val_metrics = evaluate(...)
    save_best(...)
```

#### 必须支持的功能

- AMP 混合精度
- Resume 训练
- 最优模型保存
- 配置快照保存

### 14. 评估器设计

#### `evaluator.py`

负责：

- 遍历验证集或测试集
- 计算预测计数
- 汇总 `MAE`、`MSE`、`MAPE`
- 输出结果表

#### 推荐返回结构

```python
{
    "mae": ...,
    "mse": ...,
    "mape": ...,
}
```

### 15. 消融实验设计

由于论文有完整消融，因此代码层面必须支持模块开关。

#### 建议通过配置控制

- `use_p1`
- `use_p2`
- `use_p3`

#### 配置示例

```yaml
model:
  use_p1: true
  use_p2: true
  use_p3: true
```

#### 组合要求

至少能跑出：

- `Baseline`
- `Baseline + P1`
- `Baseline + P2`
- `Baseline + P3`
- `Baseline + P1 + P2`
- `Baseline + P1 + P3`
- `Baseline + P2 + P3`
- `Baseline + P1 + P2 + P3`

### 16. 参数量和 FLOPs 统计设计

建议单独写 `utils/flops.py`，用固定输入大小统计模型复杂度。

#### 目标

- 输出总参数量
- 输出 FLOPs 或 MACs
- 与论文表格对齐

#### 注意

论文统计使用的是随机生成的 `1080 x 1920` 图像，因此如果想和论文表一致，应支持该输入尺寸的统计模式。

### 17. 推荐的实现顺序

#### 阶段 1：基础可运行

1. 数据读取与统一格式
2. 随机裁剪和密度图生成
3. `LightweightConvBlock`
4. `PyramidFeatureAggregation`
5. `DensityHead`
6. 基础训练和验证

#### 阶段 2：完整模型

1. `SpatialAttentionModule`
2. `ChannelAttentionModule`
3. `MultiAttentionFusion`
4. `LightFCCLoss`
5. 三数据集实验

#### 阶段 3：论文对齐

1. 消融实验开关
2. 参数量与 FLOPs 统计
3. 可视化脚本
4. 结果导出和汇总表

### 18. 最终落地判断

基于当前论文图片信息，Light-FCCNet 已经足够反推出一套完整的工程实现框架。最大的不确定性不在模型主结构，而在以下三类细节：

- 数据划分
- 密度图监督构造
- 损失函数超参数

因此，项目落地建议是：

- 先严格实现结构与训练框架
- 再通过实验回推缺失超参数
- 最后再尝试对齐论文表格数值

如果后续补到了更完整的原文、源码或附录，优先修订本文档中的“默认假设”部分，而不是推翻整体代码结构。
