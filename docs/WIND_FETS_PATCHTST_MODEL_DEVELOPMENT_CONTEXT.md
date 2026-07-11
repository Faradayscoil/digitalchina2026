# HR-MoE FeTS-PatchTST 风电预测模型开发上下文

> 更新日期：2026-07-10
> 项目根目录：`/mnt/d/Python/myprojects/digitalchina2026`
> 当前开发分支：`dev-FeTS-PatchTST`
> 当前代码模型名：`fets_patchtst`
> 当前架构版本：`fets_patchtst_horizon_regime_moe_v5ab`
> 建议论文模型名：**HR-MoE FeTS-PatchTST**
> 英文全称：**Horizon-Regime Mixture-of-Experts Feature-enhanced PatchTST**

本文用于在新对话中快速恢复 FeTS-PatchTST 阶段的研究上下文，集中记录任务、
数据边界、模型结构、结构演进、实验结论、论文归因边界和工程文件位置。
本文不记录硬件、命令、续训、单模型运行和故障排查等执行层内容。

`docs/WIND_FORECASTING_FETS_PROJECT_CONTEXT.md` 记录了 FeTS 开始前的历史基线、
DeepSeek/tuned PatchTST 路线和最初 FeTS 计划。该文档中“下一阶段暂不加入多尺度、
persistence 和 gating”等内容已经被后续实验推进所取代。涉及**当前模型**时，
应以本文、当前代码和当前 artifact 为准；涉及更早历史实验时，再查阅旧文档。

## 1. 当前结论速览

当前模型不是纯 FeTS，也不是单一 PatchTST，而是：

```text
原生长尺度 PatchTST 锚点
+ 中尺度 PatchTST 残差专家
+ 局部 FeTS 残差专家
+ 持续性专家
+ 逐样本、逐预测步长的 Horizon-Regime softmax router
```

当前测试结果的严谨结论是：

- 在当前统一汇总的九个模型中，FeTS-PatchTST 在五个场站的 MAE、MSE、
  RMSE、sMAPE、R²、NMAE 和 NRMSE 上均为第一。
- 八类指标乘五个场站共 40 项比较中，FeTS-PatchTST 获得 36 项第一。
- 唯一没有全胜的指标是 MAPE：五个场站中仅 `...5971` 为第一，其余四站
  均由当前 PatchTST 基线取得更低 MAPE。
- 五站平均 NRMSE 为 `0.116478`，当前 PatchTST 基线为 `0.120938`。
- 当前结果证明的是**完整组合模型的经验性能**，尚不能证明每个新增模块都
  独立有效。
- 当前 router 在多数场站高度偏向 persistence 专家，因此模型已有较强性能，
  但 FeTS、尺度专家和 router 的独立论文贡献仍必须通过消融确认。
- 测试标签没有通过代码进入训练，但本项目曾观察测试表现后继续选择结构，
  所以当前测试集已承担过模型选择反馈，不应再被称为严格最终盲测集。

## 2. 任务定义与数据边界

项目面向五个风电场的超短期功率预测：

| 项目 | 当前设定 |
| --- | --- |
| 时间分辨率 | 15 分钟 |
| 历史窗口 | 96 点，即过去 24 小时 |
| 预测窗口 | 16 点，即未来 4 小时 |
| 目标变量 | `功率` |
| 建模方式 | 五场站分别训练独立模型 |
| 未来 NWP | 不使用 |
| 模型输入 | 历史气象、时间/物理派生特征和历史功率 |
| 模型输出 | 一次性直接输出未来 16 点功率 |

五个场站为：

```text
4081950112845135880
4081950112845135895
4081950112845135971
4081950112845135975
4081950112845136015
```

数据文件位于：

```text
./wind_split/wind_train_<farm_id>.csv
./wind_split/wind_test_<farm_id>.csv
```

测试采用滚动式评价：输入张量只包含预测起点前 96 个时间位置。测试段内已经
发生的真实功率可以成为后续窗口的历史输入，但同一窗口未来 16 点真实功率只
用于 `y_true`、指标和可视化，不进入输入。测试文件最前面的 96 点作为初始
历史上下文，正式预测从其后开始。

当前预测预处理还不是严格在线因果流程：测试文件先整体执行双向时间插值及
前后填充，随后才把功率输入替换为只做前向填充的历史真实功率。因此未来功率
标签不会进入输入，但历史窗口中的气象缺口可能使用预测起点之后的气象观测值
插补。严格在线评价应按预测起点进行因果气象插补。

必须区分两件事：

1. **没有代码级测试标签泄漏**：FeTS 训练只读取训练文件，预测窗口未来标签
   不会进入模型输入。
2. **存在研究流程层面的测试反馈**：结构曾根据五个测试场站的结果继续调整，
   因而当前测试结果适合阶段性比较，不足以充当最终盲测证据。

上述第 1 点只针对未来功率标签；它不抵消气象双向插值带来的严格因果性问题。

## 3. 公共预处理与公平比较口径

FeTS-PatchTST 直接复用 `wind_dl_model_train.py` 中的：

```text
load_and_preprocess
build_scaled_arrays
make_window_dataset
```

原生 PatchTST、FeTS-PatchTST 和七个其它深度学习基线因此共享以下流程：

1. 时间排序、去重并恢复完整 15 分钟索引。
2. 非数值内容转缺失值，并修复少量气压/湿度疑似互换。
3. 按物理范围清洗风速、温度、气压、湿度和功率。
4. 从 `装机` 非零值中位数提取场站容量，容量列不作为输入。
5. 功率裁剪至 `[0, capacity]`。
6. 风向转换为正余弦特征。
7. 插值并前后填充缺失值。
8. 添加日内、星期、年内、月份周期特征。
9. 添加风速平方、立方，以及轮毂高度风速差和比值等物理特征。
10. 将历史功率追加为模型输入通道。
11. 输入和目标分别使用每个训练场站自己的 `StandardScaler`。
12. 构造全部滑动窗口后，按时间顺序将最后 15% 窗口作为验证集；训练窗口
    可以打乱，验证窗口不打乱。

一个需要在论文方法中修正或披露的细节是：当前 scaler 先在完整训练文件上
拟合，再划分训练/验证窗口，因此验证时间段参与了归一化统计量估计。测试文件
完全不参与 scaler 拟合。这不构成测试泄漏，但严格时序验证应只用训练子段拟合
scaler，再应用到验证子段。

## 4. 当前模型命名与代码事实

推荐论文名称为：

```text
HR-MoE FeTS-PatchTST
Horizon-Regime Mixture-of-Experts Feature-enhanced PatchTST
面向风电超短期预测的预测步长—工况感知多专家特征增强 PatchTST
```

名称对应关系如下：

- `HR`：router 同时依赖历史工况表示和未来 horizon embedding。
- `MoE`：融合 long、mid、short、persistence 四个完整预测候选。
- `FeTS`：局部分支采用 AdaFE 与 DSFFN 的特征感知机制。
- `PatchTST`：长尺度安全锚点及中尺度专家使用本工程 PatchTST 主干。

代码与模型文件仍使用 `fets_patchtst`，当前 artifact 必须记录架构版本：

```text
fets_patchtst_horizon_regime_moe_v5ab
```

本文所称“原生 PatchTST”是指本工程 `wind_dl_model_train.py` 中的正式基线，
尤其是 `build_patchtst_model()` 的拓扑；它不是 PatchTST 官方仓库的逐行复刻。

## 5. 当前模型完整结构

### 5.1 总体前向图

```text
历史输入 X [B, 96, C]
│
├─ Long expert
│    Patch(16, 8) → 3-layer PatchTST → y_long
│
├─ Mid expert
│    Patch(8, 4) → 2-layer PatchTST → zero-init Δ_mid
│    → y_mid = y_long + Δ_mid
│
├─ Short FeTS expert
│    Patch(4, 2) → channel embedding
│    → AdaFE + DSFFN + LayerScale
│    → power Query / non-power-feature Key-Value cross-attention
│    → 2-layer temporal Transformer
│    → fuse long-scale context → zero-init Δ_short
│    → y_short = y_long + Δ_short
│
├─ Persistence expert
│    最后一个历史功率 → 标准化空间对齐 → 重复 16 步 → y_persist
│
└─ Horizon-Regime Router
     历史多尺度上下文 + 最后时刻全部已观测特征 + horizon embedding
     → [B, 16, 4] softmax weights
     → 四专家逐样本、逐 horizon 凸融合 → y_hat [B, 16]
```

最终输出为：

\[
\hat y_h=\sum_{e\in\{long,mid,short,persistence\}}
\alpha_{h,e}\hat y_{h,e},\qquad
\alpha_{h,e}\ge 0,\quad \sum_e\alpha_{h,e}=1.
\]

中、短尺度候选均锚定长尺度输出：

\[
\hat y_{mid}=\hat y_{long}+\Delta_{mid},\qquad
\hat y_{short}=\hat y_{long}+\Delta_{short}.
\]

因此四个专家不是四套彼此独立的完整网络；mid 和 short 是以 long 为安全基线
的修正专家。

### 5.2 多尺度配置

| 分支 | Patch / stride | 实际时间覆盖 / 步长 | Patch 数 | 编码层数 | 主要目标 |
| --- | ---: | ---: | ---: | ---: | --- |
| Long | 16 / 8 | 4 小时 / 2 小时 | 12 | 3 | 日内背景、慢趋势和稳定基线 |
| Mid | 8 / 4 | 2 小时 / 1 小时 | 24 | 2 | 填补长短尺度间的状态变化 |
| Short FeTS | 4 / 2 | 1 小时 / 30 分钟 | 48 | 2 | 15 分钟级局部细节、爬坡和骤降 |

公共主干宽度为 `d_model=64`，注意力头数为 4，Transformer FFN 宽度为
128。基础 dropout 为 `0.15`，预测 head dropout 为 `0.2`。

### 5.3 Long expert：完整保留本工程 PatchTST 基线拓扑

长分支位于 `build_fets_patchtst_model()` 的 `long_*` 路径，结构为：

```text
PatchExtract(16, 8)
→ Dense patch projection
→ MergeChannels
→ learnable position embedding
→ 3 × Transformer encoder
→ RestoreChannels
→ 目标功率通道表示 flatten
+ 全通道全局平均上下文
→ MLP head
→ baseline_forecast
```

它从 patch 切分到 `baseline_forecast` 的前向拓扑与
`wind_dl_model_train.py::build_patchtst_model()` 一致，是新结构的稳定锚点。

但“拓扑一致”不等于“参数等同于独立原生 PatchTST”。长分支表示还会送入局部
head 和 router，因此在混合模型中会接收来自最终融合目标的联合梯度。独立训练
的 PatchTST 才是严格对照。

### 5.4 Mid expert：中尺度 PatchTST 残差候选

中尺度分支使用 `patch_len=8`、`stride=4` 和两层 Transformer。其目标不是
替代长分支，而是学习约 1～2 小时状态变化：

```text
mid representation → MLP → zero-initialized Δ_mid → y_long + Δ_mid
```

残差输出层权重和偏置均零初始化，使训练开始时中尺度候选与长尺度预测一致，
避免随机修正立即破坏基线。

### 5.5 Short expert：局部 FeTS 分支

#### FeTS patch 提取

`FeTSPatchExtract` 按通道独立切分短 patch，并使用尾值复制填充，输出形状为：

```text
[batch, channel, patch_num, patch_len]
```

#### Channel identity embedding

`ChannelIdentityEmbedding` 给每个输入通道添加独立可学习向量，保留功率、不同
高度风速、温度等变量的身份。当前实现是 channel embedding，不是
feature-group embedding。

#### AdaFE

`FourierPolynomialMask` 与 `AdaptiveFeatureExtraction` 实现 FeTS 风格的
Adaptive Feature Extraction：

- Fourier degree 与 polynomial degree 均为 2；
- 每个 patch 表示独立生成动态 mask；
- 用该 patch mask 的均值作为阈值；
- 前向使用二值激活，反向使用 straight-through estimator；
- 在 `d_model` 潜在表示维上使用长度 5、padding 2 的局部滑窗聚合。

必须准确表述：AdaFE 选择的是**投影后的潜在表示维**，不是直接从原始气象列
中做硬特征筛选。不能仅凭该模块宣称模型已经识别出某个原始传感器最重要。

#### DSFFN

`DualScaleFeedForward` 同时构造：

- 逐 patch 的 point-wise Conv1D 局部表示；
- 沿 patch 维求平均得到的全局上下文；
- 二者拼接、投影后回到 `d_model`。

`ffn_ratio=2`，内部宽度为 128。它在特征块内部融合局部关键变化和全局趋势，
与在输出端混合几条完整预测轨迹不是同一种机制。

#### LayerScale residual

`LayerScaleFeTSFeatureBlock` 使用：

\[
x_{out}=x+\gamma\cdot DSFFN(AdaFE(x)),
\]

其中 `γ` 是逐表示维可学习参数，初值为 `1e-3`。其作用是让 FeTS 修正在训练
早期小幅进入主表示，减少深层适配器突然覆盖原 patch 表示的风险。

#### 功率 Query—非功率特征 Key/Value 定向交叉注意力

`TargetWeatherCrossAttention` 沿用了早期类名，但其实际 Key/Value 不只包括
气象变量，而是包括全部非功率输入通道：气象、时间周期和物理派生特征。它将：

```text
功率 token → Query
所有非功率特征 token → Key / Value
```

注意力在每个短 patch 内执行，输出保留非功率特征增强后的功率 token，再沿
48 个短 patch 执行两层时间 Transformer。该结构替代早期全通道自注意力，
避免所有非功率 token 彼此进行无约束交互，并明确让这些历史特征服务于功率
表示。

#### 长尺度上下文注入局部 head

局部预测 head 拼接：

- 最近局部 token；
- 局部全局平均表示；
- 由长尺度目标表示、全通道上下文和长尺度预测共同投影得到的上下文。

随后学习零初始化 `Δ_short`，形成 `y_long + Δ_short`。这一步用于协调长短
分支，减少局部 head 在缺少总体趋势时产生方向相反的修正。

### 5.6 Persistence expert

`PersistenceForecast` 读取最后一个历史功率，将输入功率的标准分数仿射转换到
目标 scaler 空间，再重复为未来 16 步。该专家为平稳、慢变化和持续零功率
工况提供低方差候选，但它不能识别尚未在历史中出现的未来停机或来风变化。

### 5.7 Horizon-Regime Router 与凸融合

`HorizonRegimeRouter` 的输入包括：

- 长尺度上下文；
- 中尺度上下文；
- 局部最近/全局上下文；
- 最后时刻全部已观测输入特征。

内部结构为：

```text
LayerNorm
→ Dense(64, GELU)
→ Dropout(0.1)
→ 复制到 16 个 horizon
+ 16 维 horizon embedding
→ Dense(64, GELU)
→ Dense(4)
→ softmax
```

输出形状为 `[B, 16, 4]`，允许同一样本的近端和远端预测使用不同专家比例。
router 不读取未来真实功率。

输出层 kernel 零初始化，初始 bias 为 `(2, 0, 0, -2)`，对应初始权重约为：

```text
long 77.58% / mid 10.50% / short 10.50% / persistence 1.42%
```

结合中、短残差头零初始化，模型初始输出接近 long expert，随后再学习工况和
horizon 条件化的偏离。最终由 `ExpertConvexFusion` 做非负、和为 1 的凸融合。

### 5.8 损失和训练比较口径

当前 FeTS-PatchTST 与本工程原生 PatchTST 均使用：

```text
Adam
learning_rate = 5e-4
clipnorm = 1.0
Huber(delta = 1.0)
监控 val_loss
```

基础 loss 没有改成 RMSE、NRMSE 或复合目标，因此可以排除损失定义差异直接
偏向某一评价指标。但 FeTS 脚本没有固定模型初始化随机种子，现有活动模型的
训练配置也并非完全一致，所以当前横向结果还不是严格的单变量结构消融。

### 5.9 当前验证曲线的含义

当前五个场站的训练历史均表现为 `val_loss` 在前 5～7 个 epoch 附近达到最低，
随后回升，而训练 loss 继续保持更低。它说明最佳点之后出现了明确的泛化间隙，
主要应解释为模型容量较大后的过拟合，并可能叠加训练前段与时间后段验证集的
分布漂移；这不是输入张量形状或特征工程不一致导致的报错。

当前 checkpoint、学习率调整和早停都监控同一个 Huber `val_loss`，并恢复其
最佳权重。因此最终保存模型不是最后一个 epoch，但“较早达到最佳点”仍提示
多分支和 router 的容量、专家塌缩及跨时间泛化需要在后续消融中单独检查。

## 6. 结构来源及与原论文的差异

| 当前组成 | 主要来源 | 当前工程保留内容 | 当前工程的关键差异 |
| --- | --- | --- | --- |
| 长尺度主干 | PatchTST | patch 化、位置编码、通道独立共享 Transformer、直接多步 head | 指本工程 Keras 基线；`MergeChannels` 将通道折入 batch 维，跨通道信息只在预测 head 池化时融合，并非官方逐行复刻 |
| AdaFE | FeTS | Fourier/polynomial mask、动态激活、局部聚合 | Keras 重写；per-patch 阈值；STE 保持 mask 可训练；只用于短分支 |
| DSFFN | FeTS | 局部 point-wise 表示与 patch 全局均值融合 | 只用于短分支，后面仍接时间 Transformer |
| LayerScale | CaiT/LayerScale 思想 | 小初值可学习残差缩放 | 作为 FeTS adapter 稳定器，不是完整 CaiT block |
| Horizon-Regime router | 本工程自定义 MoE 结构 | 样本/步长条件化的稠密 softmax 融合 | 不对应某篇论文的原始 router |
| 频域 MoE 思想参考 | M2FMoE | 动态门控与不同模式专家的研究动机 | 当前没有频谱输入、频域专家或原始门控 |
| 稀疏尺度思想参考 | SSformer | 稀疏尺度选择、尺度特定卷积和双向尺度交互的研究动机 | 当前没有复现这些结构 |
| Persistence expert | 风电持续性先验 | 最后历史功率重复 16 步 | 作为可学习路由的候选，不是独立后处理规则 |

### 6.1 与 PatchTST 的关系

保留：

- 96→16 直接预测任务；
- 长尺度 patch 16/8；
- `d_model=64`、4 头、3 层编码器、FFN 128；
- 目标通道 flatten 与全通道全局上下文 head；
- 相同公共预处理、Adam 和 Huber loss。

新增：

- 8/4 中尺度 PatchTST；
- 4/2 局部 FeTS；
- channel embedding；
- 功率 Query / 非功率特征 Key-Value 注意力；
- 长尺度上下文注入局部 head；
- persistence 专家；
- horizon-regime 动态 MoE。

### 6.2 与原生 FeTS 的关系

保留的 FeTS 核心是 patch 提取、Fourier/polynomial mask、AdaFE、DSFFN 和
residual connection。

主要差异是：

- 官方 FeTS 为 PyTorch，当前工程为 TensorFlow/Keras 重写；
- 官方 FeTS 以 FeTS block 和 flatten head 构成主体，当前只把 FeTS 放入
  短尺度适配分支；
- 当前额外加入 LayerScale、channel embedding、定向跨变量注意力和两层
  时间 Transformer；
- 当前局部 head 读取长尺度上下文；
- 当前最终输出由四专家 MoE 融合，不是原生 FeTS head。

因此论文中应称为“FeTS-inspired/feature-aware PatchTST hybrid”，不能声称
完整复现了 FeTS 主模型。

### 6.3 M2FMoE 与 SSformer 的归因边界

M2FMoE 提供频域 MoE 与动态门控的参考；SSformer 提供稀疏尺度选择、尺度特定
卷积和双向尺度交互的参考。二者共同强化了“不应对所有样本固定使用同一尺度”
的研究动机，但 SSformer 并不是当前 horizon router 的直接来源。当前代码没有
实现：

```text
显式频谱 router
尺度相似度特征
top-k 稀疏路由
双向尺度交互
论文原始专家或损失
```

因此可将其写为思想参考，不能写成 M2FMoE 或 SSformer 的代码复现。

## 7. FeTS-PatchTST 结构演进与归档

### 7.1 各轮结构

| 阶段 | 架构版本 | 核心结构变化 | Git 提交 |
| --- | --- | --- | --- |
| 工程基线 | project PatchTST | 单一 16/8 PatchTST | `817fe4be819267eadb3c2644e20b6268d17719f2` |
| Round 01 | `fets_patchtst_hybrid_v2` | 在单一 PatchTST 路径中接入 AdaFE、DSFFN 和全通道注意力 | `09a222396c1420d608e964bf22a9c62263608b30` |
| Round 02 | `fets_patchtst_multiscale_target_aware_v3` | 完整保留长分支；新增 4/2 局部 FeTS、channel embedding、LayerScale 和功率到非功率特征的定向注意力 | `e952ef4c09a9d2fd48986cc49c5063a124e3de97` |
| Round 03 | `fets_patchtst_multiscale_context_scaled_v4` | 长尺度上下文注入局部 head；显式可学习 horizon scale 与 L2 限制局部修正 | `75b82d45f58c6a798c82661bd0b93e36bf118662` |
| 当前 v5-A/B | `fets_patchtst_horizon_regime_moe_v5ab` | 新增 8/4 中尺度专家、persistence 专家和逐样本/逐 horizon softmax router | `7d5a94a` |

### 7.2 各轮测试均值

| 阶段 | 平均 RMSE | 平均 NRMSE | 结果位置 |
| --- | ---: | ---: | --- |
| Round 01 | 16.581046 | 0.123932 | `archive/round_01_original_fets_patchtst_hybrid_v2_20260705` |
| Round 02 | 17.209558 | 0.121722 | `archive/round_02_fets_patchtst_multiscale_target_aware_v3_20260706` |
| Round 03 | 15.171212 | 0.119122 | `archive/round_03_fets_patchtst_multiscale_context_scaled_v4_20260706` |
| 当前 v5-A/B | 14.172754 | 0.116478 | `archive/round_04_fets_patchtst_horizon_regime_moe_v5ab_20260710` |

四轮归档都包含模型、最佳权重、预处理 artifact、训练历史、测试预测、逐
horizon 指标、汇总指标、可视化、`ARCHIVE_INFO.md` 和 `SHA256SUMS`。当前
v5-A/B 已建立 Round 04 只读快照；活动目录仍作为后续实验的输出位置。

### 7.3 早期“场站间零和”证据

五站 NRMSE 按 `5880 / 5895 / 5971 / 5975 / 6015` 排列：

| 阶段 | 5880 | 5895 | 5971 | 5975 | 6015 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Round 01 | 0.100702 | 0.142854 | 0.137588 | 0.140747 | 0.097770 |
| Round 02 | 0.097995 | 0.132944 | 0.136474 | 0.137233 | 0.103962 |
| Round 03 | 0.099539 | 0.134013 | 0.137198 | 0.138163 | 0.086697 |
| 当前 v5-A/B | 0.099167 | 0.135245 | 0.133535 | 0.135516 | 0.078926 |

Round 02 改善前四站却使 `...6015` 明显退化；Round 03 大幅改善 `...6015`，
但部分较小场站回退。这是引入 horizon-regime 多专家路由的重要动机。

### 7.4 R3 的显式 scale 在当前版本中的状态

Round 03 使用 `HorizonScaledResidualAdd` 显式学习逐 horizon 修正 scale，并给
scale 和修正头加 L2。当前 v5-A/B 的计算图**不再实例化该层**。

当前约束修正幅度的机制为：

- 中、短 residual 输出层零初始化；
- residual/context kernel 使用 `1e-4` L2；
- router 使用 softmax 非负凸权重；
- 初始 router 强烈偏向 long expert。

因此不能把“显式可学习 horizon scale”写成当前 v5-A/B 的有效组成部分。

## 8. 为什么过去常出现两个场站改善、其余场站退化

固定地“看近”或“看远”很难在五个场站全面最优，主要原因不是简单的 patch
数值选错，而是场站条件不同：

1. 不同场站的功率持续性、爬坡频率、零功率比例、容量和气象可预测性不同。
2. 长 patch 平滑局部变化，对慢趋势有利；短 patch 对爬坡敏感，却更容易把
   噪声当成转折。
3. 固定的 `baseline + correction` 把同一种尺度偏置施加给所有样本和 horizon，
   会把一个场站的优势变成另一个场站的过修正。
4. 模型不使用未来 NWP，未来风速突变、停机或限电若在历史中没有先兆，本质上
   不可由当前输入完全辨识。
5. Huber `val_loss` 与最终 NRMSE 相关但不等价；五个场站分别训练时，同一组
   超参数也会落到不同的偏差—方差折中点。
6. 直接 16 步预测容易学习训练分布中的平均未来轨迹：真实功率骤降时仍可能
   给出上升趋势，长零功率段也可能输出一串小非零值。

当前 v5-A/B 的结构回答是：不再固定选单一尺度，而由样本状态和 horizon 决定
long、mid、short、persistence 的比例。现有结果说明该方向比固定尺度修正更有
潜力，但不能理解为结构已经数学保证所有新数据均不退化。

## 9. 当前测试结果

以下数据来自 2026-07-08 的：

```text
./wind_results/wind_dl_all_models_test_metrics_summary.csv
./wind_results/fets_patchtst/testdata_predict_output/
  fets_patchtst_test_metrics_summary.csv
```

### 9.1 五站未加权平均

| 模型 | MAE | RMSE | MAPE | sMAPE | R² | NMAE | NRMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HR-MoE FeTS-PatchTST | **8.437259** | **14.172754** | 52.945953 | **33.583380** | **0.852924** | **0.077475** | **0.116478** |
| 当前 PatchTST 基线 | 9.416268 | 14.791676 | **46.600551** | 36.118461 | 0.840172 | 0.081937 | 0.120938 |
| CNN-LSTM | 12.838240 | 17.207404 | 81.407360 | 36.868961 | 0.838360 | 0.089868 | 0.124576 |
| CNN-ResNet-GRU | 12.844546 | 17.387990 | 77.460367 | 38.030917 | 0.832072 | 0.094305 | 0.127413 |

### 9.2 各场站 NRMSE

| 场站 | HR-MoE FeTS-PatchTST | PatchTST | 相对下降 |
| --- | ---: | ---: | ---: |
| `...5880` | **0.099167** | 0.100844 | 1.663% |
| `...5895` | **0.135245** | 0.138272 | 2.189% |
| `...5971` | **0.133535** | 0.137625 | 2.972% |
| `...5975` | **0.135516** | 0.145299 | 6.733% |
| `...6015` | **0.078926** | 0.082650 | 4.506% |

在当前九模型统一汇总中，FeTS-PatchTST 的五站 NRMSE 均为第一；这也是当前
最强的完整模型性能证据。

### 9.3 非最优项：MAPE

| 场站 | FeTS-PatchTST MAPE | 当前最优 MAPE | 最优模型 |
| --- | ---: | ---: | --- |
| `...5880` | 68.222913 | **51.760137** | PatchTST |
| `...5895` | 53.052549 | **52.412420** | PatchTST |
| `...5971` | **43.693861** | **43.693861** | FeTS-PatchTST |
| `...5975` | 71.447906 | **58.434181** | PatchTST |
| `...6015` | 28.312536 | **23.939130** | PatchTST |

预测代码只在 `|y_true| > 1e-6` 的点计算 MAPE。接近零但非零的功率仍可能造成
很大的相对误差，因此 MAPE 对当前风电零/低功率场景十分敏感，也不能完整评价
被 mask 掉的严格零功率点。论文中不应以 NRMSE 全胜掩盖 MAPE 劣势。

### 9.4 Router 诊断

当前测试集四专家平均权重为：

| 场站 | Long | Mid | Short | Persistence | 归一化熵 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `...5880` | 0.000534 | 0.001842 | 0.041526 | **0.956098** | 0.115128 |
| `...5895` | 0.002127 | 0.028898 | 0.091046 | **0.877928** | 0.279341 |
| `...5971` | 0.001069 | 0.000364 | 0.113670 | **0.884897** | 0.242292 |
| `...5975` | 0.057457 | 0.058876 | 0.172677 | **0.710990** | 0.483125 |
| `...6015` | 0.000602 | 0.033367 | 0.000232 | **0.965799** | 0.094521 |

这说明模型在多数测试样本上选择了强持续性先验，`...5975` 的专家分配最分散。
权重高不等于 persistence 单独完成了全部提升，因为其他专家和 router 参与联合
训练；但它明确提示了专家塌缩/捷径学习风险，也是下一轮消融的重点。

## 10. 单窗口方向错误和零功率非零预测的结构解释

在不引入未来 NWP 的条件下，单窗口出现“实际下降、预测上升”或“实际为零、
预测为非零序列”并不一定是预处理错误，常见结构原因是：

- 未来风况、停机和限电事件未被输入观测到，历史窗口存在一对多未来；
- 直接多步 head 倾向输出条件均值，不保证 16 步单调性或事件一致性；
- 长 patch 和 Huber 回归会平滑尖锐转折；
- 输出反标准化后只裁剪到 `[0, capacity]`，没有“长零状态必须为零”的硬规则；
- persistence 只有在最后历史功率已接近零且 router 选择它时才会产生长零轨迹；
- 定向交叉注意力读取的仍是历史非功率特征，不能代替未来天气预报。

如果论文强调零功率工况，应将可学习的运行状态/零功率 regime 建模作为独立
结构问题评价，而不能只用测试标签设计硬阈值后处理并称为模型贡献。

## 11. 已实现、已移除和未实现的方案

### 11.1 当前计算图已实现

- 完整长尺度 PatchTST 安全分支；
- 8/4 中尺度 PatchTST 残差专家；
- 4/2 局部 FeTS 专家；
- channel identity embedding；
- AdaFE/DSFFN LayerScale residual；
- 功率 Query / 非功率特征 Key-Value 定向交叉注意力；
- 长尺度上下文融合到局部 head；
- 零初始化中、短尺度残差头及 kernel L2；
- persistence expert；
- 样本与 horizon 条件化的稠密 softmax router；
- 逐 horizon router 权重与熵诊断。

### 11.2 历史实现过、当前前向图已移除

- 第一轮 `FeTSFeatureBlock` 的直接残差连接；
- 第一轮 `PatchCrossChannelAttention` 全通道自注意力；
- Round 03 `HorizonScaledResidualAdd` 显式可学习 horizon scale。

这些类仍保留在源文件中主要用于旧 `.keras` 模型反序列化兼容，不能当作当前
v5-A/B 结构组成。

当前未使用的兼容/遗留类包括：

```text
FeTSFeatureBlock
FeTSChannelPatchTranspose
PatchCrossChannelAttention
SelectChannel
HorizonScaledResidualAdd
```

### 11.3 讨论过但当前尚未实现

- 独立的原始功率自注意力旁路；
- feature-group embedding；
- 显式频谱 router 和尺度相似度特征；
- top-k 稀疏路由与专家负载均衡；
- 长短尺度双向交互；
- 简化 AdaFE 或用更轻的门控替代；
- 专门的零功率/停机 regime expert；
- 未来 NWP 分支。

当前源码也明确不包含 RevIN、自蒸馏、外部 teacher、k-fold、多 seed 集成等
历史 tuned 路线。它们不应与当前结构贡献混写。

## 12. 现有证据能证明什么、不能证明什么

### 12.1 已有证据支持

- 完整 v5-A/B 组合相较当前 PatchTST 及七个其它基线，在五站 NRMSE 上均
  取得更低值。
- 完整组合在 MAE、RMSE、sMAPE、R²、NMAE、NRMSE 上具有较强跨场站稳定性。
- 从 R1 到 v5-A/B，平均 NRMSE 总体下降，多尺度和动态路由方向具有经验潜力。
- 五个独立场站模型的平均 router 权重都已明显偏离初始化，且权重分布不同；
  这证明各模型没有始终停留在初始融合比例，但不能证明一个共享模型具有场站
  自适应能力，也不能单凭权重证明工况路由具有因果有效性。

### 12.2 现有证据尚不支持

- 不能证明 AdaFE、DSFFN、channel embedding、定向交叉注意力、mid expert、
  persistence 或 router 中任一模块单独造成了全部提升。
- 不能仅凭潜在 mask 宣称模型完成了原始气象变量层面的可解释特征选择。
- 不能把当前 router 称为 M2FMoE/SSformer 的完整复现。
- 不能声称所有指标全胜，因为 MAPE 在四个场站不占优。
- 不能声称模型已在严格最终盲测集上得到确认。
- 不能从五站当前结果推出结构在任何新场站上数学保证不退化。

## 13. 下一阶段最关键的结构消融

为了支撑一区 SCI 论文的主要创新点，建议至少建立以下直接父子对照：

1. `Long only`：独立原生 PatchTST。
2. `Long + Short FeTS`：无 mid、无 persistence、无 router。
3. `Long + Mid + Short` 固定平均：检验多尺度表示本身。
4. 完整多尺度但移除 persistence：判断当前高 persistence 权重是否为捷径。
5. 固定融合与 horizon-regime router：隔离动态路由贡献。
6. 移除 LayerScale：检验稳定残差注入的作用。
7. 功率—非功率特征定向注意力替换为全通道自注意力或不做跨通道注意力。
8. 移除 channel embedding。
9. 移除长尺度上下文到局部 head 的连接。
10. AdaFE-only、DSFFN-only 和二者组合。

消融应同时报告五站逐场站 NRMSE、逐 horizon 指标、专家权重/熵和零/低功率
分层结果。当前测试集已用于结构反馈，正式论文还需要锁定新的最终测试区间，
或采用至少三个 rolling-origin 时间窗口；这属于研究有效性要求，而不是因为
训练代码读取了测试标签。

## 14. 统一预测与指标定义

`wind_dl_model_predict.py` 当前统一支持：

```text
patchtst
fets_patchtst
bilstm
cnn_lstm
cnn_resnet_gru
wavenet
transformer
informer
autoformer
```

输出包括：

- 所有窗口、所有 horizon 的预测长表；
- 每场站总体指标；
- horizon 1～16，即 15～240 分钟的逐步指标；
- 单个完整 4 小时窗口对比；
- 重叠窗口融合后的完整测试时间轴；
- FeTS 专属 router 权重和路由熵；
- 九模型统一汇总。

总体指标将所有窗口和 16 个 horizon 展平，并只在 `y_true`、`y_pred` 均为
有限值的样本对上计算：

```text
MAE / MSE / RMSE / MAPE / sMAPE / R²
capacity-normalized MAE
capacity-normalized RMSE（NRMSE）
```

完整时间轴对同一目标时刻的重叠预测使用：

\[
w(h)=0.5^{(h-1)/4},
\]

即越近端的预测权重越高，半衰期为 4 步（1 小时）。

FeTS 预测时会校验 artifact 的 `architecture_version` 与当前训练代码一致，并
检查四专家权重逐 horizon 和为 1，避免把旧结构权重误加载为当前模型。

## 15. 工程文件与结果目录

### 15.1 代码和文档

```text
/mnt/d/Python/myprojects/digitalchina2026/
├─ wind_dl_model_train.py
│    原生 PatchTST、公共预处理、标准化和滑窗
├─ wind_dl_other_models_train.py
│    七个其它深度学习基线
├─ wind_FeTS_PatchTST_train.py
│    当前 FeTS-PatchTST 结构与训练入口
├─ wind_dl_model_predict.py
│    九模型统一预测、指标、图表和 router 诊断
└─ docs/
     ├─ WIND_FORECASTING_FETS_PROJECT_CONTEXT.md
     │    FeTS 开始前的历史背景
     └─ WIND_FETS_PATCHTST_MODEL_DEVELOPMENT_CONTEXT.md
          本文：当前结构和实验状态
```

### 15.2 当前 FeTS 活动结果

```text
./wind_results/fets_patchtst/
├─ models/                 五场站完整 .keras 模型
├─ weights/                五场站最佳 .weights.h5
├─ preprocess/             scaler、列信息、架构参数等 artifact
├─ history/                每场站训练历史 CSV 和曲线
├─ tensorboard/            训练日志
├─ tails/                  每场站训练尾部历史上下文
├─ testdata_predict_output/
│  ├─ fets_patchtst_test_metrics_summary.csv
│  ├─ fets_patchtst_test_metrics_by_horizon_all.csv
│  ├─ predictions/
│  ├─ single_window_comparisons/
│  ├─ weighted_curves/
│  ├─ router_diagnostics/
│  └─ figures/
└─ archive/                已冻结的四轮结果
```

五个 `preprocess/*.pkl` artifact 均记录当前架构版本
`fets_patchtst_horizon_regime_moe_v5ab`，这是判断活动模型结构的关键依据。

活动根目录的 `fets_patchtst_training_metrics.csv` 是完整保留的 Round 03/v4
旧训练汇总，不代表当前 v5-A/B。目前没有单一、完整的 v5-A/B 五场站训练汇总
CSV；当前结构以五个 artifact 和模型文件为准，当前测试性能以最新测试汇总
CSV 为准。

### 15.3 历史 FeTS 归档

```text
./wind_results/fets_patchtst/archive/
├─ round_01_original_fets_patchtst_hybrid_v2_20260705/
├─ round_02_fets_patchtst_multiscale_target_aware_v3_20260706/
├─ round_03_fets_patchtst_multiscale_context_scaled_v4_20260706/
└─ round_04_fets_patchtst_horizon_regime_moe_v5ab_20260710/
```

Round 04 只收录五个最终 artifact 对应的 TensorBoard 日志，并根据这些 artifact
重建五场站统一训练指标，排除了旧轮次和被替代的训练记录。归档是结构谱系
证据，不应被后续活动目录同名结果覆盖或与其它轮次模型混用。

### 15.4 PatchTST 与其它基线

```text
./wind_results/patchtst/
./wind_results/bilstm/
./wind_results/cnn_lstm/
./wind_results/cnn_resnet_gru/
./wind_results/wavenet/
./wind_results/transformer/
./wind_results/informer/
./wind_results/autoformer/
```

当前全模型汇总表为：

```text
./wind_results/wind_dl_all_models_test_metrics_summary.csv
./wind_results/wind_dl_all_models_test_metrics_by_horizon_all.csv
```

需要注意 PatchTST 结果来源：预测代码虽然定义了
`wind_results/patchtst/第2轮训练结果` 兼容路径，但当前 artifact 搜索优先读取
标准活动目录 `wind_results/patchtst/preprocess`。当前全模型 CSV 的
`loaded_model_path` 和 `artifact_path` 也指向标准活动目录。因此比较当前结果时
应以汇总 CSV 记录的实际加载路径为准，不要仅根据“第2轮训练结果”文件夹名
推断来源。

旧背景文档记录过另一历史 PatchTST 快照的平均 NRMSE `0.119733`，当前统一
汇总中的 PatchTST 为 `0.120938`。二者属于不同结果快照，不能混在同一张因果
对比表中。

## 16. 新对话应保持的研究边界

后续分析或改动应默认遵守：

1. 当前事实来源优先级：当前代码与 artifact → 当前测试汇总 → 本文 → 旧背景
   文档和历史归档。
2. 保持 FeTS 与基线的预处理、特征工程、96→16 任务和基础 Huber loss 公平，
   除非研究问题明确要求改变其中一项。
3. 区分“本工程原生 PatchTST 拓扑”和 PatchTST 官方仓库原始实现。
4. 区分 FeTS 核心模块复现与本工程新增的多尺度、定向注意力和 MoE。
5. 不把 router 权重当作因果解释；必须结合移除专家和固定融合消融。
6. 不以平均原始 RMSE 代替逐场站 NRMSE，也不以 NRMSE 全胜掩盖 MAPE 劣势。
7. 不把测试反馈误称为代码级数据泄漏，但也不把当前测试集继续包装为最终盲测。
8. 任何新结构都不能预先承诺在所有未来场站绝对不退化；只能通过多窗口、
   多场站和严格消融提高证据强度。

## 17. 主要论文与开源仓库

### PatchTST

- 论文：*A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*
- arXiv：<https://arxiv.org/abs/2211.14730>
- 官方仓库：<https://github.com/yuqinie98/PatchTST>

### FeTS

- 论文：*FeTS: A Feature-Aware Framework for Time Series Forecasting*
- AAAI 页面：<https://ojs.aaai.org/index.php/AAAI/article/view/39838>
- DOI：<https://doi.org/10.1609/aaai.v40i31.39838>
- 官方仓库：<https://github.com/lllucky111/FeTS>
- 当前代码记录的参考 revision：`d908e434b70f3cf69065004e295db13cdb9790b2`

### 多尺度路由思想参考

- M2FMoE：<https://github.com/Yaohui-Huang/M2FMoE>
- M2FMoE 论文：<https://arxiv.org/abs/2601.08631>
- SSformer：<https://github.com/yingliu-coder/SSformer>
- LayerScale/CaiT：<https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.pdf>

上述 M2FMoE、SSformer 和 LayerScale 只用于说明设计思想来源；当前模型是否
真正受益于对应机制，仍应以本项目直接消融为准。
