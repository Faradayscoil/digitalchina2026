# FeTS-PatchTST 面向 SCI 一区论文的创新路线、实验方案与结果总览

> 更新日期：2026-07-27
> 项目根目录：`/mnt/d/Python/myprojects/digitalchina2026`
> 任务：5 个开发场站 + 14 个补充风场、15 min 分辨率、历史 96 点预测未来 16 点的超短期风电功率预测
> 当前实验种子：`seed=2026`
> 当前正式模型名称：`WindPRISM`，结构映射为 `X0 = D0 = T0 = G0 = F7`
> 原 5 站协议：`legacy_seen_test_selected`，用于机制开发与消融
> 补充 14 站协议：严格按时间 70/15/15 划分、train-only 预处理的外部场站补充验证；结果选择仍按保守口径标记为描述性，而非全球最终盲测
> 背景文档：`docs/WIND_FETS_PATCHTST_MODEL_DEVELOPMENT_CONTEXT.md`

本文将 FeTS-PatchTST 项目迄今的研究讨论、创新充分性判断、实验方案、执行进度、
正式结果、文件路径和论文表述边界整理为一份可持续更新的总览。本文不是逐句
对话转录，而是按“研究问题—实验—证据—结论”的顺序重组全部关键决策。

“FeTS-PatchTST”是项目沿用的历史工程名称。Stage 1 以后经消融定型的当前部署
模型已经不是完整 FeTS 或 PatchTST，而是 **Persistence-centered lightweight
regime-gated forecaster**。后续新对话不得仅根据文档标题误认为最终模型仍含
PatchTST encoder、四专家 MoE 或 FeTS 模块。

文中必须始终区分三类结论：

1. **数值最优**：某个变体的宏平均指标最低；
2. **正式选中**：变体同时通过预声明的总体、逐场站、分工况和安全门槛；
3. **论文可主张**：结果不仅数值较好，而且有直接消融、合理归因和足够严格的
   泛化证据。

当前 `WindPRISM = X0/D0/T0/G0/F7` 已在原 5 站机制开发、同超参数公平重训以及
14 个补充风场的 15 模型统一比较中三次保持正式最优。它可以作为手稿的最终方法，
但证据应分层表述：原 5 站是已见测试集上的开发性证据；14 站采用新的无泄漏
处理与冻结模型比较，显著增强跨数据证据，但由于这些站点曾在另一条已淘汰的
teacher-data 工作流中有历史暴露，结果文件按保守原则没有标记为全球最终盲测。

---

## 1. 当前结论速览

### 1.1 当前最终模型

当前最终保留并正式命名为 **WindPRISM** 的模型为：

~~~text
96×45 历史多变量输入（24 h）
├─ Persistence P：最后历史功率重复到未来16步（4 h）
├─ Corrected C = P + lightweight causal residual
│    ├─ causal Conv1D(32, k=5)
│    ├─ causal Conv1D(32, k=3, dilation=2)
│    ├─ last token + global average + last raw features
│    └─ Dense(64) + Dropout + Dense(16) → residual
└─ 36维显式工况 P+H+D（20+12+4）
     └─ LayerNorm → Dense(24, GELU) → Dropout(0.1) → Dense(24)
          ↓
non-factorized sample × horizon sigmoid gate（hidden=16，horizon embedding=8）
          ↓
ŷ = P + gate × (C - P)
~~~

关键事实：

- 参数量：20,969；
- 原 5 个开发场站的阶段选型结果：Macro NRMSE 0.113760989，Macro NMAE
  0.077608814；
- 原 5 站按强基线超参数从零公平重训后：Macro NRMSE 0.115067120，
  Macro NMAE 0.078799410，仍为 NRMSE 第 1；
- 补充 14 站统一比较：Macro trNRMSE 0.128801222、Macro trNMAE
  0.082437232、Macro R² 0.769093510，15 个模型中综合第 1；
- 补充 14 站 Micro trNRMSE 0.132537213、Micro trNMAE 0.081770119、
  Micro R² 0.834648202，同样为第 1；
- 当前正式名称映射：`F7`（特征结构）= `G0`（Stage 3 参考）= `T0`
  （Stage 4 参考）= `D0`（Stage 4B 参考）= `X0`（Stage 5 参考）；
- 不包含原始四专家模型中的 long、mid、short FeTS 专家；
- 不包含 G1–G4 的因子化校准安全门控；
- 不包含 T1–T3 的时间、频率或时频交互 adapter；
- 不包含 X1-F/M/C/X1 的静态多尺度 adapter，也不包含 X2–X6 token 交互。

### 1.2 各阶段最终选择

| 阶段 | 研究问题 | 数值观察 | 正式结论 | 状态 |
| --- | --- | --- | --- | --- |
| 历史 Round 01–04 | 复杂 FeTS-PatchTST 是否优于旧基线 | Round 04 NRMSE 0.116478，优于旧 PatchTST 0.120938 | 完整组合有效，但无法归因到单模块 | 完成，作为动机 |
| Stage 1，B0–B6 | 最小有效预测主干是什么 | B2 Macro NRMSE 0.115700，优于重训 B6 0.116939，且仅 18,416 参数 | 形式化原规则选 B6；后续研究按 Pareto 选择 B2 主干并保留 B6 参考 | 完成 |
| Stage 2A，R0–R6 | 显式工况编码是否优于静态/隐式门控 | R4 NRMSE 0.113822 最低 | 以 R4 为特征筛选母结构 | 完成 |
| Stage 2B，F0–F8/FP | 哪些显式工况特征有效 | F7=P+H+D NRMSE 0.113761 最低 | 删除 M、C 和辅助任务，保留 F7 | 完成 |
| Stage 3，G0–G4 | 校准、安全损失和 Persistence 保护是否可晋级 | G1 NRMSE 0.113606 数值最低；G2/G3 校准更好 | 全部新模型因 ramp 门槛失败，回退 G0/F7 | 完成 |
| Stage 4，T0/M0/T1–T3 | 最小 residual 与轻量时频增强是否有效 | T3 corrected candidate 最好，但 fused NRMSE 0.114492 | 全部新模型失败宏精度/逐场/ramp 门槛，回退 T0 | 完成 |
| Stage 4B，D0/D0R/D1–D3 | T1 候选收益能否通过重新生成 oracle/Q90 和重训门控转化 | D0R 数值最低 0.113721，仅比 D0 好 0.0349%，但仅 2/5 场站改善 | 全部新变体未过 0.2%/逐场守门，回退 D0 | 完成 |
| Stage 5A，X0/X1-F/M/C/X1 | 轻量 fine/mid/coarse 静态历史表示本身是否改善 corrected candidate | X1-C candidate 数值最低 0.115680；X1-F 位于 0.1% 最优带且更轻 | Stage 5A 形式化选 X1-F；full X1 仅进入后续闭环诊断 | 完成 |
| X1R 门控闭环 | full X1 候选收益能否经同 candidate 的新校准安全门控转化 | X1-fixed-G0 fused 数值为 0.113691（诊断，不可选）；X1R NRMSE 0.114370，但 NMAE/校准更好 | 精度、逐场、dynamic/ramp 与留一守门失败，正式回退 X0 | 完成 |
| Stage 5B，X2–X6 | token 级单向/双向跨尺度交互是否应启动 | X1R 没有把 X1 candidate 收益转成稳健 fused 收益 | `stage5b_x2_x3_unlocked=false`，停止 X2–X6 | 未解锁 |
| Part 3 Round 1，A0–A5 | TimeAlign 风格残差对齐和 regime-QDF 是否能稳定增强 X0 | A1 candidate NRMSE 0.114888，较 A0 改善 1.095%，但只有 3/5 场站 NRMSE 不退化；A2–A5 同样未过守门 | 无新变体通过稳健选型或 Stage B 解锁条件，保留 A0/WindPRISM | 完成，Stage B 不启动 |
| Part 3 Round 2，公平重训 | WindPRISM 的优势是否只来自训练预算或 batch 差异 | 从零、batch=256、lr=5e-4、80 epochs 后 Macro NRMSE 0.115067，优于历史 HR-MoE 0.116478 和 PatchTST 0.120938 | 排除“仅因 warm-start 或 batch=192 获胜”的主要混淆，WindPRISM 保持正式模型 | 完成 |
| Part 3 Round 3，外部 14 站 | 在无泄漏处理和 15 模型强基线矩阵下能否保持综合优势 | WindPRISM Macro trNRMSE 0.128801、trNMAE 0.082437、R² 0.769094，14 站中 6 站 NRMSE 第 1 | Macro/Micro 均综合第 1；较 DLinear/PatchTST/HR-MoE 的 Macro trNRMSE 分别改善 1.54%/2.14%/2.88% | 完成 |

### 1.3 当前最重要的科学结论

1. 项目真正稳定的性能来源不是原始四专家堆叠，而是
   **Persistence 物理先验 + 轻量因果修正 + 显式工况门控**。
2. 显式工况统计比仅 horizon 门控和隐式卷积门控更能形成样本依赖路由，但并非
   特征越多越好；当前 `P+H+D` 优于完整 `P+H+M+D+C`。
3. 校准、安全损失和 abstention 能明显降低 Brier、ECE、后悔与伤害率，却尚未
   同时保持 ramp 和总体 NRMSE，表现为明确的精度—安全 Pareto 权衡。
4. T1/T3 能改善 corrected candidate，但统一新门控没有把候选增益稳定转化为
   五场站 fused 增益；当前瓶颈不是简单“缺少一个时频模块”。
5. Stage 4B 证明“为新 candidate 重建 train-only oracle/Q90 并重训 gate”是
   必需的受控闭环，却不足以保证精度收益转化；T1 + fixed G0 的诊断值甚至优于
   新门控，说明 gate objective 与尾部误差仍错配。
6. Stage 5A 证明静态 fine/mid/coarse 表示可小幅改善 corrected candidate，但
   X1R 又证明更好的校准、安全和平均绝对误差可以伴随 NRMSE、dynamic/ramp
   恶化。不能把“candidate 改善”或“门控校准改善”等同于部署模型晋级。
7. Part 3 Round 1 的否定结果进一步说明：把 TimeAlign 式未来残差对齐或二次
   多步目标直接叠加到现有 candidate，并不能稳定转化为五站 fused 增益，因此
   不再启动 xCPD/variable×patch Stage B。
8. Part 3 Round 2 证明 WindPRISM 在与原生 PatchTST/旧强基线对齐训练预算后仍
   保持 NRMSE 第 1，轻量优势不是由 batch size 或 warm-start 单独造成。
9. Part 3 Round 3 在 14 个补充风场、15 个模型上确认 WindPRISM 的 Macro 和
   Micro 综合最优，并在 H3–H16 的逐 horizon NRMSE 上均为最优；最接近的
   DLinear、PatchTST 和 HR-MoE 差异经 Holm 校正后未达到显著，故论文应写成
   “综合最优且具有明显参数效率”，不能写成“统计显著击败所有基线”。
10. 现有证据已从 2026-07-19 时的“专业型 Q2 更现实”提升到
    **具备冲击专业型 SCI/JCR Q1 的完整方法—消融—外部补充验证链条，但仍属
    Q1 边缘而非高把握录用**。继续堆叠新模型结构的边际价值低于完成统计后处理、
    统一硬件效率报告和严谨手稿表述。

---

## 2. 任务、数据与评价边界

| 项目 | 当前设定 |
| --- | --- |
| 时间分辨率 | 15 min |
| 历史窗口 | 96 点，即过去 24 h |
| 预测窗口 | 16 点，即未来 4 h |
| 目标 | 功率 |
| 场站 | 原 5 个开发场站 + 14 个补充场站，均按场站独立训练 |
| 未来 NWP | 不使用 |
| 主选择指标 | 原 5 站为容量归一化 Macro NRMSE；补充 14 站因无可信装机容量，正式使用 train-q99.9 参考值归一化的 Macro trNRMSE |
| 辅助指标 | NMAE、MAE、RMSE、R²、逐 horizon、逐工况、candidate、门控校准与安全指标 |
| 当前训练随机性 | 各轮固定 `seed=2026`；用户已说明同模型多 seed 结论一致，手稿引用前仍需定位并归档对应 artifact |
| 当前批量 | 主开发实验和外部 14 站为 192；Part 3 Round 2 公平重训为 256；外部 14 站未触发 192→128 OOM 回退 |

五个场站：

~~~text
4081950112845135880
4081950112845135895
4081950112845135971
4081950112845135975
4081950112845136015
~~~

数据位置：

~~~text
wind_split/wind_train_<farm_id>.csv
wind_split/wind_test_<farm_id>.csv
wind_split/supplementary_other_wind_data/JSFD001 ... JSFD014
~~~

外部 14 站从原始 Excel 独立预处理，明确拒绝使用其它分支生成的
`processed_npz`。处理后的 bundle、审计、模型与结果统一保存在：

~~~text
wind_results/part3_new_module_supplement/
└─ 03_external14_leakage_free_strong_baseline_benchmark/
~~~

### 2.1 公平比较口径

各阶段尽量保持以下内容不变：

- 相同的 96→16 任务；
- 相同的场站数据和容量归一化口径；
- 相同的历史气象、派生变量和历史功率输入；
- 相同的时间顺序验证切分；
- 相同的 Huber 主损失框架；
- 相同 seed；
- 只改变实验矩阵中被研究的结构、特征或损失。

### 2.2 两层证据协议与限制

原 5 站结果应被标记为 `legacy_seen` 或 `legacy_seen_test_selected`，原因是：

- 测试标签没有进入训练输入，但研究过程查看测试表现后继续选择了结构；
- F0–F8、G0–G4、T0–T3、D0–D3、X0–X1 与 X1R 均按用户要求使用当前
  测试集做阶段选型；
- 当前测试预处理先在整段文件做双向气象插值，严格在线因果性不足；
- scaler 在完整训练文件上拟合后再切训练/验证窗口，验证段参与归一化统计；
- 当前主归档以 seed=2026 为准。

补充 14 站纠正了原协议的核心数据泄漏风险：

- 只读取原始 Excel，按时间顺序 70%/15%/15% 固定训练、验证和测试段；
- scaler、缺失值重建统计、功率参考值和工况配置均只由训练段确定；
- 以 `available_at <= decision_time` 约束历史功率和测风信息；
- 固定 45 通道 `FEATURE_SCHEMA_V1`，预测目标仍为未来 16 步；
- 无可信装机容量时使用每站训练段功率 q99.9，指标必须称为 trNRMSE/trNMAE；
- 每个模型都在其首次正式测试预测前冻结并恢复最佳验证 checkpoint；但 15 模型
  矩阵是“旧 10 模型结果冻结 + 四个现代基线和 Persistence 一次性追加”的扩展，
  不是 15 模型在任何测试预测前同时冻结。

但补充 14 站也不是严格意义的“从未接触数据集/未见场站零样本迁移”：

- 每站仍是自身前 70% 训练、后 15% 测试的 **within-station chronological
  holdout**，不是在完全未见场站直接推理；
- JSFD001–JSFD014 曾在另一条后来淘汰的 teacher-data 工作流中出现，因此最终
  marker 保守设置为 `test_is_final_blind_evaluation=false`；
- `test_used_for_selection=true` 表示本轮在测试结果上描述性选出最终模型，但
  `test_targets_used_for_training_or_validation_selection=false`，且 WindPRISM
  在增加 iTransformer、TimesNet、TimeMixer、DLinear 和 Persistence 后未改变；
- 时间戳语义由用户确认无误，不过现有自动审计仍保留
  `assumed_interval_start/uncertain_alignment` 标签；手稿应以原始元数据或人工
  核验记录补齐证据，而不能删除该审计边界；
- 14 站主比较只有 seed=2026；既有多 seed 稳定性若用于论文，必须引用确切文件，
  不得把其它已淘汰模型的多 seed 结果替代 WindPRISM 证据。

---

## 3. 2025–2026 年近期论文的创新规律

### 3.1 文献范围说明

这里聚焦单一风电预测任务中的时序、时频、物理约束、工况和空间依赖方法，
不把图像—文本—语音等多模态融合作为本项目目标。期刊 JCR 分区会随年份和
学科类别变化；下表用于总结近期高影响力能源预测论文的创新组织方式，不代表
本文承诺每篇在每一版 JCR 中都固定为一区。

### 3.2 代表性论文及其创新组织

| 年份 | 代表工作 | 面向的真实问题 | 创新点组织方式 | 对本项目的启示 |
| --- | --- | --- | --- | --- |
| 2025 | [Physics-constrained wind power forecasting aligned with probability distributions for noise-resilient deep learning](https://www.sciencedirect.com/science/article/pii/S030626192500025X)，Applied Energy | 风速噪声和预测物理不合理 | 将风功率曲线的概率分布知识通过 KDE 与 JS divergence 嵌入损失，并在 25 台机组和多噪声条件验证 | 物理先验必须进入可检验机制；Persistence 安全锚点需要直接对照和风险指标 |
| 2025 | [Non-stationary GNNCrossformer](https://www.sciencedirect.com/science/article/abs/pii/S0306261924018750)，Applied Energy | 非平稳、多场站时空依赖 | stationarization、de-stationary attention、跨时间/跨变量注意力和 GNN 分别对应具体依赖 | 模块应逐一对应问题；多尺度或跨场站机制需要直接消融 |
| 2025 | [Developing an interpretable wind power forecasting system using a transformer network and transfer learning](https://www.sciencedirect.com/science/article/abs/pii/S0196890424010963)，Energy Conversion and Management | 新场站小样本与黑箱性 | 特征筛选、注意力解释和参数共享迁移学习构成方法与证据闭环 | F0–F8 特征组消融比仅展示注意力权重更可信；跨场站泛化仍需补证 |
| 2025 | [A novel frequency sparse downsampling interaction transformer for wind power forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0360544225018419)，Energy | 风电高频波动、趋势与计算复杂度 | 周期/趋势分解、频域稀疏注意、下采样交互，并报告三风场和多预测尺度 | 时频增强必须证明对波动/ramp有效，并同时给出复杂度收益 |
| 2025 | [Fine-grained ultra-short-term wind power forecasting based on TFT integrated with turbine power time-series clustering](https://www.sciencedirect.com/science/article/abs/pii/S0360544225036370)，Energy | 机组异质性与细粒度预测 | 功率序列聚类、并行 TFT、确定性/概率预测与扩展性 | 只看宏平均不足；需要分场站、分工况、稳定性和可扩展性证据 |
| 2025 | [Ultra-short-term wind power forecasting based on optimized decomposition and deep learning](https://www.sciencedirect.com/science/article/pii/S2590174525004477)，Energy Conversion and Management: X | 非平稳、模态混叠和分布偏移 | 特征筛选、优化分解、多尺度图/稀疏注意组合 | 分解或多尺度不是目的，必须隔离各部分增益并控制候选漂移 |
| 2026 | [Network integrating multiscale analysis and nonlinear representation for short-term wind power forecasting](https://www.sciencedirect.com/science/article/pii/S0960148126006750)，Renewable Energy | 多尺度时频利用不足、特征提取与非线性建模脱节 | Wavelet-Frequency-Time Transformer、频带选择和注意力解释 | 若继续跨尺度，应实现真正的尺度交互和频带选择，而非只拼接三个分支 |
| 2026 | [A time-frequency adaptive transformer for long-term wind power forecasting under complex meteorological fluctuations](https://www.sciencedirect.com/science/article/pii/S0957417426006536)，Expert Systems with Applications | 复杂气象下的时频依赖和频谱信息损失 | 两阶段时频建模、复谱机制、按任务自适应分配注意头，并在三数据集验证 | T0–T3 只是最小探针；只有跨场站稳定提升后才值得扩大时频结构 |
| 2026 | [STWFormer: Interpretable wavelet-based spatio-temporal transformer](https://www.sciencedirect.com/science/article/pii/S0378779626003548)，Electric Power Systems Research | 时间—频率演化和多变量空间关系 | 小波多尺度网络与时空解耦注意并行，强调解释与真实数据验证 | token/尺度交互必须有清晰归因，不应以复杂结构名称代替证据 |
| 2026 | [Enhancing short-term wind power forecasting through virtual prediction and wavelet packet transform](https://www.sciencedirect.com/science/article/pii/S0378779625012271)，Electric Power Systems Research | 波动信号频率子带的可预测性差异 | 小波包分解、分频预测、组合权重 | “多候选+融合”应验证每个候选质量和融合是否真正超过最佳候选 |
| 2026 | [A physics-aware dynamic graph and mixture-of-experts framework for wind power forecasting](https://www.sciencedirect.com/science/article/pii/S0142061526002115)，International Journal of Electrical Power & Energy Systems | 物理机理、空间依赖与工况异质性 | 动态图与 physics/data experts 的条件融合 | “物理候选+数据候选+动态门控”本身已不是充分新颖点，必须有更专门的失效问题与直接证据 |
| 2026 | [A novel hybrid short-term and ultra-short-term wind power forecasting method based on Weather Research and Forecasting: WRF-iTransformer-PSO](https://www.sciencedirect.com/science/article/pii/S0360544226010601)，Energy | 数值天气预报与数据模型协同 | WRF 预报、iTransformer 与超参数优化形成气象—预测链 | 不使用未来 NWP 的本文必须明确任务边界，并以轻量、可靠和历史信息利用形成差异化 |
| 2026 | [Ultra-short-term wind power prediction for enhanced reliability considering error distribution characteristics and guided correction](https://www.sciencedirect.com/science/article/pii/S0360544226015033)，Energy | 平均误差之外的尾部风险和可靠校正 | 显式建模误差分布并按风险引导修正 | dynamic/ramp、过估风险、regret/harm 与校准可以成为核心问题，但必须与最终精度闭环 |
| 2025 | [A cross-dataset benchmark for neural network-based wind power forecasting](https://www.sciencedirect.com/science/article/pii/S0960148125011255)，Renewable Energy | 不同数据集、尺度和初始化下结论不可比 | 八个全球数据集上的统一神经网络基准 | 跨数据集、随机种子和统一协议已接近方法论文的必要证据，而非附属实验 |

### 3.3 归纳出的创新规律

近期高质量风电预测论文的创新点通常具有以下共同规律：

1. **从问题出发，而不是从模块清单出发。**
   结构要明确对应非平稳、ramp、高频噪声、场站异质性、物理边界或分布漂移。

2. **多尺度与时频仍是高频方向，但强调选择性和交互。**
   常见做法是趋势/周期分解、wavelet/Fourier、稀疏频率选择、局部—全局交互；
   简单并联、池化后拼接越来越难构成充分创新。Stage 5A 的 X1 属于后者，
   因此即使 candidate 小幅改善，也不能直接包装成一区级跨尺度创新。

3. **动态融合从“能变化”转向“可校准、可解释、可保护”。**
   仅展示平均门控权重不够，需要 oracle、Brier/ECE、regret、harm、饱和率和
   分工况收益，且候选改变后必须重新定义 oracle。

4. **物理或工程先验成为重要差异化来源。**
   风功率曲线、容量边界、Persistence、ramp 和低功率约束比任意堆叠注意力
   更容易形成可信的领域贡献。

5. **解释性要通过特征筛选和直接消融支撑。**
   不能只把 attention heatmap 当因果解释；应设计加入/删除特征组和冻结候选
   控制，防止 candidate drift 混淆归因。

6. **跨场站泛化、稳健性和复杂度是方法的一部分。**
   近期论文通常报告多个数据集/场站、噪声或工况稳健性、参数量或效率；单一
   平均指标很难支撑一区方法贡献。

7. **概率预测与不确定性正在增强，但不是所有论文的必选项。**
   若不做概率预测，至少要通过校准、风险和安全回退说明确定性模型何时可信。

8. **证据标准高于结构新颖度。**
   直接父子消融、逐场站/逐 horizon/逐工况结果、复杂度、稳定性和失败分析，
   往往比再增加一个命名复杂的模块更重要。

9. **候选质量、门控质量和最终融合质量必须三层分开。**
   Stage 4B 与 X1R 的结果表明，candidate 变好、Brier/ECE 变好或 regret/harm
   下降都不自动推出 fused NRMSE 变好；每个 candidate 改变后都必须重新生成
   train-only oracle/Q90，并以 candidate→fused 收益转化率验收。

---

## 4. 最初结构的创新充分性判断与路线调整

### 4.1 原 HR-MoE FeTS-PatchTST 的优点

原 `fets_patchtst_horizon_regime_moe_v5ab` 包括：

- long PatchTST；
- mid PatchTST residual；
- short FeTS residual，包括 AdaFE、DSFFN、LayerScale 和功率到非功率特征
  的定向注意；
- Persistence expert；
- 逐样本、逐 horizon 的四专家 softmax router。

历史 Round 04 快照取得 Macro NRMSE 0.116478，优于同一历史汇总中的本工程
PatchTST 0.120938；五场站 NRMSE 均改善。这个结果说明**完整组合具有经验
潜力**。

### 4.2 为什么当时仍不足以支撑一区主创新

原结构存在四个关键证据缺口：

1. 没有直接证明 long、mid、short FeTS、Persistence 和动态 router 各自有效；
2. 多数场站 router 高度偏向 Persistence，存在专家塌缩/捷径风险；
3. 模型复杂度较高，后续公平重训 B6 达 885,395 参数；
4. 测试集已参与结构反馈，完整组合的提升不能替代直接消融和严格泛化。

因此研究路线没有立即继续叠加“防塌缩稀疏路由、因果时频、跨尺度交互”，而是
先做最小有效结构搜索。该顺序符合近期论文强调的“问题—机制—证据”规律。

### 4.3 路线重构

~~~text
复杂四专家 HR-MoE
    ↓ Stage 1：先找最小有效预测主干
Persistence + lightweight causal residual（B2）
    ↓ Stage 2：显式工况编码与特征归因
P+H+D 两候选动态融合（F7）
    ↓ Stage 3：校准、因子化与 Persistence 安全保护
新门控未过 ramp 门槛 → 回退 G0
    ↓ Stage 4：只在满足5.1条件后做最小时频探针
candidate略有改善但 fused 未改善 → 回退 T0
    ↓ Stage 4B：T1/F7 candidate × direct/calibrated/factorized gate 闭环
新门控未稳定转化候选收益 → 回退 D0
    ↓ Stage 5A：fine/mid/coarse 独立静态表示
candidate 小幅改善，形式化选 X1-F；full X1 进入闭环诊断
    ↓ X1R：冻结 full X1，重建 oracle/Q90 并重训非因子化校准安全 gate
校准/安全改善但 NRMSE、dynamic/ramp 失败 → 回退 X0
    ↓ Part 3 Round 1：residual alignment + regime-QDF
A1 宏平均 candidate 改善但跨站守门失败 → A0，Stage B 不解锁
    ↓ Part 3 Round 2：五站从零公平训练
WindPRISM 仍为 NRMSE 第1，排除主要训练预算混淆
    ↓ Part 3 Round 3：外部14站无泄漏15模型基准
WindPRISM Macro/Micro 综合第1 → 结构定型并停止新增模型
~~~

---

## 5. 面向一区论文的主要创新点及证据状态

| 拟写创新点 | 当前可用证据 | 证据状态 | 论文表述边界 |
| --- | --- | --- | --- |
| Persistence 中心的最小因果修正预测器 | B2 比 B6 Macro NRMSE 低 1.06%，参数少 97.92%，训练时间少约 97.78% | **已支持** | 可称轻量、物理先验中心的 corrected forecasting；不能称所有复杂专家均必要 |
| 显式风电工况驱动的逐样本、逐 horizon 两候选融合 | R4 优于静态/隐式门控；F7 最终为 20,969 参数、NRMSE 0.113761 | **已支持，但效应需谨慎** | 可称显式工况编码有效；F7 对 F4 的宏优势很小且仅 1/5 场严格更优 |
| P/H/D 特征组的可解释筛选与冻结候选归因 | F0–F8 端到端消融、FP0/FP4 Frozen-Pair、candidate drift 报告 | **已支持** | P、H、D 可保留；M、C 和辅助任务不能写成有效贡献 |
| 校准、regret 与 Persistence 安全保护框架 | G2/G3 显著改善 Brier/ECE、regret/harm；G4 展示安全—精度 Pareto | **作为分析框架支持** | 不能声称最终模型已采用校准安全门控，因为 G1–G4 均未晋级 |
| 因果时频增强 residual | T1/T3 改善 corrected candidate，但最终 fused NRMSE 和 ramp 退化 | **未支持为最终创新** | 只能作为否定消融和瓶颈诊断，不能写成最终模型贡献 |
| 轻量 fine/mid/coarse 历史表示 | X1-F/M/C/X1 均已训练；candidate 改善 0.280%–0.413%，但 frozen-G0/X1R 未形成合格部署增益 | **候选层面支持，最终结构未采用** | 可写静态多尺度负/弱阳性消融；不能称最终模型含多尺度模块 |
| candidate-specific oracle/Q90 与门控闭环 | Stage 4B、X1R 均严格冻结 candidate、重建 train-only oracle/Q90 并做身份审计 | **实验方法支持，精度闭环失败** | 可作为严谨归因和可靠性评价贡献；不能声称新 gate 提升最终 NRMSE |
| X1R 非因子化校准安全门控 | 相同 X1 candidate 下 Brier/ECE、regret/harm 和饱和率显著改善，但 NRMSE 比 X0 恶化 0.5353% | **安全/校准支持，部署不晋级** | 必须同时报告 NMAE 改善与 NRMSE/dynamic/ramp 失败，不得选择性汇报 |
| 真正的 fine/mid/coarse token 级双向跨尺度交互 | X2–X6 未训练且未解锁 | **未完成** | 不得出现在当前模型结构或已完成贡献列表中 |
| TimeAlign 风格残差对齐与 regime-QDF | Part 3 A1–A5；A1 candidate 宏平均改善但仅 3/5 场站不退化，全部未过 Stage B 守门 | **未支持为最终创新** | 可作为高成功先验的否定消融；不得写入 WindPRISM 推理图 |
| 从零公平训练下的结构有效性 | Part 3 Round 2：WindPRISM NRMSE 0.115067，第 1；参数 20,969 | **已支持** | 可排除主要 warm-start/batch 混淆；NMAE 仍由 HR-MoE 略优 |
| 外部 14 站强基线与中长 lead 泛化 | 15 模型 Macro/Micro 综合第 1，H3–H16 NRMSE 第 1，显著优于 11/14 基线 | **已支持，协议边界明确** | 必须称 trNRMSE/trNMAE；不是 globally blind，且最近三基线差异不显著 |
| 原四专家 FeTS-PatchTST/防塌缩稀疏路由 | Stage 1 未证明复杂专家不可删除；两候选结构不再需要四专家负载均衡 | **不再作为当前主线** | 只能作为历史动机或对照，不能作为最终模型创新 |

当前最稳妥的论文主线是：

> 以 Persistence 为低方差物理锚点，用轻量因果 residual 形成可修正候选，再用
> 经直接特征组消融筛选出的显式风电工况编码驱动逐样本、逐 horizon 融合，并
> 以 candidate drift、oracle 校准、regret/harm 和 Persistence 保护诊断融合
> 可靠性。

其中前半句是最终模型结构贡献，后半句主要是系统实验与安全评价贡献。不能把
未晋级的 G2/G3/G4、T1–T3、X1/X1R 写入最终部署结构。

---

## 6. 历史阶段：Round 01–04 复杂模型演进

### 6.1 实验比较

| 历史阶段 | 核心变化 | 平均 RMSE | 平均 NRMSE | 结果位置 |
| --- | --- | ---: | ---: | --- |
| Round 01 | 单 PatchTST 路径接入 AdaFE、DSFFN 和全通道注意 | 16.581046 | 0.123932 | `wind_results/fets_patchtst/archive/round_01_original_fets_patchtst_hybrid_v2_20260705/` |
| Round 02 | long 主干 + short FeTS + channel embedding + 定向注意 | 17.209558 | 0.121722 | `wind_results/fets_patchtst/archive/round_02_fets_patchtst_multiscale_target_aware_v3_20260706/` |
| Round 03 | long context 注入局部 head + horizon scale | 15.171212 | 0.119122 | `wind_results/fets_patchtst/archive/round_03_fets_patchtst_multiscale_context_scaled_v4_20260706/` |
| Round 04 | long/mid/short/Persistence + horizon-regime router | 14.172754 | **0.116478** | `wind_results/fets_patchtst/archive/round_04_fets_patchtst_horizon_regime_moe_v5ab_20260710/` |

### 6.2 Round 04 与历史 PatchTST

| 场站尾号 | Round 04 HR-MoE | 历史 PatchTST | 相对下降 |
| --- | ---: | ---: | ---: |
| 5880 | 0.099167 | 0.100844 | 1.663% |
| 5895 | 0.135245 | 0.138272 | 2.189% |
| 5971 | 0.133535 | 0.137625 | 2.972% |
| 5975 | 0.135516 | 0.145299 | 6.733% |
| 6015 | 0.078926 | 0.082650 | 4.506% |
| Macro | **0.116478** | 0.120938 | 3.69% |

这些数值属于历史快照。后续 Stage 1 的 B1/B6 是 seed=2026 的公平重训，不得
把历史 0.116478/0.120938 与重训 0.116939/0.121784 混成同一直接消融。

### 6.3 历史阶段结论

- 完整 HR-MoE 在当时九模型汇总中具有较强经验性能；
- Round 02–03 曾出现某些场站改善、另一些场站退化的“零和调优”；
- router 多数情况下高度依赖 Persistence；
- 不能由完整组合结果证明 FeTS、mid、short 或动态 router 的独立贡献；
- 因此该阶段提供的是 Stage 1 的研究动机，而不是当前论文最终结构。

---

## 7. 第一阶段：B0–B6 最小有效结构搜索

### 7.1 实验目标和矩阵

本阶段先回答“复杂度应先缩小还是先增加创新模块”。决定是先缩小复杂度，原因是
如果无法确认哪些模块有效，继续加路由或跨尺度交互会进一步破坏可归因性。

| 变体 | 结构 | 研究问题 |
| --- | --- | --- |
| B0 | Persistence | 解析式物理基线 |
| B1 | long PatchTST | 单一长尺度基线 |
| B2 | Persistence + lightweight causal residual | 最小可学习修正是否足够 |
| B3 | long + Persistence，静态 horizon softmax | Persistence 与 long 的固定分步融合 |
| B4M | long + mid + Persistence，静态融合 | mid 的独立增量 |
| B4S | long + short FeTS + Persistence，静态融合 | short FeTS 的独立增量 |
| B5 | long + mid + short + Persistence，静态融合 | 多尺度候选本身的增量 |
| B6 | 全部四候选，样本–horizon 动态 softmax | 完整 v5 拓扑公平重训 |

B1 的 long 分支复用本工程 `wind_dl_model_train.py` 中
`build_patchtst_model()` 的长尺度拓扑，但它是在 Stage 1 框架下以 seed=2026
重新训练的结果，不是旧 PatchTST artifact，也不是官方 PatchTST 仓库逐行复刻。

### 7.2 测试集总体结果

| 变体 | 参数量 | Macro NMAE | Macro NRMSE | Macro R² |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0 | **0.077918** | 0.122664 | 0.833248 |
| B1 | 210,960 | 0.082630 | 0.121784 | 0.837526 |
| **B2** | **18,416** | 0.080718 | **0.115700** | **0.856006** |
| B3 | 210,992 | 0.082396 | 0.121491 | 0.838045 |
| B4M | 487,056 | 0.080266 | 0.120795 | 0.840195 |
| B4S | 477,269 | 0.081268 | 0.120295 | 0.845966 |
| B5 | 753,333 | 0.081736 | 0.120324 | 0.844085 |
| B6 | 885,395 | 0.078293 | 0.116939 | 0.850746 |

### 7.3 各场站 NRMSE

| 变体 | 5880 | 5895 | 5971 | 5975 | 6015 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.102828 | 0.140817 | 0.140418 | 0.150372 | 0.078885 |
| B1 | 0.102323 | 0.139783 | 0.138568 | 0.146476 | 0.081773 |
| B2 | **0.094176** | 0.136396 | **0.133712** | **0.133599** | 0.080615 |
| B3 | 0.102603 | 0.138233 | 0.137628 | 0.146944 | 0.082047 |
| B4M | 0.101860 | 0.138831 | 0.137490 | 0.144945 | 0.080849 |
| B4S | 0.100271 | 0.135779 | 0.135576 | 0.138099 | 0.091750 |
| B5 | 0.099007 | 0.136583 | 0.136012 | 0.141861 | 0.088159 |
| B6 | 0.100092 | **0.134667** | 0.134661 | 0.137563 | **0.077710** |

B2 相对 B6 在 5880、5971、5975 改善，在 5895 和 6015 分别恶化约 1.28% 和
3.74%。这解释了为什么“宏平均最优”和“严格跨场站门槛”给出不同选择。

### 7.4 复杂度对比

| 项目 | B2 | B6 | B2 相对 B6 |
| --- | ---: | ---: | ---: |
| 参数量 | 18,416 | 885,395 | -97.92% |
| 五场站模型文件合计近似值 | 0.261 MiB | 10.753 MiB | 显著降低 |
| 五场站训练总时间 | 约 395 s | 约 17,758.6 s | -97.78% |

### 7.5 “形式化选择 B6”与“后续采用 B2”并不矛盾

Stage 1 最初预声明规则要求：

- Macro NRMSE 不超过 B6 的 +0.5%；
- 至少 4/5 场站不劣于 B6 的 +1%；
- 合格后优先参数最少。

B2 虽然宏平均优于 B6，但只有 3/5 场站满足逐场条件，因此正式自动报告选中
B6。后续研究没有篡改该报告，而是基于新的研究问题选择 B2 作为
**Pareto research backbone**：

- B2 的 Macro NRMSE 最低；
- 只有 B6 的 2.08% 参数；
- 可将后续实验集中到“Persistence 与 corrected 两候选如何按工况融合”；
- B6 始终保留为复杂完整模型参考。

论文中必须同时报告这两个事实，不能写成“Stage 1 自动规则直接选出了 B2”。

### 7.6 阶段结论

- 删除任意单个复杂分支并不是导致性能崩溃的唯一解释；事实上 B2 最简单且
  宏平均最好；
- long、mid、short FeTS 组合没有显示与其复杂度相称的增益；
- 四专家防塌缩稀疏路由不再是下一阶段首要问题，因为研究主干已缩成两候选；
- 后续应优先研究显式工况编码和两候选动态融合，而不是继续压缩 B2。

### 7.7 结果路径

- 实验清单：`wind_results/fets_patchtst_min/stage1_variant_manifest.csv`
- 训练汇总：`wind_results/fets_patchtst_min/stage1_training_metrics.csv`
- 测试汇总：`wind_results/fets_patchtst_min/testdata_predict_output/fets_patchtst_min_test_metrics_summary.csv`
- 逐 horizon：`wind_results/fets_patchtst_min/testdata_predict_output/fets_patchtst_min_test_metrics_by_horizon_all.csv`
- 变体比较：`wind_results/fets_patchtst_min/testdata_predict_output/fets_patchtst_min_variant_comparison.csv`
- 正式选择报告：`wind_results/fets_patchtst_min/testdata_predict_output/fets_patchtst_min_minimal_effective_selection.md`
- 参数—NRMSE 图：`wind_results/fets_patchtst_min/testdata_predict_output/fets_patchtst_min_parameter_nrmse_pareto.png`
- 各变体模型、权重、history 和预测：`wind_results/fets_patchtst_min/<variant>/`

---

## 8. 第二阶段 A：R0–R6 显式工况编码器

### 8.1 实验目标和矩阵

B2 被解释为两个候选：

~~~text
Persistence candidate
Corrected candidate = Persistence + lightweight causal residual
~~~

R2–R5 均从同一 B2 最佳权重初始化，并给 corrected candidate 添加直接监督，
以缓解 `gate × residual` 的尺度不可辨识。

| 变体 | 结构 | 是否新训练 |
| --- | --- | --- |
| R0 | Stage 1 B0 Persistence | 否，直接引用 |
| R1 | Stage 1 B2 | 否，直接引用 |
| R2 | 两候选 + horizon-only 静态 sigmoid gate | 是 |
| R3 | 两候选 + 隐式 causal-Conv 动态 gate | 是 |
| R4 | 两候选 + 43 维显式风电工况统计 + 动态 gate | 是 |
| R5 | R4 + stable/up/down、低功率、变化幅度辅助任务 | 是 |
| R6 | Stage 1 B6 四专家完整模型 | 否，直接引用 |

R5 的未来工况标签只作为训练期辅助目标；推理仍只使用 96 步历史，不读取未来
真实功率或未来工况。

### 8.2 验证集结果

| 模型 | 参数量 | Macro Val NRMSE | 跨场站标准差 |
| --- | ---: | ---: | ---: |
| R0 | 0 | 0.114691 | 0.021832 |
| R1 | 18,416 | 0.108087 | 0.020856 |
| R2 | 18,432 | 0.108209 | 0.021155 |
| R3 | 20,129 | 0.108212 | 0.021156 |
| **R4** | **21,151** | **0.107904** | 0.020994 |
| R5 | 21,636 | 0.107951 | 0.021027 |
| R6 | 885,395 | 0.108655 | **0.020583** |

### 8.3 测试集描述性结果

| 模型 | 参数量 | Macro NMAE | Macro NRMSE | NRMSE 标准差 | 加权曲线 NRMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 0 | 0.077918 | 0.122664 | 0.027286 | 0.075992 |
| R1 | 18,416 | 0.080718 | 0.115700 | 0.023526 | 0.075581 |
| R2 | 18,432 | 0.080164 | 0.115434 | **0.023201** | 0.075469 |
| R3 | 20,129 | 0.080170 | 0.115439 | 0.023205 | 0.075474 |
| **R4** | **21,151** | **0.077789** | **0.113822** | 0.023874 | **0.074567** |
| R5 | 21,636 | 0.078608 | 0.114387 | 0.023820 | 0.074909 |
| R6 | 885,395 | 0.078293 | 0.116939 | 0.023985 | 0.075163 |

### 8.4 结果解释

- R4 同时取得最低验证和测试 NRMSE，测试相对 R1 改善约 1.62%，相对 R6
  改善约 2.67%，参数相对 R6 少约 97.61%；
- R2 与 R3 几乎相同，说明隐式卷积上下文没有产生可辨识的动态路由收益；
- R2/R3 的 corrected 高饱和率约为 0.975/0.997，近似“总选 corrected”；
- R4 的样本门控变化量增至约 0.076959，高饱和率降到约 0.723，显式统计确实
  增强了样本依赖；
- R5 相对 R4 测试 NRMSE 恶化约 0.50%，辅助工况 Macro F1 约 0.483，解释头
  没有转化为预测增益，因此不保留。

旧 R0–R6 的 oracle 校准字段未对所有非有限真值做严格逐点屏蔽，不能用于严谨
跨模型 Brier 增量结论；门控均值、变化量和饱和率仍可用于描述行为。

### 8.5 阶段结论

R4 证明显式风电工况统计优于 horizon-only 和隐式卷积门控，因此成为 F0–F8
特征组筛选的母结构。R5 同时说明“增加辅助任务”不会自动带来主任务提升。

### 8.6 结果路径

- 实验清单：`wind_results/regime_encoder_patchtst/stage2_experiment_manifest.csv`
- 训练汇总：`wind_results/regime_encoder_patchtst/stage2_training_metrics.csv`
- 验证比较：`wind_results/regime_encoder_patchtst/stage2_validation_comparison.csv`
- 测试比较：`wind_results/regime_encoder_patchtst/testdata_predict_output/regime_encoder_patchtst_test_comparison_descriptive.csv`
- 逐场站：`wind_results/regime_encoder_patchtst/testdata_predict_output/regime_encoder_patchtst_test_metrics_summary.csv`
- 逐 horizon：`wind_results/regime_encoder_patchtst/testdata_predict_output/regime_encoder_patchtst_test_metrics_by_horizon_all.csv`
- 分工况：`wind_results/regime_encoder_patchtst/testdata_predict_output/regime_encoder_patchtst_test_metrics_by_regime_all.csv`
- candidate：`wind_results/regime_encoder_patchtst/testdata_predict_output/regime_encoder_patchtst_test_candidate_all.csv`
- gate：`wind_results/regime_encoder_patchtst/testdata_predict_output/regime_encoder_patchtst_test_gate_all.csv`
- R2–R5 artifact：`wind_results/regime_encoder_patchtst/r2_horizon_gate/` 至 `r5_explicit_regime_aux/`

---

## 9. 第二阶段 B：F0–F8 特征筛选与 FP 冻结候选归因

### 9.1 特征组定义

| 组 | 维数 | 含义 |
| --- | ---: | --- |
| P | 20 | 功率末值、多窗口均值/斜率/波动/变化幅度、低功率占比等 |
| H | 12 | 轮毂高度风速末值、多窗口均值/斜率/波动/变化幅度 |
| M | 3 | 多高度风速均值、离散度、轮毂与多高度均值差 |
| D | 4 | 多滞后风向转角与平均转角 |
| C | 4 | 功率—风速斜率乘积、风速立方代理偏差、变化相关性 |

### 9.2 F0–F8 实验矩阵

| 变体 | 特征组 | 维数 | 参数量 | 作用 |
| --- | --- | ---: | ---: | --- |
| F0 | P | 20 | 20,553 | 仅功率状态 |
| F1 | P+H | 32 | 20,865 | H 增量 |
| F2 | P+H+M | 35 | 20,943 | M 增量 |
| F3 | P+H+M+D | 39 | 21,047 | D 增量 |
| F4 | P+H+M+D+C | 43 | 21,151 | 完整 R4，直接引用 |
| F5 | H+M+D | 19 | 20,527 | 删除 P |
| F6 | P+M+D | 27 | 20,735 | 删除 H |
| F7 | P+H+D | 36 | 20,969 | 删除 M |
| F8 | P+H+D+C | 40 | 21,073 | 无 M 时检验 C |

补充 F8 时，已有 F0–F7 没有重新训练；F4 直接引用 R4。

### 9.3 测试集正式排名

| 排名 | 模型 | 特征组 | 参数量 | Macro NMAE | Macro NRMSE | NRMSE 标准差 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | **F7** | **P+H+D** | **20,969** | **0.077609** | **0.113761** | 0.023811 |
| 2 | F4 | P+H+M+D+C | 21,151 | 0.077789 | 0.113822 | 0.023874 |
| 3 | F1 | P+H | 20,865 | 0.077883 | 0.113875 | 0.023852 |
| 4 | F3 | P+H+M+D | 21,047 | 0.078051 | 0.113998 | 0.024235 |
| 5 | F2 | P+H+M | 20,943 | 0.078172 | 0.114057 | 0.024199 |
| 6 | F6 | P+M+D | 20,735 | 0.078314 | 0.114122 | 0.024155 |
| 7 | F0 | P | 20,553 | 0.078319 | 0.114221 | 0.023869 |
| 8 | F8 | P+H+D+C | 21,073 | 0.078683 | 0.114354 | **0.023083** |
| 9 | F5 | H+M+D | 20,527 | 0.079848 | 0.115232 | 0.023788 |

### 9.4 关键父子比较

| 比较 | Macro NRMSE 相对变化 | 改善场站 | 结论 |
| --- | ---: | ---: | --- |
| F0→F1，加入 H | -0.303% | 4/5 | 轮毂高度风速有效 |
| F1→F7，加入 D，无 M/C | -0.100% | 4/5 | 风向变化有小幅稳定增量 |
| F7→F3，加入 M，无 C | +0.208% | 3/5 | 宏平均恶化，不保留 M |
| F3→F4，加入 C，有 M | -0.154% | 3/5 | 仅弱且不一致的 M×C 现象 |
| F7→F8，加入 C，无 M | **+0.521%** | 1/5 | C 在无 M 条件下明显无益 |
| F5→F3，加入 P | -1.072% | 3/5 | P 是不可删除的核心组 |
| F6→F3，加入 H | -0.109% | 4/5 | 反向消融再次支持 H |

### 9.5 为什么还需要 FP0/FP4

F0–F8 会联合微调 corrected candidate 和 gate。不同特征可能同时改变候选，
因此最终差值不能全部归因于门控特征。FP 控制固定完全相同的 B2 候选，只训练
工况编码器和门控：

| 探针 | 门控特征 | 总参数 | 可训练参数 | 冻结参数 |
| --- | --- | ---: | ---: | ---: |
| FP0 | P+H+D | 20,969 | 2,553 | 18,416 |
| FP4 | P+H+D+C | 21,073 | 2,657 | 18,416 |

FP 不参与 F0–F8 正式排名，它回答的是“在候选不变时，C 是否直接改善门控”。

### 9.6 FP0/FP4 结果

| 指标 | FP0 | FP4 | FP4−FP0 |
| --- | ---: | ---: | ---: |
| Macro NRMSE | 0.115080 | 0.115053 | -0.00002688，-0.0234% |
| Oracle Brier | 0.505791 | 0.503658 | -0.002132 |
| ECE-10 | 0.512659 | 0.510147 | -0.002512 |
| corrected 优/劣门控间隔 | 0.014225 | 0.015495 | +0.001270 |
| Positive regret | 0.016887 | 0.016805 | -0.000082 |
| Harm rate | 0.462297 | 0.461879 | -0.000418 |

预声明实践标准要求至少 4/5 场站达到相应阈值；以上最多只有 3/5，因此均判为
“无明确实践效应”。端到端 F7→F8 又显示 C 使 NRMSE 恶化 0.521%。两类证据
一致支持删除 C。

### 9.7 Candidate drift、复杂度和稳定性

- F0–F8 corrected candidate 的宏平均 NRMSE 相对跨度为 0.6478%，超过预设
  0.2% 漂移阈值；独立特征归因必须使用 FP；
- FP0/FP4 的五场站缩放候选、物理候选、oracle 标签和有效掩码一致性全部通过；
- F7 参数量 20,969，float32 理论参数存储 83,876 bytes；
- F7 `.keras` 平均约 366,233 bytes，权重文件平均约 353,096 bytes；
- F7 五场站平均训练时间约 160.91 s，平均 11 epoch；
- 跨场站 NRMSE 标准差 0.023811，范围 0.076720–0.133401；
- 这只是单 seed 下的空间稳定性，不是随机种子稳定性；
- 延迟、FLOPs、吞吐和峰值显存没有在统一硬件重测，正式报告明确留空。

### 9.8 最终 F7 结构

~~~text
两候选：Persistence / Persistence + lightweight causal residual
36维显式特征：P+H+D
LayerNorm
Dense(24, GELU) → Dropout(0.1) → Dense(24, GELU)
与8维 horizon embedding 合并
hidden_dim=16 的 sample × horizon sigmoid gate
逐点凸融合
训练损失包含 0.5 × corrected candidate direct supervision
总参数：20,969
~~~

F7 相对 F4 的 Macro NRMSE 仅改善约 0.0534%，且严格逐场只在 1/5 场更优。
因此 F7 是当前测试排名下更简洁的最优结构，但不能写成多 seed 显著性结论。

### 9.9 结果路径

- F0–F8 正式报告：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f0_f8_test_selection_output/feature_screening_f0_f8_test_final_selection.md`
- 变体比较：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f0_f8_test_selection_output/feature_screening_f0_f8_test_variant_comparison.csv`
- F7 artifact：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f7_no_multiheight/`
- F8 artifact：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f8_no_multiheight_with_consistency/`
- FP0：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/fp0_frozen_candidate_phd/`
- FP4：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/fp4_frozen_candidate_phdc/`
- Frozen-Pair 报告：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/frozen_pair_probe_test_output/feature_screening_frozen_pair_control_report.md`
- C 特征结论：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f0_f8_probe_analysis_output/feature_screening_c_feature_conclusion.md`
- 复杂度、漂移和门控报告：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f0_f8_probe_analysis_output/feature_screening_complexity_drift_gate_report.md`
- 完成标记：`wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f0_f8_probe_analysis_output/feature_screening_f0_f8_fp_bundle_complete.json`

---

## 10. 第三阶段：G0–G4 门控校准与 Persistence 保护

### 10.1 为什么仍需第三阶段

Stage 2 的 B2/F7 已不再包含 long、mid、short 多专家，因此原计划的 top-k、
负载均衡和四专家防塌缩稀疏路由不再匹配当前模型。该方向被有理由地取消，而
不是遗漏。

两候选门控仍存在三个真实问题：

- G0 corrected 高饱和率较高；
- gate 是否选择到更优候选缺少严格校准；
- 错误选择 corrected 时缺少 Persistence 安全回退。

因此第三阶段把“防塌缩路由”改写为更适合两候选结构的门控校准、regret/harm
诊断和 Persistence abstention。

### 10.2 实验矩阵

固定候选和输入：

~~~text
候选1：Persistence
候选2：Persistence + lightweight corrected residual
显式工况：F7 的 P+H+D，36维
seed=2026，batch_size=192，参数上限30,000
~~~

| 变体 | 结构 | 是否新训练 |
| --- | --- | --- |
| G0 | F7 非因子化 sample×horizon sigmoid gate | 否，直接引用 F7 |
| G1 | 因子化 `π(i,h)=q(i)×s(h)` + dynamic supervision | 是 |
| G2 | 非因子化 gate + soft-oracle calibration + Persistence safety | 是 |
| G3 | 因子化 gate + calibration + dynamic supervision + safety | 是 |
| G4 | 对同一次 G3 输出按统一阈值 κ 做 Persistence abstention | 否，后处理 |
| Hard top-1 | G3 gate 硬离散化 | 否，仅负对照 |

基础和附加损失：

~~~text
L_base = L_fused + 0.5 × L_corrected
G1 = L_base + 0.05 L_dynamic
G2 = L_base + 0.10 L_cal + 0.05 L_safe
G3 = L_base + 0.10 L_cal + 0.05 L_dynamic + 0.05 L_safe
κ ∈ {0.45, 0.50, 0.55, 0.60, 0.65}
~~~

实际新训练 `G1–G3 × 5=15` 个模型；G0 和 G4 不重复训练。

### 10.3 正式晋级门槛

新增模型必须同时通过：

- Macro NRMSE 不超过 G0 的 +0.2%；
- 至少 4/5 场站不超过 G0 的 +1%；
- stable 和 low-power NRMSE 分别至少改善 10% 和 5%；
- 相对 Persistence 的 stable/low-power 缺口分别关闭至少 25%/20%；
- ramp-up 和 ramp-down 均不得恶化超过 0.5%；
- 固定 G0 oracle 下 Brier 至少改善 10%，ECE 至少改善 15%；
- corrected 高饱和率低于 50%；
- dynamic 与 stable 的平均门控差至少 0.15；
- stable/low-power 安全轴不得恶化，且 regret 或 harm 至少一项改善 20%；
- 参数量低于 30,000。

G4 还需同时相对 G0 和 G3 通过精度与安全门槛。只有全部守门通过后才按 NRMSE
排序；0.1% 内近似平局再看 regret、Brier、参数和推理时间。

### 10.4 测试集宏平均结果

下表校准指标采用固定 G0 候选 oracle，避免 candidate drift 改变分类任务。

| 模型 | Macro NRMSE | 相对 G0 | Macro NMAE | Positive regret | Harm rate | 固定 G0 Brier | 固定 G0 ECE | 高饱和率 | 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | **0.113606** | **-0.136%** | **0.075185** | 0.010486 | 0.352006 | 0.249620 | 0.130023 | 0 | 20,409 |
| **G0** | 0.113761 | 0 | 0.077609 | 0.015589 | 0.430711 | 0.408621 | 0.415817 | 0.744590 | 20,969 |
| G2 | 0.113780 | +0.017% | 0.075396 | 0.010230 | 0.349257 | **0.237420** | **0.072978** | 0 | 20,969 |
| G3 | 0.113813 | +0.046% | 0.075194 | **0.009984** | **0.346064** | 0.237693 | 0.078691 | 0 | 20,409 |
| G4，κ=0.65 | 0.119054 | +4.653% | 0.076344 | 0.003226 | 0.070884 | 0.237693 | 0.078691 | 0 | 20,409 |

### 10.5 各场站 NRMSE

| 模型 | 5880 | 5895 | 5971 | 5975 | 6015 |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 | **0.094051** | 0.132413 | **0.133401** | 0.132219 | 0.076720 |
| G1 | 0.094305 | **0.131439** | 0.134517 | **0.132173** | **0.075597** |
| G2 | 0.094368 | 0.132033 | 0.134236 | 0.132367 | 0.075898 |
| G3 | 0.094599 | 0.131820 | 0.134291 | 0.132453 | 0.075903 |
| G4 | 0.099274 | 0.136034 | 0.138909 | 0.142369 | 0.078683 |

### 10.6 为什么数值最低的 G1 没有被选中

| 模型 | Stable 改善 | Low-power 改善 | Ramp-up 变化 | Ramp-down 变化 | 固定 Brier 改善 | 固定 ECE 改善 | 正式资格 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G1 | 48.90% | 26.71% | **+0.586%** | +0.193% | 38.91% | 68.73% | 不通过 |
| G2 | 37.09% | 19.99% | **+0.680%** | +0.359% | 41.90% | 82.45% | 不通过 |
| G3 | 47.36% | 25.88% | **+0.836%** | +0.340% | 41.83% | 81.08% | 不通过 |
| G4 | 82.78% | 39.04% | **+4.613%** | **+6.336%** | 41.83% | 81.08% | 不通过 |

G1–G3 的唯一共同硬失败项是 ramp-up 超过 +0.5% 上限。G1 虽然总体 NRMSE
最低，但不能绕过预声明门槛。正式状态：

~~~text
fallback_g0_no_candidate_passed_guards
~~~

### 10.7 G4 阈值比较

| κ | Macro NRMSE | 相对 G0 | Positive regret | Harm rate | Corrected coverage | 严格合格 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.45 | 0.114409 | +0.569% | 0.007908 | 0.243882 | 0.671436 | 否 |
| 0.50 | 0.114920 | +1.019% | 0.007435 | 0.223931 | 0.611564 | 否 |
| 0.55 | 0.115694 | +1.699% | 0.006686 | 0.193662 | 0.516224 | 否 |
| 0.60 | 0.116890 | +2.751% | 0.005311 | 0.137028 | 0.348103 | 否 |
| 0.65 | 0.119054 | +4.653% | **0.003226** | **0.070884** | 0.174932 | 否，内部安全 fallback |

κ=0.65 将 regret 和 harm 相对 G0 分别降低约 79.31% 和 83.54%，但 NRMSE
恶化约 4.65%，且 0/5 场站满足相对 G0 的 +1% 保护。Hard top-1 的 Macro
NRMSE 为 0.119367，同样不合格。

### 10.8 阶段结论

- 因子化、校准和安全损失消除了 G0 的高饱和，并显著改善校准与安全；
- 这些改善没有同时通过 ramp 精度门槛；
- abstention 能控制风险，但当前保护过强导致总体精度不可接受；
- 最终仍选 G0/F7，即 20,969 参数的非因子化两候选门控；
- 可以把 G0–G4 写成系统的校准和安全机制研究，不能称最终模型已采用 G2/G3/G4。

### 10.9 结果路径

- 训练完成：`wind_results/controlled_gate_cali/controlled_gate_cali_training_bundle_complete.json`
- 训练汇总：`wind_results/controlled_gate_cali/controlled_gate_cali_training_metrics.csv`
- 实验清单：`wind_results/controlled_gate_cali/controlled_gate_cali_experiment_manifest.csv`
- 测试完成：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_bundle_complete.json`
- 最终选型：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_final_selection.md`
- 变体比较：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_variant_comparison.csv`
- 逐场站：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_metrics_summary.csv`
- 逐 horizon：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_metrics_by_horizon.csv`
- candidate：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_candidate_metrics.csv`
- 分工况：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_regime_metrics.csv`
- 安全诊断：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_gate_safety.csv`
- 校准：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_reliability.csv`
- G4 κ：`wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_g4_kappa_test_selection.csv`
- G1–G3 artifact：`wind_results/controlled_gate_cali/g1/`、`g2/`、`g3/`

---

## 11. 第四阶段：最小 residual 与 T0/M0/T1–T3 时频矩阵

### 11.1 5.1 启动条件及判断

原方案规定满足任一条件才启动 corrected residual 增强：

1. G4 已通过安全目标，但总体精度仍不足；
2. oracle ceiling 显示 gate 还有空间，但 corrected 候选优势不足；
3. G2/G3 校准成功，但 dynamic/ramp 候选质量成为主要瓶颈。

不得只因“论文缺结构创新”直接训练完整时频跨尺度模型。每次 corrected candidate
改变后，必须重建 train-only soft oracle、逐 horizon `|C-P| Q90`，重新校准
gate，并重新检查 Persistence 安全和 ramp 门槛。

第三阶段出现“校准和安全改善、ramp 不晋级”，满足启动条件。这里的“最小
residual”是控制父结构，不等同于完整 T1–T3 时频创新。

### 11.2 实验矩阵

| 变体 | 结构 | 作用 |
| --- | --- | --- |
| T0 | Stage 3 G0/F7 直接引用 | 原 gate、原 candidate 的正式基线 |
| M0 | 冻结 F7 corrected，无 adapter；重训统一因子化校准安全 gate | 隔离新 gate 本身的影响 |
| T1 | F7 residual + 零初始化轻量因果时间 adapter | 检验历史时间变化增强 |
| T2 | F7 residual + 仅读 96 步历史功率的 rFFT adapter | 检验历史频域增强 |
| T3 | 时间表示 + 频率表示 + 逐维乘性交互 | 检验轻量时频互补 |

M0/T1/T2/T3 使用相同 `P+H+D` 输入和统一 factorized calibrated safe gate。
T1–T3 先训练新增 adapter，再冻结 candidate 训练 gate。实际新训练
`M0/T1/T2/T3 × 5=20` 个模型；T0 不训练、不 forward、不复制 candidate archive。

训练/预测代码中保留 Stage 3 的 oracle、校准和安全逻辑，是为了对每个新 candidate
重新校准同一 gate，并不会重复训练 G0–G4。

### 11.3 正式选择门槛

- Macro NRMSE 不超过 T0 的 +0.2%；
- 至少 4/5 场站不超过 T0 的 +1%；
- fused ramp-up/down 不超过 T0 的 +0.5%；
- corrected 总体 NRMSE/NMAE 不超过 T0 的 +0.2%；
- corrected dynamic/ramp-up/ramp-down 不超过 T0 的 +0.5%；
- Positive regret 不超过 T0 的 +0.5%；
- Harm rate 不超过 T0 的绝对值 +0.002；
- 参数量低于 30,000。

### 11.4 测试集宏平均结果

| 模型 | Macro NRMSE | 相对 T0 | Macro NMAE | Corrected NRMSE | Corrected 相对 M0 | Positive regret | Harm rate | 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **T0** | **0.113761** | — | 0.077609 | 0.116160 | — | 0.015589 | 0.430711 | 20,969 |
| M0 | 0.114446 | +0.602% | 0.075282 | 0.116160 | 0 | 0.008442 | 0.324892 | **20,409** |
| T1 | 0.114491 | +0.642% | 0.075281 | 0.115681 | -0.413% | 0.008375 | 0.321822 | 23,561 |
| T2 | 0.114466 | +0.620% | 0.075257 | 0.116150 | -0.009% | 0.008415 | 0.324785 | 21,161 |
| T3 | 0.114492 | +0.642% | **0.075253** | **0.115577** | **-0.502%** | **0.008332** | **0.319355** | 24,697 |

### 11.5 各场站 NRMSE

| 模型 | 5880 | 5895 | 5971 | 5975 | 6015 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 | **0.094051** | 0.132413 | **0.133401** | **0.132219** | 0.076720 |
| M0 | 0.095817 | 0.131804 | 0.134953 | 0.133651 | **0.076004** |
| T1 | 0.095839 | 0.131779 | 0.134992 | 0.133792 | 0.076053 |
| T2 | 0.095841 | 0.131776 | 0.134993 | 0.133697 | 0.076021 |
| T3 | 0.095834 | **0.131775** | 0.135008 | 0.133813 | 0.076029 |

所有新增模型只在 5895、6015 改善，在另三站恶化超过 1%，因此逐场站门槛均
只有 2/5 通过。

### 11.6 分工况与守门结果

| 模型 | Fused dynamic | Fused ramp-up | Fused ramp-down | Corrected dynamic | Corrected ramp-up | Corrected ramp-down | 未通过项 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T0 | **0.122018** | **0.133931** | 0.111065 | 0.122988 | 0.134890 | 0.112054 | 无 |
| M0 | 0.123376 | 0.135406 | 0.112433 | 0.122988 | 0.134890 | 0.112054 | 宏精度、逐场、ramp |
| T1 | 0.123426 | 0.135527 | 0.112459 | 0.122680 | **0.134829** | 0.111569 | 宏精度、逐场、ramp |
| T2 | 0.123403 | 0.135892 | **0.112023** | 0.123045 | 0.135607 | 0.111499 | 宏精度、逐场、ramp、candidate 工况 |
| T3 | 0.123432 | 0.135633 | 0.112362 | **0.122577** | 0.134861 | **0.111319** | 宏精度、逐场、ramp |

### 11.7 机制解释

1. **M0 定位到 gate 转化问题。**
   M0 与 T0 使用相同冻结 candidate，但 M0 换成统一因子化校准安全 gate，
   Macro NRMSE 已恶化 0.602%。多数 fused 损失来自 gate 拓扑/目标，而不是
   adapter 本身。

2. **T3 是最好 candidate，不是最好最终模型。**
   T3 corrected 相对 M0 改善 0.502%，但 fused 相对 T0 恶化 0.642%，且只在
   2/5 场站改善。

3. **T3 的安全和 NMAE 改善不能覆盖主精度失败。**
   T3 相对 T0 的 NMAE、regret、harm 有改善，但 NRMSE、跨场站和 ramp 硬门槛
   失败，因此不能替代 T0。

4. **联合互补性很弱。**
   描述性对照 `T1 + T2 - M0 - T3` 的 overall corrected NRMSE 约
   `+9.35×10^-5`，仅 3/5 场站为正。T3 又不是严格复用 T1/T2 主效应头的
   2×2 因子设计，因此不能宣称因果时频交互。

### 11.8 阶段结论

正式状态：

~~~text
fallback_t0_no_new_variant_passed_guards
~~~

- 在 Stage 4 当时，五场站最终 fused 预测仍选择 T0/G0/F7；
- 若只看新 fused 模型，M0 最低；若只看 corrected candidate，T3 最低；
- 两者都不能替代正式目标下的 T0；
- T1–T3 应作为有价值的否定消融；
- 当前瓶颈是 candidate 优势的跨场站一致性和 gate 收益转化，而非简单缺少
  更大的时频网络；
- 当时不应立即堆叠完整 fine/mid/coarse 或 token 级跨尺度结构；后续先完成
  Stage 4B 收益闭环，才按受控矩阵启动 Stage 5A 静态表示。

### 11.9 结果和可视化路径

- 训练完成：`wind_results/time_freq_model/time_freq_model_training_bundle_complete.json`
- 训练汇总：`wind_results/time_freq_model/time_freq_model_training_metrics.csv`
- 实验清单：`wind_results/time_freq_model/time_freq_model_experiment_manifest.csv`
- 测试完成：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_bundle_complete.json`
- 最终选型：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_final_selection.md`
- 变体比较：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_variant_comparison.csv`
- 逐场站：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_summary.csv`
- 逐 horizon：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_horizon.csv`
- candidate：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_candidate.csv`
- candidate drift：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_candidate_drift.csv`
- 分工况：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_regime.csv`
- 安全：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_safety.csv`
- 校准：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_calibration.csv`
- 复杂度：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_complexity.csv`
- 联合互补性：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_joint_complementarity.csv`
- T0 复用清单：`wind_results/time_freq_model/testdata_predict_output/time_freq_model_t0_source_reuse_manifest.csv`
- 汇总图：`wind_results/time_freq_model/testdata_predict_output/figures/`
- M0/T1/T2/T3 artifact：`wind_results/time_freq_model/m0/`、`t1/`、`t2/`、`t3/`
- 补图清单：`wind_results/time_freq_model/time_freq_model_visualization_backfill_inventory.csv`
- 补图 manifest：`wind_results/time_freq_model/time_freq_model_visualization_backfill_manifest.json`

---

## 12. 第四阶段 B：D0–D3 门控收益转化闭环

### 12.1 为什么在 T0–T3 后补做闭环

Stage 4 已经观察到 T1/T3 corrected candidate 有小幅收益，但统一新 gate 的 fused
结果退化。为区分“candidate 无效”“gate 拓扑无效”和“oracle/辅助目标错配”，
Stage 4B 固定 candidate 身份，按每个 candidate 重新生成 train-only soft oracle
和逐 horizon `|C-P| Q90`，只比较收益转化链路。D0 只读引用 T0/G0/F7，避免
重复训练和重复预测。

### 12.2 五变体矩阵

| 变体 | Candidate | Gate/目标 | 受控问题 |
| --- | --- | --- | --- |
| D0 | F7 | 原非因子化 G0 | 正式部署参考，不重训 |
| D0R | 冻结 T1 | 非因子化 direct gate，无校准/安全辅助 | T1 + 新 direct gate 能否转化收益 |
| D1 | 冻结 F7 | 非因子化 calibrated-safe，`calibration=0.1`、`safety=0.05` | 同 F7 下辅助目标效应 |
| D2 | 冻结 T1 | 与 D1 同构的非因子化 calibrated-safe | 同 gate 下 T1 相对 F7 的 candidate 效应 |
| D3 | 冻结 T1 | 因子化 calibrated-dynamic-safe，另加 `dynamic=0.05` | 因子化拓扑与 dynamic 辅助项的联合效应 |

D1/D2/D3 使用各自 train-only oracle/Q90。`fixed-G0-on-T1 replay` 只用于诊断
“重训 gate 是否比原 G0 更好”，不进入五变体正式排名。D3 相对 D2 同时改变
拓扑和 dynamic 辅助项，因此只能解释为联合效应，不能归因为单一因子化结构。

实际训练 `D0R/D1/D2/D3 × 5=20` 个模型；D0 五场站全部只读复用。训练固定
`seed=2026`、`batch_size=192`，全部 candidate identity 与 oracle/Q90 统计审计
通过。

### 12.3 测试集结果

| 变体 | Macro NRMSE | Macro NMAE | 相对 D0 NRMSE | 不退化/严格改善场站 | 参数量 | 正式守门 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| D0 | **0.113760989** | 0.077608814 | 0 | 5/5（参考） | 20,969 | **通过，最终选择** |
| D0R | **0.113721293** | 0.077472461 | -0.0349% | 2/5、2/5 | 24,121 | 失败 |
| D1 | 0.114362328 | 0.075474 | +0.5286% | 2/5、2/5 | 20,969 | 失败 |
| D2 | 0.114357238 | 0.075401 | +0.5241% | 2/5、2/5 | 24,121 | 失败 |
| D3 | 0.114485458 | 0.075246 | +0.6368% | 2/5、2/5 | 23,561 | 失败 |

预声明门槛要求新变体 Macro NRMSE 至少改善 0.2%、至少 4/5 场站不退化且
3/5 严格改善，ramp-up/down 不超过 D0 的 +0.5%，并通过 candidate、安全和
30k 参数守门。D0R 虽是数值最低 fused NRMSE，但只改善 0.0349% 且仅 2/5
场站改善，不能正式晋级。D1–D3 的 NMAE、校准或安全指标有所改善，但 NRMSE
明显退化。

同一 T1 candidate 的 fixed-G0 replay NRMSE 为 0.113649；D0R 反而比它差
0.0639%，说明本轮 gate 重训没有转化 candidate 潜力。最终正式回退 **D0**。

### 12.4 代码、结果与完成标记

- 训练：`wind_time_freq_model_stage4b_train.py`
- 预测：`wind_time_freq_model_stage4b_predict.py`
- 结果根目录：`wind_results/time_freq_model/supplement_round2_stage4b_gate_closure/`
- 实验矩阵：`stage4b_gate_closure_experiment_manifest.csv`
- 正式报告：`testdata_predict_output/stage4b_gate_closure_test_final_selection.md`
- 正式比较：`testdata_predict_output/stage4b_gate_closure_test_variant_comparison.csv`
- 训练完成标记：`stage4b_gate_closure_training_bundle_complete.json`
- 测试完成标记：`testdata_predict_output/stage4b_gate_closure_test_bundle_complete.json`
- 变体 artifact：`d0r/`、`d1/`、`d2/`、`d3/`
- 正式可视化 inventory：`testdata_predict_output/stage4b_gate_closure_visual_inventory.csv`

正式输出目录还保存 summary、horizon、candidate、regime、safety、calibration、
complexity、controlled contrasts、candidate invariants 和 source reuse manifest。
预测 marker 记录正式可视化 67 张；四个新变体各有五场站训练 history 图。

---

## 13. 第五阶段 A：轻量 fine/mid/coarse 历史表示

### 13.1 研究问题与结构边界

Stage 5A 先回答“多尺度历史表示本身是否改善 corrected candidate”，不直接训练
token 级跨尺度交互。四个新变体全部冻结 F7 residual、P+H+D context 和原 G0，
只训练零初始化的 candidate-delta adapter；正式主指标是 **corrected-candidate
NRMSE**，冻结 G0 的 fused 回放只作诊断。

这些 adapter 不是 PatchTST：代码没有接入 PatchTST encoder 或 Transformer
patch token。各尺度以 causal Conv1D（8 filters，stride=1）提取局部表示，再做
右对齐抽样、LayerNorm、global average/max pooling 和轻量 Dense 映射。`patch`
与 `stride` 只定义历史感受野和右对齐采样间隔。

### 13.2 X0–X1 矩阵

| 变体 | 尺度结构 | Candidate adapter | Token 交互 | 总参数量 |
| --- | --- | --- | --- | ---: |
| X0 | D0/G0/F7 直接引用 | 无 | 无 | 20,969 |
| X1-F | fine：patch=4、stride=2 | 单尺度表示→Dense(32)→16步 delta | 无 | 22,369 |
| X1-M | mid：patch=8、stride=4 | 单尺度表示→Dense(32)→16步 delta | 无 | 22,401 |
| X1-C | coarse：patch=16、stride=8 | 单尺度表示→Dense(32)→16步 delta | 无 | 22,465 |
| X1 | fine/mid/coarse 分别编码至 16 维后 concat + static Dense | 三尺度静态融合→16步 delta | 无 | 24,177 |

X0 五场站只读复用，新训练 `X1-F/M/C/X1 × 5=20` 个模型；`seed=2026`、
`batch_size=192`。所有 source/candidate 不变量审计通过。

### 13.3 Candidate 测试结果与形式化选择

| 变体 | Candidate NRMSE | Candidate NMAE | 相对 X0 改善 | 不退化/严格改善场站 | Frozen-G0 fused 诊断 | 守门/选择 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| X0 | 0.116159752 | 0.081113333 | 0 | 5/5（参考） | 0.113760989 | 参考通过 |
| X1-F | 0.115789078 | 0.080402038 | 0.3191% | 4/5、4/5 | 0.113716658 | **通过并正式选择** |
| X1-M | 0.115834 | 0.080443 | 0.2805% | 4/5、4/5 | 0.113750 | 未达到 0.3% |
| X1-C | **0.115680075** | **0.080193012** | **0.4129%** | 4/5、4/5 | 0.113691625 | 通过，数值最低 |
| X1 | 0.115705166 | 0.080242515 | 0.3913% | 4/5、4/5 | **0.113690771** | 通过 |

预声明条件要求 candidate Macro NRMSE 至少改善 0.3%、至少 4/5 场站不退化、
3/5 严格改善，dynamic/ramp 不恶化超过 0.5%，删除最大收益场后仍为正且参数
小于 30k。X1-F、X1-C、X1 通过，X1-M 未过 0.3% 门槛。

X1-C 是 candidate 数值最低模型；X1-F 与它只差约 0.0943%，处在预声明的
0.1% 最优带内，因此再按参数量/效率优先，Stage 5A **形式化选择 X1-F**。
后续 X1R 特意使用 full X1，是为了测试完整三尺度 candidate 在相同旧 G0 与新
gate 下的收益转化；这不改写 Stage 5A 的正式选择，也不能把 X1-fixed-G0 当作
本阶段可部署最优。

### 13.4 阶段结论

- 轻量静态历史表示在 candidate 层面有小幅、但非五场站一致的收益；
- coarse 单尺度数值最好，full 静态融合没有明显超过 coarse，尚不能证明尺度
  互补或真正跨尺度交互；
- frozen G0 只转化了约 0.01%–0.06% 的 fused 收益，仍需同 candidate 闭环；
- Stage 5A 不是 PatchTST，也不是 X2–X6 token 交互创新。

### 13.5 代码、结果与完成标记

- 训练：`wind_multiscale_correc_cand_train.py`
- 预测：`wind_multiscale_correc_cand_predict.py`
- 结果根目录：`wind_results/multiscale_correc_cand/`
- 实验矩阵：`multiscale_correc_cand_experiment_manifest.csv`
- 正式报告：`testdata_predict_output/multiscale_correc_cand_test_final_selection.md`
- 正式比较：`testdata_predict_output/multiscale_correc_cand_test_variant_comparison.csv`
- 训练完成标记：`multiscale_correc_cand_training_bundle_complete.json`
- 测试完成标记：`testdata_predict_output/multiscale_correc_cand_test_bundle_complete.json`
- 新变体 artifact：`x1_f/`、`x1_m/`、`x1_c/`、`x1/`
- 可视化 inventory：`testdata_predict_output/multiscale_correc_cand_visual_inventory.csv`

测试 marker 记录正式可视化 96 张；每个新变体各保存五场站 candidate history，
并保存 prediction、candidate archive、frozen-G0 replay、逐 horizon、逐工况、
复杂度和不变量文件。

---

## 14. 第五阶段 A 闭环：full-X1 的 X1R 门控重校准

### 14.1 为什么单独做 X1R

Stage 5A 的 full X1 corrected candidate 比 X0 改善 0.3913%，但冻结旧 G0 后
fused 只改善 0.0617%。X1R 固定 full X1 的 Persistence/corrected 两候选，剪除
旧 G0，重新生成每场站 train-only soft oracle 和逐 horizon Q90，只训练一个与
Stage 4B D2 同构的非因子化 sample×horizon calibrated-safe gate。这样可在
candidate 完全相同的条件下，将“旧 G0”和“新门控”的影响分开。

X1R 仍使用 36 维 P+H+D context，`calibration_weight=0.1`、
`safety_weight=0.05`、`dynamic_weight=0`。总参数 24,177；阶段可训练参数为
gate-only 993、context/objective 2,553。固定 `seed=2026`、`batch_size=192`，
五场站 candidate max/mean drift 均为 0。

### 14.2 正式对照与宏平均结果

| 变体 | 身份 | Macro NRMSE | Macro NMAE | Regret | Harm@0.005 | Brier | ECE | 高饱和率 | 可选 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| X0 | D0/F7 + 原 G0 部署参考 | **0.113760989** | 0.077608814 | 0.015589 | 0.430711 | 0.408621 | 0.415817 | 0.744590 | 是 |
| X1-fixed-G0 | full X1 + 原 G0 回放 | **0.113690771** | 0.077242222 | 0.015081 | 0.415879 | 0.408727 | 0.408654 | 0.744590 | 否，仅诊断 |
| X1R | full X1 + 新 calibrated-safe gate | 0.114369918 | **0.075363723** | **0.008739** | **0.323984** | **0.240506** | **0.072355** | **0** | 是 |

X1-fixed-G0 是全表 NRMSE 数值最低，但属于诊断模型，预注册时不具备正式部署
资格。正式排名只比较 X0 与 X1R。X1R 相对 X0：

- NRMSE 恶化 0.5353%，相对同 candidate 的 X1-fixed-G0 恶化 0.5974%；
- NMAE 改善 2.8928%；regret 降 43.94%，harm 降 24.78%；
- Brier 降 41.14%，ECE 降 82.60%，高饱和率从 74.46% 降至 0；
- 只在 2/5 场站改善；删除最大收益场站后，余下四站反而恶化 0.7392%；
- 16 个 horizon 全部退化，幅度约 0.219%–0.674%。

### 14.3 逐场站与工况解释

| 场站尾号 | X1R 相对 X0 NRMSE | 结论 |
| --- | ---: | --- |
| 5880 | +1.6255% | 退化 |
| 5895 | -0.3352% | 改善 |
| 5971 | +1.1606% | 退化 |
| 5975 | +0.7595% | 退化 |
| 6015 | -0.7726% | 改善 |

X1R 在 dynamic、ramp-up、ramp-down、`change_ge_20` 上分别恶化约 1.0104%、
1.1353%、0.9966%、4.0691%；但 stable/`change_00_02` 改善约 42.67%，low-power
改善约 22.98%。新 gate 更保守、校准更好，显著降低大量小误差和低功率误差，
所以 NMAE 下降；同时对大变化/尾部过度靠近 Persistence，平方误差主导的 NRMSE
反而恶化。这是明确的 L1—L2—ramp 目标错配，而不是简单的训练失败。

### 14.4 最终定型与 Stage 5B 决策

X1R 通过 candidate 身份、参数、Brier/ECE、regret/harm 和饱和率守门，但失败于
宏 NRMSE、同 candidate 优越性、逐场站一致性、留一稳健性和 dynamic/ramp。
因此：

- 当前最终部署结构仍为 **X0=D0=T0=G0=F7**，20,969 参数；
- Stage 5A 的 X1-F 是 candidate 研究的形式化最优，不替代部署 X0；
- X1-fixed-G0 是诊断数值最优，不具备正式选择资格；
- `selection_guard_pass=false`，`stage5b_x2_x3_unlocked=false`；
- 停止 X2–X6，不因“论文缺 token 交互”越过启动门槛。

即使 X1R 全部通过，原协议也只允许先启动 X2/X3 单向交互，不能直接跳到 X6。
每个新 candidate 仍必须重建自己的 train-only oracle/Q90 并重新校准 gate。

### 14.5 代码、结果与完成标记

- 训练：`wind_multiscale_correc_cand_x1r_train.py`
- 预测：`wind_multiscale_correc_cand_x1r_predict.py`
- 结果根目录：`wind_results/multiscale_correc_cand/x1r_gate_closure/`
- 实验矩阵：`x1r_gate_closure_experiment_manifest.csv`
- 正式报告：`testdata_predict_output/x1r_gate_closure_test_final_selection.md`
- 正式比较：`testdata_predict_output/x1r_gate_closure_test_variant_comparison.csv`
- 训练完成标记：`x1r_gate_closure_training_bundle_complete.json`
- 测试完成标记：`testdata_predict_output/x1r_gate_closure_test_bundle_complete.json`
- 正式可视化 inventory：`testdata_predict_output/x1r_gate_closure_visual_inventory.csv`

正式输出还包括 summary、horizon、regime、safety、calibration、candidate、
complexity、controlled contrasts、Stage 5A candidate evidence、candidate invariants
和 source reuse manifest。预测 marker 记录 36 张正式可视化；X1R 五场站 history
以及宏、逐 horizon、逐场站、可靠性与安全图均已归档。

---

## 15. 跨阶段数据比较

> 本节全部数值只属于原 5 个开发场站，不得与第 28 节外部 14 站的
> trNRMSE/trNMAE 直接混排，也不能用 0.113761 与 0.128801 的大小判断模型发生
> 退化。二者的场站、功率参考值和预处理协议均不同。

### 15.1 当前统一 seed 主线

| 节点 | 结构 | Macro NRMSE | 参数量 | 相对上一关键节点 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| B0 | Persistence | 0.122664 | 0 | — | 物理基线 |
| B1 | Long PatchTST | 0.121784 | 210,960 | -0.72% vs B0 | 单 long 收益有限 |
| B6 | 四专家动态模型 | 0.116939 | 885,395 | -3.98% vs B1 | 复杂完整参考 |
| B2/R1 | Persistence + residual | 0.115700 | 18,416 | -1.06% vs B6 | 最优轻量主干 |
| R4/F4 | 完整显式工况门控 | 0.113822 | 21,151 | -1.62% vs B2 | 显式工况有效 |
| F7/G0/T0/D0/X0 | P+H+D 显式门控 | **0.113761** | 20,969 | -0.053% vs F4 | 当前正式最终 |
| G1 | 因子化动态监督 | **0.113606** | 20,409 | -0.136% vs G0 | 数值最低但 ramp 不合格 |
| T3 | 时频 candidate + 新 gate | 0.114492 | 24,697 | +0.642% vs T0 | candidate 改善未转化 |
| D0R | T1 candidate + 重训 direct gate | 0.113721 | 24,121 | -0.0349% vs D0 | 数值小幅改善但仅 2/5 场站改善 |
| X1-fixed-G0 | full X1 candidate + 原 G0 | 0.113691 | 24,177 | -0.0617% vs X0 | 诊断模型，不具备选择资格 |
| X1R | full X1 + 新校准安全 gate | 0.114370 | 24,177 | +0.5353% vs X0 | 校准/安全更好但正式精度失败 |

这张表的 NRMSE 均为 fused 口径，不与 Stage 5A 的 corrected-candidate 主指标混排。
G1、D0R、X1-fixed-G0 均不能标为最终最优，因为“数值最低”“具有正式选择资格”
和“通过全部守门”是三个不同概念。

Stage 5A candidate 应单列阅读：X0=0.116160、X1-F=0.115789、X1-M=0.115834、
X1-C=0.115680、full X1=0.115705；形式化选择 X1-F，数值最低为 X1-C。

### 15.2 当前最终模型相对关键基线

| 对比 | NRMSE 变化 | 参数变化 | 可用结论 |
| --- | ---: | ---: | --- |
| F7 vs B0 Persistence | -7.26% | +20,969 | 学习修正和工况融合显著优于纯 Persistence |
| F7 vs B1 Long PatchTST | -6.59% | -90.06% | 当前轻量结构优于独立 long-only |
| F7 vs B2 | -1.68% | +13.86% | 显式工况动态融合提供增量 |
| F7 vs B6 | -2.72% | -97.63% | 更轻且宏平均更优 |
| F7 vs F4 | -0.053% | -0.86% | 特征精简没有损失宏指标，但效应很小 |

---

## 16. 现行实验方案进度

| 工作包 | 代码 | 训练 | 预测/报告 | 最终结果 | 进度 |
| --- | --- | --- | --- | --- | --- |
| 原 FeTS-PatchTST 固定 seed | `wind_FeTS_PatchTST_train.py` | seed 已固定 2026 | 历史 Round 04 已归档 | 完整模型仅作历史参考 | 完成 |
| Stage 1 B0–B6 | `wind_FeTS_PatchTST_min_train.py` / `_predict.py` | 8×5，B0 解析式 | 完整 | B2 作为 Pareto 主干，B6 为形式化参考 | 完成 |
| Stage 2 R0–R6 | `wind_RegimeEncoder_PatchTST_train.py` / `_predict.py` | R2–R5 新训 | 完整 | R4 最优母结构 | 完成 |
| Stage 2 F0–F8 | `wind_RegimeEncoder_PatchTST_feature_screen_train.py` / `_predict.py` | F4 引用，其余按矩阵 | 完整 | F7=P+H+D | 完成 |
| FP0/FP4 | 同上 | 仅 gate 可训练 | 完整 | C 无明确贡献 | 完成 |
| 复杂度、drift、解释报告 | feature screen predict | 不新增主模型 | 完整 | 参数/文件/训练时间完成；统一硬件运行效率缺失 | 部分完成 |
| Stage 3 G0–G4 | `wind_controlled_gate_cali_train.py` / `_predict.py` | G1–G3 新训 | 完整 | 回退 G0 | 完成 |
| Stage 4 T0/M0/T1–T3 | `wind_time_freq_model_train.py` / `_predict.py` | M0–T3 新训 | 完整 | 回退 T0 | 完成 |
| Stage 4 可视化补齐 | `wind_time_freq_model_visualize.py` | 不重训 | 79 张补图/复核矩阵 | manifest=complete | 完成 |
| Stage 4B D0/D0R/D1–D3 | `wind_time_freq_model_stage4b_train.py` / `_predict.py` | D0 只读，D0R/D1–D3 共 20 个新模型 | 完整，67 张正式图 | 回退 D0 | 完成 |
| Stage 5A X0/X1-F/M/C/X1 | `wind_multiscale_correc_cand_train.py` / `_predict.py` | X0 只读，四个 adapter 共 20 个新模型 | 完整，96 张正式图 | candidate 形式化选 X1-F；部署不变 | 完成 |
| X1R 同 candidate 门控闭环 | `wind_multiscale_correc_cand_x1r_train.py` / `_predict.py` | 冻结 full X1，新训五站 gate | 完整，36 张正式图 | 回退 X0，X2/X3 未解锁 | 完成 |
| Part 3 Round 1 A0–A5 | `wind_part3_round1_stage_a_timealign_qdf_train.py` / `_predict.py` | A0 只读；A1–A5 五站新训 | 完整 | A1 数值改善但未过跨站守门；回退 A0，Stage B 不解锁 | 完成 |
| Part 3 Round 2 五站公平重训 | `wind_part3_round2_f7_g0_strong_baseline_train.py` / `wind_dl_model_predict.py` | F7/G0 五站从零训练 | 完整 | NRMSE 第 1，排除 warm-start/batch 混淆 | 完成 |
| Part 3 Round 3 外部 14 站预处理 | `wind_part3_round3_external14_preprocess.py` | 14 站原始 Excel 独立处理 | 审计与 bundle 完整 | 70/15/15、train-only scaler/q99.9、45 通道 | 完成 |
| Part 3 Round 3 强基线 | `wind_part3_round3_external14_all_models_train.py` / `_predict.py` | 196/196 可训练任务；Persistence 无需训练 | 210/210 预测、15 模型统一结果 | WindPRISM Macro/Micro 综合第 1 | 完成 |
| 多 seed/K-fold | 既有多 seed 结果待精确归档；Round 3 为 seed=2026 | 不重复 Round 3 多 seed | — | 用户确认同模型多 seed 结论一致；手稿引用前必须补 artifact 路径 | 证据待归档 |
| 严格时序/新盲测 | 外部 14 站已完成严格时序处理；全球最终盲测未完成 | 14 站 within-station holdout | 完整审计 | 已修复主要泄漏风险，但 marker 仍非 globally blind | 部分完成 |
| 静态 fine/mid/coarse 表示 | Stage 5A 代码 | 已完成 | 完整 | candidate 小幅有效，未成为部署结构 | 完成 |
| X2–X6 token 单/双向跨尺度 | — | 未训练 | — | X1R 守门失败，明确未解锁 | 停止 |

---

## 17. 接下来实验的现行决策方案

> **状态更新（2026-07-27）**：本节 17.2 中原先提出的第二条 Q1 方法路线已经按
> 第 23 节执行了 Part 3 Round 1 A0–A5；没有新变体通过守门。随后完成了五站
> 公平重训和外部 14 站 15 模型强基准。当前决策已经从“继续开发下一结构”
> 更新为“停止新增结构训练，进入统计整理和 SCI 手稿写作”。第 25–31 节为
> 当前权威状态，本节保留用于解释当时为什么设计该补充路线。

### 17.1 当前不应直接做的事

- 不恢复四专家 top-k 稀疏路由；当前两候选结构不存在同样的负载均衡问题；
- 不因为论文需要“第三个结构创新”直接训练完整时频跨尺度模型；
- 不把 G2/G3/G4、T1–T3、X1 或 X1R 的局部优点写成最终模型已实现的能力；
- 不启动 X2–X6；X1R 的预声明解锁条件已经明确失败；
- 不在没有新证据时继续在 legacy-seen 测试集反复手工调阈值。

### 17.2 下一步优先问题

原 Stage 4B→5A→X1R 路线已经把以下链路实际验证完毕：

~~~text
candidate 增益
    ↓
oracle / gate 可辨识性
    ↓
跨场站一致的 fused 增益
    ↓
dynamic / ramp 守门
~~~

结论是：静态多尺度可以小幅改善 candidate，但现有二分类 soft-oracle 和
calibrated-safe gate 会把模型推向大量稳定/低功率样本的 L1 收益，牺牲大变化
样本的平方误差与 ramp。继续微调 X1R 或直接增加 token 交互，成功先验较低。

当时有两条诚实路线：

1. **以现有 X0 定稿。** 补齐强基线、复杂度、统计和协议
   证据，突出轻量结构、显式工况和完整负结果；
2. **仍以 Q1 为目标，开启新的问题驱动方法线。** 不再围绕 X1R 调阈值，而是
   先在设计用训练/验证数据验证 Persistence 残差对齐和工况条件化多步损失；
   只有候选五站一致改善后才进入选择性 variable×patch 交互与收益幅度门控。
   详见第 23 节当时的 PARQ-Wind 预注册路线。

实际执行选择了第 2 条的 Stage A，但 A1–A5 未解锁更复杂的 Stage B；因此最终
又回到第 1 条，并通过 Round 2/3 把公平重训、现代强基线和外部 14 站证据补齐。

任何 corrected candidate 的新结构都必须遵守：

1. 训练集重建 soft oracle；
2. 重算逐 horizon `|C-P| Q90`；
3. 重新训练和校准同一门控；
4. 同时报 candidate 和 fused 指标；
5. 继续检查五场站、dynamic、ramp、regret、harm 和参数门槛。

### 17.3 启动更完整 residual 的必要条件

任何未来新 candidate 同时满足以下条件，才进入下一结构阶段：

- corrected 总体、dynamic、ramp 至少不退化，并有明确改善；
- 改善不能只集中在 1–2 个场站；
- 重新校准后，candidate 改善能转化为 fused NRMSE 改善；
- Persistence 安全、ramp 和参数守门继续通过。

当前 T3 和 X1 只满足“corrected 有小幅宏改善”，X1R 不满足跨场站、fused、
逐 horizon 和 dynamic/ramp 条件，因此不能继续扩展原 X 系列。

### 17.4 Stage 5B 原控制矩阵与当前状态

原控制矩阵及截至 2026-07-19 的实际状态如下：

| 对照 | 结构 | 要隔离的因果问题 | 当前状态 |
| --- | --- | --- | --- |
| X0 | 当前 D0/T0/F7 | 最终基线 | 完成，当前部署模型 |
| X1-F/M/C/X1 | 单尺度或 fine/mid/coarse 独立编码后静态融合 | 多尺度表示本身是否有效 | 完成；candidate 小幅有效 |
| X1R | full X1 candidate + candidate-specific 新 gate | 多尺度 candidate 收益能否闭环 | 完成；守门失败 |
| X2 | X1 + coarse→fine 单向 token 交互 | 长趋势是否帮助局部 ramp | 未解锁 |
| X3 | X1 + fine→coarse 单向 token 交互 | 局部变化是否修正全局趋势 | 未解锁 |
| X4 | X1 + fine↔mid 双向交互 | 相邻尺度交互增量 | 未解锁 |
| X5 | X1 + mid↔coarse 双向交互 | 中长尺度交互增量 | 未解锁 |
| X6 | fine↔mid↔coarse token 级双向交互 | 真正跨尺度结构的联合增量 | 未解锁 |

原协议要求 X1R 全部通过才只先解锁 X2/X3。实际 marker 已记录
`stage5b_x2_x3_unlocked=false`，所以 X2–X6 全部停止；不得以 X1 candidate
有效为由绕过 fused 闭环，也不得直接用 X6 替代父子对照。

### 17.5 原补证缺口与截至 2026-07-27 的完成状态

| 证据 | 2026-07-19 状态 | 当前状态 | 剩余边界 |
| --- | --- | --- | --- |
| 新的独立时间外推/rolling-origin | 未做 | 14 站完成 within-station chronological holdout | 不是 globally blind 或零样本 unseen-site |
| 严格因果处理和 train-only scaler | 未做 | 14 站已完成并有 hash/audit | 原 5 站历史协议不追溯重做 |
| 至少 3 seeds 与置信区间 | 未做 | 用户确认既有多 seed 结论一致；14 站 farm bootstrap 已做 | 多 seed artifact 路径仍需归档，Round 3 仅 seed=2026 |
| 新场站/外部数据 | 未做 | JSFD001–014 已完成 | 有另一淘汰工作流的历史数据暴露 |
| 同协议近期强基线 | 不完整 | 15 模型已完成 | 最近三基线差异未达 Holm 显著 |
| 统一硬件 latency/FLOPs/throughput/VRAM | 部分缺失 | 参数/文件/Pareto 完成 | 混合硬件 latency 仍不能作为公平结论 |
| 概率预测/不确定性 | 未开展 | 仍未开展，非当前必做 | 若将概率可靠性作为主卖点仍会被追问 |

因此这一节不再支持“外部数据、严格处理、现代强基线全部缺失”的旧结论。当前
真正优先的是无重训统计后处理、统一硬件推理复测和手稿边界控制。

---

## 18. 工程复现、冒烟测试和故障修复记录

### 18.1 固定随机种子

2026-07-11 起，FeTS-PatchTST 相关训练统一固定 `seed=2026`：

- 场站构模前重置 Python、NumPy、TensorFlow/Keras 随机状态；
- 请求 deterministic ops；
- Stage 1 的共同同名分支也从同一 seed 初始化；
- Stage 2–5/X1R artifact 均记录 seed。

这提高同环境复现性，但不保证跨 TensorFlow、CUDA、cuDNN 和硬件逐位一致。

### 18.2 Batch size 与冒烟测试

原生 `wind_dl_model_train.py` 默认 batch=256，完整 FeTS 模型曾经 OOM。正式
Stage 1 的 40 个 artifact 均记录 batch=192，Stage 2–5/X1R 直接以 192 为默认或
协议校验值。

Part 3 Round 2 为公平对齐旧基线，WindPRISM 使用 batch=256 并成功完成；Round 3
外部 14 站统一使用 batch=192，代码具备 HR-MoE OOM 时全站一致回退 128 的策略，
实际 196 个训练任务均未触发回退。

单场站、单 epoch 冒烟测试的作用是验证：

- 数据读取和滑窗；
- 模型构图与 GPU 训练步；
- loss 有限；
- 显存、checkpoint、目录和序列化链路。

它不能判断最终收敛、测试性能，也不能证明所有最大变体在 batch=256 下安全。
因此正式统一使用 192 是合理选择。代码中的 save/load smoke test 另用于验证
`.keras` 和自定义层重载一致性，两者不是同一个测试。

### 18.3 Feature archive 与 CSV 不一致报错

原报错：

~~~text
候选archive与对应预测CSV窗口/真值/fused不一致
~~~

根因不是窗口或真值错位，而是浮点表示链不同：

- TensorFlow 推理原值为 float32；
- 旧 CSV 写入 float32 的短十进制；
- NPZ 将同一值提升为 float64；
- Pandas 读回后按 float64 与固定绝对容差比较，少量点触发假阳性。

修复：

- 新推理让指标、CSV、NPZ 共用同一 float64 表示；
- `_csv_float_matches_archive()` 先做严格 `atol=1e-7`；
- 兼容旧文件时只允许 archive 确为 float32 精确提升，且 CSV 回转到 float32 后
  位级相同，不粗暴放宽物理单位容差；
- 将行数、sample/horizon 键、真值和 fused 校验分别报错。

修复后 FP0/FP4 五站 scaled candidate 位级一致，Frozen-Pair 验收全部通过。

### 18.4 G0 跨运行时重建容差报错

终端 NUMA 提示只是 WSL/内核缺少 NUMA 信息的 TensorFlow informational log；
日志随后已正常创建 RTX 3080 Ti Laptop GPU 并加载 cuDNN，不是异常根因。

真正报错：

~~~text
max_norm=6.52349e-05
mean_norm=3.71392e-08
~~~

Stage 2 F7 使用 Keras `predict` 图路径，原 G0 诊断逐 batch eager 调用。不同
CUDA kernel 在极少数长 horizon 点产生尾差，最大值略超旧 `5e-5`，均值却仅
`3.7e-8`。

修复：

- G0 改用 `diagnostic.predict(dataset)`；
- 继续严格检查时间键、真值和有限值；
- 跨运行时最大容量归一化容差调至 `1e-4`，均值仍为 `1e-6`；
- 归档 max、mean、p99.9 和超过旧阈值的点数。

修复后五站最大差约 `0.85e-7–1.11e-7`，超过旧 `5e-5` 的点数均为 0。

### 18.5 时频可视化补齐

原 `wind_time_freq_model_train.py` 只保存 candidate/gate history CSV、
checkpoint 和 hash marker，没有调用绘图函数。为避免修改已被 bundle hash
锁定的正式训练/预测代码，新增独立后处理脚本：

~~~text
wind_time_freq_model_visualize.py
~~~

补齐/复核矩阵共 79 张：

- T1–T3 candidate 的 loss/MAE/RMSE 三子图：15 张；
- M0–T3 gate 的 composite loss/forecast MAE/forecast RMSE 三子图：20 张；
- M0–T3 五站 gate-by-regime/horizon 和 reliability：40 张；
- 汇总 reliability、accuracy-safety、candidate horizon、regime：4 张。

M0 candidate 是冻结 F7，T0 本轮没有训练，脚本不会伪造不存在的训练曲线。

路径：

- `wind_results/time_freq_model/time_freq_model_visualization_backfill_manifest.json`
- `wind_results/time_freq_model/time_freq_model_visualization_backfill_inventory.csv`
- `wind_results/time_freq_model/<m0|t1|t2|t3>/history/`
- `wind_results/time_freq_model/<m0|t1|t2|t3>/testdata_predict_output/figures/`
- `wind_results/time_freq_model/testdata_predict_output/figures/`

### 18.6 Stage 5A 上游 complete marker 报错与代码变化量解释

首次运行 `wind_multiscale_correc_cand_train.py` 曾报：

~~~text
FileNotFoundError: 缺少上游complete marker:
.../stage4b_gate_closure_test_bundle_complete.json
~~~

根因是依赖校验过严：Stage 5A **训练**只消费 Stage 4B 的训练 bundle，用它重建
冻结 F7 candidate；正式重跑 Stage 4B 训练会有意使旧预测 marker 失效，而 Stage 5A
训练并不读取 Stage 4B 测试预测。把预测 complete marker 作为训练前置条件，会
错误阻塞一个数据依赖完整的训练流程。

修复后：

- Stage 5A 训练强制校验 Stage 4B training marker 及关键文件 hash；
- 若 prediction marker 存在，则继续校验它锁定当前 training marker；
- 若 prediction marker 暂缺，只打印提示并允许训练；
- Stage 5A **预测/测试选型**仍强制要求先运行
  `wind_time_freq_model_stage4b_predict.py`，发布与当前训练 marker 匹配的正式
  Stage 4B prediction bundle。

当时看到训练文件“新增约 77 行、删除 2 行”、预测文件“新增约 55 行、删除
2 行”，是相对已经生成的 Stage 5A 初版代码的修复差异，不是整个新模型只有这
些代码。Stage 5A 复用了已有 F7 candidate、数据滑窗、工况 context、G0 回放、
归一化、指标、绘图和 bundle 审计，只新增轻量 scale adapter 和实验编排；这种
受控复用正是为了隔离“多尺度表示”变量。模型新颖性应由实际计算图、参数和
父子消融判断，不能由单次 Git diff 行数判断。

---

## 19. 各阶段正式数据路径总索引

| 阶段 | 首选分析文件 | 最终选择 | 完成标记/主目录 |
| --- | --- | --- | --- |
| 历史 Round 04 | `wind_results/fets_patchtst/archive/round_04_fets_patchtst_horizon_regime_moe_v5ab_20260710/ARCHIVE_INFO.md` | 历史 HR-MoE | `wind_results/fets_patchtst/archive/round_04_fets_patchtst_horizon_regime_moe_v5ab_20260710/` |
| Stage 1 | `wind_results/fets_patchtst_min/testdata_predict_output/fets_patchtst_min_variant_comparison.csv` | B6 为原规则选择；B2 为后续 Pareto 主干 | `wind_results/fets_patchtst_min/` |
| Stage 2 R | `wind_results/regime_encoder_patchtst/testdata_predict_output/regime_encoder_patchtst_test_comparison_descriptive.csv` | R4 | `wind_results/regime_encoder_patchtst/` |
| Stage 2 F | `wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f0_f8_test_selection_output/feature_screening_f0_f8_test_variant_comparison.csv` | F7 | `wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f0_f8_probe_analysis_output/feature_screening_f0_f8_fp_bundle_complete.json` |
| FP | `wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/frozen_pair_probe_test_output/feature_screening_frozen_pair_control_report.md` | 删除 C | `wind_results/regime_encoder_patchtst/stage2_feature_screening_f0_f7/f0_f8_probe_analysis_output/feature_screening_f0_f8_fp_bundle_complete.json` |
| Stage 3 G | `wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_final_selection.md` | G0 | `wind_results/controlled_gate_cali/testdata_predict_output/controlled_gate_cali_test_bundle_complete.json` |
| Stage 4 T | `wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_final_selection.md` | T0 | `wind_results/time_freq_model/testdata_predict_output/time_freq_model_test_bundle_complete.json` |
| Stage 4 可视化 | `wind_results/time_freq_model/time_freq_model_visualization_backfill_inventory.csv` | 79 张补图完成 | `wind_results/time_freq_model/time_freq_model_visualization_backfill_manifest.json` |
| Stage 4B D | `wind_results/time_freq_model/supplement_round2_stage4b_gate_closure/testdata_predict_output/stage4b_gate_closure_test_final_selection.md` | D0；D0R 仅数值小幅最低 | `wind_results/time_freq_model/supplement_round2_stage4b_gate_closure/testdata_predict_output/stage4b_gate_closure_test_bundle_complete.json` |
| Stage 5A X | `wind_results/multiscale_correc_cand/testdata_predict_output/multiscale_correc_cand_test_final_selection.md` | candidate 形式化选 X1-F；X1-C 数值最低 | `wind_results/multiscale_correc_cand/testdata_predict_output/multiscale_correc_cand_test_bundle_complete.json` |
| X1R 闭环 | `wind_results/multiscale_correc_cand/x1r_gate_closure/testdata_predict_output/x1r_gate_closure_test_final_selection.md` | 回退 X0，X2–X6 未解锁 | `wind_results/multiscale_correc_cand/x1r_gate_closure/testdata_predict_output/x1r_gate_closure_test_bundle_complete.json` |
| Part 3 Round 1 | `wind_results/part3_new_module_supplement/01_stage_a_timealign_residual_alignment_qdf/testdata_predict_output/stage_a_test_final_selection.md` | A0/WindPRISM；A1–A5 不晋级 | `wind_results/part3_new_module_supplement/01_stage_a_timealign_residual_alignment_qdf/stage_a_training_bundle_complete.json` 与测试完成 marker |
| Part 3 Round 2 | `wind_results/part3_new_module_supplement/02_strong_baseline_f7_g0_fair_training/testdata_predict_output/part3_round2_all_models_test_macro_comparison.csv` | 公平从零训练 F7/G0 为五站 NRMSE 第 1 | `wind_results/part3_new_module_supplement/02_strong_baseline_f7_g0_fair_training/part3_round2_strong_baseline_training_bundle_complete.json` |
| Part 3 Round 3 预处理 | `wind_results/part3_new_module_supplement/03_external14_leakage_free_strong_baseline_benchmark/data_audit/round3_external14_data_audit.csv` | 14 站预处理全部完成 | `wind_results/part3_new_module_supplement/03_external14_leakage_free_strong_baseline_benchmark/round3_preprocess_bundle_complete.json` |
| Part 3 Round 3 训练/预测 | `wind_results/part3_new_module_supplement/03_external14_leakage_free_strong_baseline_benchmark/testdata_predict_output/round3_external14_test_final_selection.md` | WindPRISM 在 15 模型中综合第 1 | `round3_training_bundle_complete.json` 与 `testdata_predict_output/round3_external14_prediction_bundle_complete.json` |

读取测试性能时应优先看各阶段 `variant_comparison.csv` 或
`final_selection.md`，不要只看单个场站预测 CSV，也不要用验证集文件替代用户
要求的测试集选型结论。Stage 5A 的正式主指标是 corrected candidate；Stage 4B
和 X1R 的正式主指标是 fused。两种口径不能直接混排。

---

## 20. 论文写作建议与结论边界

### 20.1 建议的论文贡献顺序

1. **Persistence-centered lightweight corrective forecaster**
   用 Persistence 作为物理低方差锚点，以 18,416 参数的轻量因果 residual
   建立 corrected candidate，取代 885,395 参数的四专家堆叠。

2. **Explicit wind-regime-conditioned sample–horizon fusion**
   用历史功率、轮毂风速和风向变化的 36 维显式工况编码驱动两候选动态融合，
   并通过 R/F 系列直接消融证明特征选择。

3. **Candidate-controlled interpretation and safety evaluation**
   用 Frozen-Pair 控制 candidate drift，并用 oracle、Brier/ECE、regret、
   harm、ramp、低功率和 Persistence abstention 系统分析门控可靠性；Stage 4B
   与 X1R 进一步用同 candidate 闭环分离候选和 gate 效应。

第三点更适合表述为“评价与机制验证贡献”，而不是最终部署结构已经通过的安全
模块。

### 20.2 可以写入正文的结果

- B2 相对 B6 的精度—复杂度优势；
- R4 相对 R2/R3 的显式工况增益和样本门控变化；
- F0–F8 与 FP0/FP4 对 P/H/M/D/C 的直接归因；
- F7 相对 B0/B1/B2/B6 的总体、逐场站和复杂度比较；
- G0–G4 的校准—安全—精度 Pareto 及 ramp 失败原因；
- T0–T3 的 candidate/fused 分离结果，作为为何不继续堆叠时频模块的否定消融；
- D0–D3 的 candidate-specific oracle/Q90 闭环以及新 gate 未转化 T1 收益；
- X1-F/M/C/X1 的静态多尺度 candidate 消融，明确 X1-C 数值最低、X1-F 是
  预声明 tie-band 下的形式化选择；
- X1R 在相同 full-X1 candidate 下校准/安全显著改善、但 NRMSE 与 dynamic/ramp
  失败的 L1—L2—尾部权衡。
- Part 3 A0–A5 中 QDF、local/global residual alignment 的否定结果及停止
  variable×patch Stage B 的守门决策；
- Part 3 Round 2 的从零公平重训，证明 WindPRISM 的 NRMSE/复杂度优势并非仅由
  warm-start 或 batch 差异造成；
- 外部 14 站统一 15 模型比较、逐站、逐 horizon、W/T/L、Holm 显著性、
  参数 Pareto 和数据质量敏感性；
- WindPRISM 在外部 14 站 Macro/Micro 综合第 1、H3–H16 NRMSE 第 1，同时如实
  报告 H1–H2 由 DLinear 最优、训练受限子集由 DLinear 更优。

### 20.3 不能写成已证实的主张

- “四专家 FeTS-PatchTST 中每个专家都必要”；
- “最终模型使用防塌缩动态稀疏路由”；
- “C 物理一致性特征有效”；
- “辅助工况任务提升了预测”；
- “最终模型具有通过精度门槛的校准安全保护”；
- “因果时频增强或静态/Token 级跨尺度交互提升了最终部署预测”；
- “最终模型采用 X1-F、full X1 或 X1R”；
- 在未给出确切 artifact 前写“WindPRISM 已完成充分多 seed 稳定性验证”；
- 把外部 14 站写成“完全未见场站零样本泛化”“全球最终独立盲测”；
- “所有 14 站均最优”或“所有 horizon、所有指标都优于全部基线”；
- “统计显著优于 DLinear、PatchTST 和 HR-MoE”；
- “WindPRISM 是参数量最小或推理速度最快的模型”；
- 把混合 4090/3080 Ti 环境的现有 `inference_seconds` 当作同硬件延迟结论。

### 20.4 对 SCI 一区创新充分性的最终评估

与最初复杂 HR-MoE 相比，当前论文故事更清晰、可归因且更符合近期趋势：

- 有领域先验；
- 有轻量化；
- 有显式工况；
- 有直接特征组消融；
- 有 candidate drift 控制；
- 有校准和安全诊断；
- 有失败结构的否定证据。

> 本小节原先截至 2026-07-19 的“专业型 Q2”判断，已被 Part 3 Round 1–3 的
> 实际结果更新。保留早期判断的原因是记录补实验为什么发生；当前投稿判断以
> 第 30 节为准。

Stage 4B、Stage 5A 和 X1R 增加了强归因与失败分析，但没有改变最终结构；
Part 3 Round 1 又否定了残差对齐/QDF 的继续扩展。真正改变证据强度的是：

1. Round 2 在统一训练预算下从零公平重训 WindPRISM；
2. Round 3 对 14 个补充场站从原始 Excel 重建严格时序数据；
3. 统一比较 15 个模型，包括 iTransformer、TimesNet、TimeMixer、DLinear 和
   Persistence；
4. 补齐逐站、逐 horizon、W/T/L、Wilcoxon-Holm、bootstrap CI 和复杂度 Pareto。

因此当前不再建议以“只有 Q2 证据”概括项目。更准确的评估是：
**方法创新具有领域针对性，实验链条可支撑向专业型 Q1 投稿，但最接近三个基线
差异不显著、14 站并非全球最终盲测、归一化依赖训练段 q99.9，故仍是 Q1 边缘，
不是高把握 Q1。** 当前应停止为增加结构名称而继续训练，优先完成统计后处理、
同硬件推理复测和手稿边界控制。

---

## 21. 代码入口

| 工作 | 训练 | 预测/分析 |
| --- | --- | --- |
| 原 HR-MoE | `wind_FeTS_PatchTST_train.py` | 由既有 FeTS 预测流程完成 |
| Stage 1 | `wind_FeTS_PatchTST_min_train.py` | `wind_FeTS_PatchTST_min_predict.py` |
| Stage 2 R | `wind_RegimeEncoder_PatchTST_train.py` | `wind_RegimeEncoder_PatchTST_predict.py` |
| Stage 2 F/FP | `wind_RegimeEncoder_PatchTST_feature_screen_train.py` | `wind_RegimeEncoder_PatchTST_feature_screen_predict.py` |
| Stage 3 | `wind_controlled_gate_cali_train.py` | `wind_controlled_gate_cali_predict.py` |
| Stage 4 | `wind_time_freq_model_train.py` | `wind_time_freq_model_predict.py` |
| Stage 4 补图 | — | `wind_time_freq_model_visualize.py` |
| Stage 4B 门控闭环 | `wind_time_freq_model_stage4b_train.py` | `wind_time_freq_model_stage4b_predict.py` |
| Stage 5A 多尺度 candidate | `wind_multiscale_correc_cand_train.py` | `wind_multiscale_correc_cand_predict.py` |
| X1R 门控闭环 | `wind_multiscale_correc_cand_x1r_train.py` | `wind_multiscale_correc_cand_x1r_predict.py` |
| Part 3 Round 1 TimeAlign/QDF | `wind_part3_round1_stage_a_timealign_qdf_train.py` | `wind_part3_round1_stage_a_timealign_qdf_predict.py` |
| Part 3 Round 2 F7/G0 公平重训 | `wind_part3_round2_f7_g0_strong_baseline_train.py` | `wind_dl_model_predict.py` 中的统一 Round 2 输出 |
| Part 3 Round 3 外部 14 站预处理 | `wind_part3_round3_external14_preprocess.py` | 同文件生成审计和固定 bundle |
| Part 3 Round 3 十五模型统一基准 | `wind_part3_round3_external14_all_models_train.py` | `wind_part3_round3_external14_all_models_predict.py` |

---

## 22. 现有方案的投稿定位与建议

### 22.1 当前最现实的定位

截至 2026-07-19，本节曾把论文定位为专业型 Q2；截至 2026-07-27，外部 14 站
严格时序处理、15 模型统一强基线和跨站统计已经完成，当前可按
**专业型 SCI/JCR Q1 冲击稿**组织。这里的“可冲击”不等于高录用概率：对
DLinear、PatchTST、HR-MoE 的优势尚未达到 Holm 校正显著，外部 14 站不是全球
最终盲测，且部署时延还没有在统一硬件复测。若目标期刊审稿标准更强调通用
模型结构的新颖性而不是领域机制与系统证据，仍存在被评价为 Q2 增量工作的风险。

期刊分区会随年份、学科类别和机构口径变化；以下排序是研究主题与当前证据的
**相对匹配度**，不是录用概率承诺，投稿当年必须重新核验 JCR/中科院分区、
开放获取费用和最新 scope。

### 22.2 三本相对匹配的投稿期刊

| 顺序 | 期刊 | 与现方案的匹配点 | 投稿前应重点补强 |
| --- | --- | --- | --- |
| 1 | [IET Renewable Power Generation](https://ietresearch.onlinelibrary.wiley.com/journal/17521424) | 风电功率预测、电力系统运行、轻量部署和真实场站数据直接匹配 | 强化 ramp、可靠性、工程价值、同协议强基线和统计检验 |
| 2 | [Wind Energy](https://onlinelibrary.wiley.com/journal/10991824) | 风电主题最集中，P/H/D、Persistence 与容量归一化容易形成领域叙事 | 避免写成通用网络拼装，强化风电物理解释和场站泛化 |
| 3 | [Energy Reports](https://www.sciencedirect.com/journal/energy-reports) | 能源/电力应用范围较宽，能容纳轻量预测与系统实验 | 仍有 desk-reject 风险；需压缩庞杂实验并突出清晰主贡献 |

### 22.3 按现有 X0 定稿的论文结构

1. Introduction：超短期风电非平稳、Persistence 强基线及复杂模型部署问题；
2. Related Work：Patch/Transformer、物理先验、工况门控、可靠性评价；
3. Problem and Protocol：96→16、五站开发证据与 14 站补充验证、容量归一化
   与 train-reference 归一化、两层协议边界；
4. Method：轻量 causal residual、P+H+D 显式编码、sample×horizon 融合；
5. Controlled Development：B/R/F/FP 直接父子消融与 candidate drift；
6. Main Results：14 站 15 模型总体、逐场站、逐 horizon、显著性和复杂度；
7. Reliability and Negative Ablations：G/T/D/X、校准、安全、ramp 和闭环失败；
8. Limitations：测试选型、非 globally blind、q99.9 参考、受限训练子集和
   混合硬件效率；
9. Conclusion。

---

## 23. 历史补充方案：PARQ-Wind 预注册路线及其实际结局

> 本节是 2026-07-19 制定的方案原貌，不再代表待执行任务。实际只执行了
> Stage A 的 A0–A5，结果见第 26 节：A1 虽取得最低 candidate Macro NRMSE，
> 但跨站一致性和 Stage B 守门失败；因此 Stage B 的 variable×patch 与 Stage C
> 没有启动，最终模型仍为 WindPRISM。另一个协议偏差是，原计划要求只看训练/
> 验证集，实际按当时用户要求使用了已见测试集作描述性选型，不能称为确认性
> 预注册结果。

### 23.1 为什么不再继续微调 X1R 或直接训练 X2–X6

X1R 已经证明当前瓶颈不是“gate 不够校准”：它把 Brier、ECE、regret、harm 和
饱和率大幅改善，却同时让 16 个 horizon 的 NRMSE 全部退化，并伤害 dynamic、
ramp 和大变化样本。继续调 `calibration/safety` 权重很可能只在 L1、小误差与
L2、尾部之间移动 Pareto 点。静态 X1 candidate 的收益又只有约 0.4%，不足以
支持直接增加完整 token 交互。

因此下一代路线必须先解决两个更基础的问题：

1. corrected residual 是否学到了 Persistence **失效时未来 16 步残差的形态**；
2. 训练目标是否显式处理 NRMSE/NMAE、跨 horizon 误差相关和 ramp 尾部。

下面的 `PARQ-Wind` 是当时将 2026 顶会时序思想改造成风电任务的研究假设。
其中 residual alignment 与 regime-QDF 已通过 A1–A5 实现并得到否定性守门结论；
选择性 variable×patch 和收益幅度门控仍未实现，不能写入当前论文已完成贡献。

### 23.2 拟议方法：PARQ-Wind

拟议全名：**Persistence-Anchored Residual Alignment and Regime-conditioned
Quadratic Routing for Wind Power Forecasting**。保持纯数值时序、非多模态。

#### 23.2.1 Persistence 锚定的未来残差对齐

定义训练期真值残差 `r*=Y-P` 与候选残差 `r_hat=C-P`，由只在训练期可见未来
真值的 stop-gradient teacher 提供分布关系，推理时完全删除 teacher：

- local alignment：把未来 16 步分为四个 1 h patch，强调局部 ramp 形态；
- global alignment：对齐完整 4 h 残差轨迹及跨 patch 关系；
- 由 P+H+D 工况权重决定局部/全局对齐强度。

思想来源于 ICLR 2026
[TimeAlign: Bridging Past and Future—Distribution-Aware Alignment for Time Series Forecasting](https://iclr.cc/virtual/2026/poster/10007329)。
风电化改造不是直接对齐原始序列，而是对齐 **Persistence 何时失效、失效幅度
和 ramp 轨迹**，因此仍保留工程物理锚点。

#### 23.2.2 工况条件化多步二次目标

借鉴 ICLR 2026
[Quadratic Direct Forecast](https://iclr.cc/virtual/2026/poster/10006776)，用
P+H+D 软工况生成半正定 16×16 horizon 矩阵：

~~~text
W(z) = Σ_k π_k(z) L_k L_kᵀ + εI
L = eᵀW(z)e + λ1 SmoothL1(e) + λr Huber(Δŷ-Δy)
~~~

- 非均匀对角项学习不同 horizon 难度；
- 非对角项学习 16 步误差相关与连续轨迹；
- stable/dynamic/ramp 三类基矩阵由软工况混合；
- SmoothL1 保持 NMAE，差分项直接约束 ramp，二次项对应 NRMSE/尾部。

这比 X1R 的“corrected 是否胜过 Persistence”二分类 oracle 更直接地优化多指标
目标错配。

#### 23.2.3 风电物理分组的 variable×patch 选择性交互

只有残差对齐和 QDF 已经形成五场站一致 candidate 收益时才启动。将 45 个变量
按功率、轮毂风速、多高度风速、风向和其它气象分组，构造 variable×patch 二维
token 场：

- 局部 depthwise 2D 交互捕获同一时间邻域的物理变量组合；
- 图谱低/中/高频分解做 patch 级选择，而不是三个尺度池化后静态拼接；
- 只预测 `C-P` 的增量，不破坏 Persistence 锚点；
- 每个方向/频带设置直接父子消融和路由可视化。

思想来源于 ICLR 2026
[xCPD: Routing Channel-Patch Dependencies with Graph Spectral Decomposition](https://iclr.cc/virtual/2026/poster/10006906)
和 [VPNet](https://openreview.net/forum?id=CNVL194fO5)。相对 Stage 5A 的核心变化
是输入依赖的 patch/变量选择与真实 token 交互，而非静态 concat。

#### 23.2.4 收益幅度门控与 Persistence 保护

将二分类 oracle 改为多目标反事实收益：

~~~text
u = α(e_P²-e_C²) + β(|e_P|-|e_C|) + γ(R_P-R_C)
target_gate = sigmoid(u / τ)
~~~

其中 `R` 是 ramp/一阶差分损失。gate 学习“corrected 能改善多少”，而不是只学
“corrected 是否略好”；再用收益不确定性和逐 horizon Q90 决定何时回退
Persistence。目标是避免 X1R 在大量稳定样本上获得漂亮校准，却对 dynamic/ramp
过度保守。

### 23.3 Stage A：优先验证残差对齐与多步目标

| 编号 | 结构 | 研究问题 |
| --- | --- | --- |
| A0 | X0 | 当前父基线 |
| A1 | X0 + regime-QDF | 多步相关目标是否独立有效 |
| A2 | X0 + local residual alignment | 是否改善局部 ramp 形态 |
| A3 | X0 + global residual alignment | 是否改善 4 h 整体轨迹 |
| A4 | X0 + local + global alignment | 两层对齐是否互补 |
| A5 | A4 + regime-QDF | 推荐的第一版 Q1 candidate |

原计划规定 Stage A 只使用设计用训练/验证数据选择，不再查看五个 legacy-seen
test 调参；实际执行改为按 legacy test 作描述性选型，已固定写入 marker。A5
进入 Stage B 前原建议同时满足：

- 5/5 场站验证 NRMSE 严格改善；
- 5/5 场站 NMAE 不退化，理想状态均严格改善；
- Macro NRMSE、NMAE 均至少改善 0.5%；
- dynamic、ramp-up、ramp-down、`change_ge_20` 均不退化；
- 无连续 horizon 系统退化；参数尽量保持约 30k。

### 23.4 Stage B：条件启动选择性 variable×patch 交互

| 编号 | 结构 | 作用 |
| --- | --- | --- |
| B0 | A5 | 新 candidate 父基线 |
| B1 | A5 + 物理分组局部 variable×patch | 隔离局部跨变量关系 |
| B2 | A5 + 图谱频带路由 | 隔离低/中/高频选择 |
| B3 | A5 + 二者 | 检验局部与图谱选择互补 |

若 B1/B2 的 corrected candidate 不能形成五场站一致改善，则停止，不扩展 B3
或更大的双向结构。参数预算、初始化和训练步必须受控。

### 23.5 Stage C：candidate→fused 收益闭环

| 编号 | 结构 | 研究问题 |
| --- | --- | --- |
| C0 | 最优 candidate + 旧 G0 | 旧 gate 的直接转化能力 |
| C1 | 收益幅度门控 | 多指标收益是否优于二分类 oracle |
| C2 | C1 + calibration | 校准是否在不损精度下增益 |
| C3 | C2 + Persistence safety | 高风险样本能否安全回退 |

每次 candidate 改变都必须重新生成 train-only oracle/utility 和逐 horizon Q90、
重新训练 gate、验证 candidate identity。建议要求 fused 至少转化 candidate NRMSE
收益的 50%，五场站 NRMSE/NMAE 不退化，并通过 dynamic/ramp、regret/harm、
Brier/ECE 和参数守门。

### 23.6 原计划的一期确认性证据与完成状态

原五个测试集已是 `legacy_seen_test_selected`，不能再承担确认性评价。此后
已经完成的事项包括：14 个补充站的严格时序预处理、PatchTST/iTransformer/
TimesNet/TimeMixer/DLinear 等 15 模型统一比较、逐站/逐 horizon 结果、
Wilcoxon-Holm 与参数复杂度。尚未完成或只能部分声称的事项包括：

- 外部 14 站是 within-station holdout 且存在历史数据暴露，不是全球最终盲测；
- Round 3 只有 seed=2026；用户确认的既有同模型多 seed 证据需要精确归档；
- 当前只有 NRMSE 的配对显著性，NMAE、Friedman/CD 和效应量表仍可后处理；
- 参数量完整，但 FLOPs、统一硬件延迟、吞吐和峰值显存尚未形成公平主表；
- A1–A5 的训练期 teacher 已做推理输入隔离，但因未晋级，不属于最终模型；
- 原五站 dynamic/ramp 机制分析完整，外部 14 站不再追加 post-hoc P/H/D 消融，
  以免继续消耗其补充验证价值。

### 23.7 若新路线成功，一区论文大致结构

1. Introduction：Persistence 工况失效、历史—未来残差分布差和多步误差相关；
2. Related Work：物理/数据融合、工况门控、时频/patch、分布对齐、多步目标；
3. Problem Formulation：96→16、Persistence residual、工况与无未来信息推理约束；
4. PARQ-Wind：residual alignment、regime-QDF、选择性交互、收益安全门控；
5. Experimental Protocol：新 holdout/公开数据、强基线、多 seed、统计与复杂度；
6. Main Results：总体、五场站、逐 horizon、dynamic/ramp、外部泛化；
7. Ablation and Mechanism：A/B/C 矩阵、QDF、频带路由、candidate→fused；
8. Reliability and Deployment：Brier/ECE、regret/harm、过估风险和资源；
9. Limitations and Conclusion。

---

## 24. 参考资料

### 24.1 项目结构来源

- [PatchTST: A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730)
- [PatchTST official repository](https://github.com/yuqinie98/PatchTST)
- [FeTS: A Feature-Aware Framework for Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/39838)
- [FeTS repository](https://github.com/lllucky111/FeTS)
- [M2FMoE](https://arxiv.org/abs/2601.08631)
- [LayerScale/CaiT](https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.pdf)
- [iTransformer official repository](https://github.com/thuml/iTransformer)
- [THUML Time-Series-Library（TimesNet 等）](https://github.com/thuml/Time-Series-Library)
- [TimeMixer official repository](https://github.com/kwuking/TimeMixer)
- [DLinear reference repository](https://github.com/honeywell21/DLinear)
- [TimeAlign reference repository](https://github.com/TROUBADOUR000/TimeAlign)

FeTS、M2FMoE 和 LayerScale 只说明设计思想来源；是否对本项目有效必须由本项目
直接消融决定。当前最终 X0/D0/T0/G0/F7 已不包含完整 FeTS、PatchTST encoder
和四专家 M2FMoE 结构。

### 24.2 2025–2026 风电预测代表论文

- [Physics-constrained wind power forecasting aligned with probability distributions for noise-resilient deep learning](https://doi.org/10.1016/j.apenergy.2025.125295)
- [Non-stationary GNNCrossformer](https://doi.org/10.1016/j.apenergy.2024.124492)
- [Developing an interpretable wind power forecasting system using a transformer network and transfer learning](https://doi.org/10.1016/j.enconman.2024.119155)
- [A novel frequency sparse downsampling interaction transformer for wind power forecasting](https://doi.org/10.1016/j.energy.2025.136199)
- [Fine-grained ultra-short-term wind power forecasting based on TFT integrated with turbine power time-series clustering](https://doi.org/10.1016/j.energy.2025.137995)
- [Network integrating multiscale analysis and nonlinear representation for short-term wind power forecasting](https://doi.org/10.1016/j.renene.2026.125849)
- [A time-frequency adaptive transformer for long-term wind power forecasting under complex meteorological fluctuations](https://doi.org/10.1016/j.eswa.2026.131740)
- [STWFormer](https://doi.org/10.1016/j.epsr.2026.113061)
- [Virtual prediction and wavelet packet transform for short-term wind power forecasting](https://doi.org/10.1016/j.epsr.2025.112640)
- [A physics-aware dynamic graph and mixture-of-experts framework for wind power forecasting](https://www.sciencedirect.com/science/article/pii/S0142061526002115)
- [A novel hybrid short-term and ultra-short-term wind power forecasting method based on Weather Research and Forecasting: WRF-iTransformer-PSO](https://doi.org/10.1016/j.energy.2026.140955)
- [Ultra-short-term wind power prediction for enhanced reliability considering error distribution characteristics and guided correction](https://doi.org/10.1016/j.energy.2026.141397)
- [A cross-dataset benchmark for neural network-based wind power forecasting](https://doi.org/10.1016/j.renene.2025.123463)

### 24.3 2026 时序预测方法启发

- [TimeAlign: Bridging Past and Future—Distribution-Aware Alignment for Time Series Forecasting](https://iclr.cc/virtual/2026/poster/10007329)
- [Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models](https://iclr.cc/virtual/2026/poster/10006776)
- [xCPD: Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](https://iclr.cc/virtual/2026/poster/10006906)
- [VPNet](https://openreview.net/forum?id=CNVL194fO5)
- [TimeRecipe](https://iclr.cc/virtual/2026/poster/10010822)
- [ProtoTS](https://iclr.cc/virtual/2026/poster/10010284)

上述顶会思想最初用于提出补充研究假设。TimeAlign/QDF 的风电化 Stage A 已实际
执行但未通过晋级守门；其余未执行结构仍只能视为假设。任何风电化改造必须由
直接父子消融和确认性评价证明，不能因为引用了顶会模块就自动构成一区创新。

---

## 25. Part 3 补充实验总览与证据分层

### 25.1 为什么启动第三部分补充实验

截至 X1R，原 5 站已经形成了完整的结构消融链，但一区投稿仍有三类关键质疑：

1. WindPRISM 的优势是否只来自旧 B2 权重、较小 batch 或训练预算不一致；
2. 2026 年常见现代强基线是否缺失，导致“只战胜较旧模型”；
3. 原 5 站测试集参与过多轮选型，是否缺乏新的跨数据证据。

因此第三部分没有继续无条件扩大网络，而是按以下顺序补证：

| 轮次 | 目的 | 核心问题 | 结果 |
| --- | --- | --- | --- |
| Part 3 Round 1 | 验证高成功先验模块 | TimeAlign 风格残差对齐与 regime-QDF 能否稳定增强 WindPRISM | A1 数值改善但跨站不稳；A1–A5 全部不晋级 |
| Part 3 Round 2 | 公平训练控制 | WindPRISM 在与原 PatchTST/旧基线对齐训练预算后是否仍优 | 五站 NRMSE 第 1，排除主要训练混淆 |
| Part 3 Round 3 | 外部 14 站强基准 | 严格时序处理、15 模型统一比较下是否仍综合最优 | WindPRISM Macro/Micro 综合第 1 |

统一结果根目录：

~~~text
wind_results/part3_new_module_supplement/
├─ 01_stage_a_timealign_residual_alignment_qdf/
├─ 02_strong_baseline_f7_g0_fair_training/
└─ 03_external14_leakage_free_strong_baseline_benchmark/
~~~

### 25.2 两层数据证据不能混排

| 维度 | 原 5 个开发场站 | 补充 14 个 JSFD 场站 |
| --- | --- | --- |
| 作用 | 结构搜索、直接消融、工况/门控机制分析 | 严格时序预处理后的强基线补充验证 |
| 测试协议 | `legacy_seen_test_selected` | 冻结模型统一评价，但保守标记非 globally blind |
| 归一化 | 已有容量口径 NRMSE/NMAE | 训练段功率 q99.9，必须称 trNRMSE/trNMAE |
| 数据处理 | 历史工程管线，存在整段 scaler/双向插值边界 | 原始 Excel、70/15/15、train-only 全部统计 |
| 主要价值 | 解释“为什么是 F7/G0” | 解释“面对 15 模型和更多场站是否仍有效” |
| 不可主张 | 独立盲测 | 完全未见场站零样本泛化、全球最终盲测 |

原 5 站的 0.113761 和外部 14 站的 0.128801 分母、场站与处理协议不同，
**绝不能直接比较数值大小，也不能据此说外部数据导致模型退化**。

### 25.3 最终名称与结构是否改变

Part 3 没有改变最终模型计算图。论文模型统一命名为 **WindPRISM**，代码和历史
实验中的别名仍为：

~~~text
WindPRISM
= F7 feature-screen winner
= G0 controlled-gate reference
= T0 time-frequency reference
= D0 gate-closure reference
= X0 multiscale reference
= A0 Part-3 Stage-A reference
~~~

当前名称是对最终方法的论文级标签；历史工程标题“FeTS-PatchTST”继续用于追踪
演进，但 WindPRISM 本身没有 PatchTST encoder、FeTS block 或四专家 MoE。

---

## 26. Part 3 Round 1：TimeAlign 残差对齐与 regime-QDF

### 26.1 研究目的与代码

这一轮对应第 23 节 Stage A，目标是在不直接训练完整 variable×patch 结构前，
先验证两个高成功先验：

- 将 TimeAlign 的历史—未来分布对齐思想改为
  **Persistence 未来残差的局部/全局对齐**；
- 用 P+H+D 工况生成多步相关的 **regime-conditioned quadratic direct
  forecast（QDF）** 训练目标。

代码：

~~~text
wind_part3_round1_stage_a_timealign_qdf_train.py
wind_part3_round1_stage_a_timealign_qdf_predict.py
~~~

结果根目录：

~~~text
wind_results/part3_new_module_supplement/
└─ 01_stage_a_timealign_residual_alignment_qdf/
~~~

### 26.2 A0–A5 实验矩阵

| 编号 | 结构 | 隔离的问题 |
| --- | --- | --- |
| A0 | X0/D0/T0/G0/F7 只读引用 | 当前 WindPRISM 父基线 |
| A1 | A0 + regime-conditioned QDF | 跨 horizon 二次误差相关目标是否独立有效 |
| A2 | A0 + local residual alignment | 四个 1 h patch 的局部残差形态对齐是否有效 |
| A3 | A0 + global residual alignment | 完整 4 h 残差轨迹关系对齐是否有效 |
| A4 | A0 + local + global alignment | 两层残差对齐是否互补 |
| A5 | A4 + regime-QDF | 对齐与多步目标是否形成联合收益 |

实现边界：

- A0 只读引用，不训练、不复制，也不重新 forward；
- A1–A5 从同一个 F7 父快照开始，只微调 B2 residual 的 4 个加权层；
- Persistence、P+H+D 工况编码器和旧 G0 门控保持冻结；
- teacher 只在训练期看到 `Y - Persistence`，推理 `.keras` 仍只有历史输入；
- projector 与 QDF 只属于训练 wrapper，不增加正式推理参数。

### 26.3 训练协议与复杂度

| 项目 | 设置 |
| --- | --- |
| seed | 2026 |
| batch | 192 |
| 最大 epochs | 60 |
| validation | 训练文件尾部 15% |
| optimizer | Adam，lr=1e-4 |
| early stopping | patience=8 |
| teacher warm-up | 3 epochs |
| 正式推理参数 | 所有变体均为 20,969 |

训练 wrapper 的额外参数只服务于训练：

- A1：21,761，较推理模型多 792；
- A2–A4：38,437，较推理模型多 17,468；
- A5：39,205，较推理模型多 18,236。

训练宏验证 candidate：

| 变体 | Val candidate NRMSE | Val candidate NMAE | 观察 |
| --- | ---: | ---: | --- |
| A1 | **0.107788** | 0.068033 | 验证 NRMSE 最低 |
| A2 | 0.110571 | 0.068176 | local alignment 无总体优势 |
| A3 | 0.108077 | 0.069753 | NRMSE 次优但 NMAE 较差 |
| A4 | 0.109342 | 0.068137 | 两层对齐没有互补 |
| A5 | 0.108709 | **0.067826** | 验证 NMAE 最低 |

### 26.4 测试集结果与正式选择

测试集正式主口径为 corrected candidate；括号内是套用 frozen G0 的 fused
诊断，不参与 Stage A 正式选择。

| 变体 | Candidate NRMSE | Candidate NMAE | 相对 A0 NRMSE | Frozen-G0 fused NRMSE | Frozen-G0 fused NMAE | NRMSE 不退化场站 | 正式结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A0 | 0.116160 | 0.081113 | — | **0.113761** | 0.077609 | 5/5 | **正式保留** |
| A1 | **0.114888** | 0.077660 | +1.095% | 0.114316 | **0.075995** | 3/5 | 数值最低，但未过稳健门 |
| A2 | 0.118646 | 0.077829 | -2.140% | 0.118600 | 0.076966 | 2/5 | 退化 |
| A3 | 0.115422 | 0.078231 | +0.635% | 0.114922 | 0.076409 | 2/5 | 跨站不稳 |
| A4 | 0.117097 | 0.077876 | -0.807% | 0.116780 | 0.076544 | 2/5 | 退化 |
| A5 | 0.116176 | **0.077321** | -0.014% | 0.115879 | 0.076075 | 2/5 | 近似持平但不晋级 |

Stage B 原守门要求同时满足：

- 5/5 场站 candidate NRMSE 严格改善；
- 5/5 场站 candidate NMAE 不退化；
- Macro NRMSE 和 NMAE 均至少改善 0.5%；
- dynamic、ramp-up、ramp-down、`change_ge_20` 均不退化；
- 16 个 horizon 均不退化。

A1–A5 没有任何一个通过，因此：

1. 正式选择回退 A0/WindPRISM；
2. 不启动 xCPD/variable×patch Stage B；
3. 不为“论文缺结构创新”继续增加训练；
4. A1 的宏平均改善只能作为有价值的否定性结果，不能替代跨场站稳健证据。

### 26.5 归因边界与早停故障修复

A0 没有做相同训练预算的 continuation control。因而 A1–A5 相对 A0 的绝对差异
可能混有继续训练效应；模块净效应优先依靠 A2/A3/A4/A5 的同父比较。若 A1
曾晋级核心方法，严格因果论证应补 A0R；但所有新变体均被否决后，没有必要为
最终 WindPRISM 主张继续训练 A0R。

首次 A2 早停时曾错误抛出：

~~~text
ValueError: a2/<farm_id> history为空或包含非有限值
~~~

根因是阶段式训练在 early stopping 后返回的 history 结构与完整训练假设不同，
代码把“存在空的非必需阶段”误判为整个 history 无效。提交 `a2cddc7` 修正为
按实际执行阶段和有限值检查；这不改变模型与既有有效结果。

### 26.6 关键产物

~~~text
01_stage_a_timealign_residual_alignment_qdf/
├─ stage_a_experiment_manifest.csv
├─ stage_a_training_metrics.csv
├─ stage_a_training_complexity.csv
├─ stage_a_training_bundle_complete.json
└─ testdata_predict_output/
   ├─ stage_a_test_final_selection.md
   ├─ stage_a_test_variant_comparison.csv
   ├─ stage_a_test_pairwise_vs_a0.csv
   ├─ stage_a_test_candidate.csv
   ├─ stage_a_test_horizon.csv
   ├─ stage_a_test_regime.csv
   ├─ stage_a_test_calibration.csv
   ├─ stage_a_test_safety.csv
   └─ stage_a_test_bundle_complete.json
~~~

协议必须写为 `legacy_seen_test_selected`。原计划“只用验证集解锁”的文字没有
在实际执行中成立，手稿不得把本轮包装成预注册确认性试验。

---

## 27. Part 3 Round 2：WindPRISM 五站公平从零重训

### 27.1 为什么需要这一轮

WindPRISM 的历史 F7/G0 结果使用 batch=192，并沿着 B2→R→F 的开发链得到。
原 PatchTST 和旧强基线曾使用 batch=256 等不同预算。为排除“最终模型只是因为
warm-start、batch 或训练预算不同而获胜”，Round 2 固定 F7/G0 结构并从随机
初始化开始完整重训。

训练代码：

~~~text
wind_part3_round2_f7_g0_strong_baseline_train.py
~~~

预测和十模型统一比较集成在：

~~~text
wind_dl_model_predict.py
~~~

结果目录：

~~~text
wind_results/part3_new_module_supplement/
└─ 02_strong_baseline_f7_g0_fair_training/
~~~

### 27.2 公平训练协议

| 项目 | 设置 |
| --- | --- |
| 初始化 | 全模型随机初始化，禁止加载 B2/F7/Stage-A checkpoint |
| seed | 2026 |
| batch | 256 |
| 最大 epochs | 80 |
| optimizer | Adam，lr=5e-4，clipnorm=1 |
| 主损失 | Huber delta=1 |
| forecast/candidate 权重 | 1.0 / 0.5 |
| early stopping | patience=10 |
| ReduceLROnPlateau | patience=4，factor=0.5，min lr=1e-6 |
| 监控 | `val_forecast_power_loss` |
| 参数量 | 20,969 |

5 个场站实际训练 11–13 epochs，best epoch 为 1–3，平均训练约 138.1 s/站。
训练阶段不读取 test；验证只用于 checkpoint，不按验证结果改变结构。

验证宏平均：

- fused NRMSE 0.107771、NMAE 0.068330；
- corrected candidate NRMSE 0.108760、NMAE 0.071062；
- Persistence NRMSE 0.114691、NMAE 0.068502。

### 27.3 五站测试结果

WindPRISM 公平重训的逐站 fused 结果：

| 场站尾号 | NRMSE | NMAE |
| --- | ---: | ---: |
| 5880 | 0.094599 | 0.066714 |
| 5895 | 0.135255 | 0.091260 |
| 5971 | 0.133903 | 0.094069 |
| 5975 | 0.134504 | 0.096180 |
| 6015 | 0.077075 | 0.045773 |
| **Macro** | **0.115067** | **0.078799** |

十模型统一 Macro：

| NRMSE 排名 | 模型 | Macro NRMSE | Macro NMAE | Macro R² |
| ---: | --- | ---: | ---: | ---: |
| 1 | WindPRISM 公平重训 | **0.115067** | 0.078799 | **0.856225** |
| 2 | HR-MoE FeTS-PatchTST | 0.116478 | **0.077475** | 0.852924 |
| 3 | PatchTST | 0.120938 | 0.081937 | 0.840172 |
| 4 | CNN-LSTM | 0.124576 | 0.089868 | 0.838360 |
| 5 | CNN-ResNet-GRU | 0.127413 | 0.094305 | 0.832072 |
| 6 | Informer | 0.129987 | 0.095530 | 0.825649 |
| 7 | Transformer | 0.130693 | 0.095176 | 0.825454 |
| 8 | BiLSTM | 0.142669 | 0.104699 | 0.801204 |
| 9 | Autoformer | 0.156602 | 0.116907 | 0.750376 |
| 10 | WaveNet | 0.156856 | 0.120289 | 0.762032 |

相对关键基线：

- 相对 HR-MoE，NRMSE 改善 1.211%，NMAE 退化 1.710%，参数减少 97.63%；
- 相对原生 PatchTST，NRMSE 改善 4.855%，NMAE 改善 3.829%，参数减少 90.06%；
- 逐 horizon H1–H16 的 Macro NRMSE 均排名第 1；
- 逐站 NRMSE 排名为 1/2/2/1/1，即 3 站第一、2 站第二。

因此公平结论应写成：

> 在相同级别的从零训练预算下，WindPRISM 取得最低 Macro NRMSE、所有 lead 的
> 最低 Macro NRMSE，并以 20,969 参数接近最优 NMAE；其优势不是旧 B2
> warm-start 或 batch=192 单独造成。

不能写成“所有指标均第一”，因为 HR-MoE 的 Macro NMAE 略低。

### 27.4 关键产物

~~~text
02_strong_baseline_f7_g0_fair_training/
├─ manifests/part3_round2_fair_training_protocol.json
├─ manifests/part3_round2_training_protocol_comparison.csv
├─ part3_round2_strong_baseline_training_metrics.csv
├─ part3_round2_strong_baseline_validation_summary.csv
├─ part3_round2_strong_baseline_training_bundle_complete.json
├─ models/  weights/  preprocess/  history/
├─ validation_diagnostics/  visualizations/
└─ testdata_predict_output/
   ├─ part3_round2_all_models_test_macro_comparison.csv
   ├─ part3_round2_all_models_test_metrics_summary.csv
   ├─ part3_round2_all_models_test_metrics_by_horizon.csv
   └─ predictions/ router_diagnostics/ weighted_curves/ figures/
~~~

---

## 28. Part 3 Round 3：外部 14 站无泄漏强基线基准

### 28.1 最终实验范围

原始补充数据：

~~~text
wind_split/supplementary_other_wind_data/
├─ JSFD001/
├─ ...
└─ JSFD014/
~~~

其它开发分支留下的 `processed_npz` 明确不作为输入。本轮从每个站原始功率和
测风 Excel 重新解析、审计、对齐、划分和构建特征。

三份入口代码：

~~~text
wind_part3_round3_external14_preprocess.py
wind_part3_round3_external14_all_models_train.py
wind_part3_round3_external14_all_models_predict.py
~~~

正式结果根目录：

~~~text
wind_results/part3_new_module_supplement/
└─ 03_external14_leakage_free_strong_baseline_benchmark/
~~~

### 28.2 最终数据预处理协议

#### 28.2.1 时间切分与因果边界

每站按时间顺序固定：

~~~text
前 70% → train
中 15% → validation
后 15% → test
~~~

并满足：

- 先划分，再只用训练段拟合 scaler、填充值、异常阈值和功率参考值；
- 历史窗口为 96 点，目标窗口为未来 16 点；
- 跨 split 的目标窗口重叠数必须为 0；
- 对功率和测风分别记录 timestamp semantics 与 `available_at`；
- 决策时刻 `t` 只能使用 `available_at <= t` 的历史记录；
- CCF 只作为训练段诊断，不根据 test 自动平移时间；
- 每站记录原始文件 SHA、处理代码 SHA、数组 SHA、split index SHA 和 bundle SHA。

用户后续确认原始时间戳无误；不过当前自动 artifact 仍把时间戳语义写为
`assumed_interval_start` 并将对齐状态标为 `uncertain_alignment`。手稿如要使用
“已人工核验”，应另存原始字段说明或人工审计签名，不能靠口头确认覆盖现有
artifact。

#### 28.2.2 固定 45 通道 schema

所有模型使用 `FEATURE_SCHEMA_V1`：

- 输入形状固定为 `(96, 45)`；
- 功率 target index 固定为 44；
- schema hash 为
  `a2f44e932044c2609a8c0e1cf6a446f37b4a0cfb71b8bf232a5bae6c568c680c`；
- 实际列顺序遵循代码和 manifest：
  **风速 → 气象 → 风向 sin/cos → 时间 → 派生 → 功率**；
- 关键语义列通过 alias 映射，不依赖原 Excel 的任意列位置；
- 训练前逐位校验与原 F7 schema，按语义重建每站
  `regime_feature_config.json`，训练阶段不再动态猜测。

原始可用通道少于 45 时，不改变模型输入维度，而是按训练段和物理语义重建：

- 标量传感器优先同高度、邻近高度、训练段统计和中性回退；
- 风向在角度空间做圆统计重建，再转换为 sin/cos；
- 禁止分别对 sin 和 cos 独立插值；
- 方向重建后检查单位圆约束；
- 缺失通道、重建来源和方向组可用性全部写入审计。

#### 28.2.3 功率参考值与指标命名

14 站没有统一、可信且已核验的装机容量字段，最终协议为：

1. 若未来能核验静态容量，则使用 capacity-normalized 指标；
2. 当前统一使用每站训练段功率的 99.9 分位数；
3. 参考值只由该站训练段计算并锁入 `power_reference.json`；
4. 论文必须称 **train-reference-normalized RMSE/MAE**，缩写
   **trNRMSE/trNMAE**；
5. 不得将其误写为严格装机容量归一化 NRMSE/NMAE。

#### 28.2.4 训练可行性与显存策略

训练窗口分级：

| 训练窗口数 | 等级 |
| ---: | --- |
| ≥40,000 | sufficient |
| 20,000–39,999 | limited |
| 5,000–19,999 | constrained |
| <5,000 | insufficient |

正式训练前在最大训练站对 HR-MoE、WaveNet、Transformer 做 batch=192 单 epoch
GPU 预检。若 HR-MoE 发生 OOM，代码可把该模型在所有 14 站统一回退为
batch=128，并在报告记录；实际运行中没有任何模型触发回退。

### 28.3 数据审计结果

整体：

- 14/14 场站预处理完成；
- 训练窗口 595,617，验证窗口 134,204，测试窗口 131,865；
- 每个模型共有 2,109,840 个测试预测点；
- 所有场站 `cross_split_target_overlap_count=0`；
- 9 站 `sufficient`，5 站 `limited`；
- limited 站为 JSFD005、JSFD006、JSFD007、JSFD008、JSFD012；
- 所有测试段均超过 7,000 个窗口，没有 insufficient test 站；
- 10 站原始可用 45 通道，3 站 38 通道，1 站 32 通道；
- 14 站方向组均可用于 WindPRISM，13 站直接可用、1 站语义重建；
- 14 站功率参考类型全部为 `train_power_q999`；
- 14 站均未按 CCF 自动移动时间。

CCF 诊断：

- 3 站稳定 exact-zero；
- 9 站为 weak/ambiguous；
- JSFD006 的峰值 lag 为 +2；
- JSFD012 的 lag 稳定性异常；
- 这些诊断不自动证明时间错位，只用于敏感性和人工核验。

数据质量中特别需要披露：

- JSFD014 原始功率存在 29,184 条重复记录，预处理按固定去重规则处理；
- JSFD003 只有 32 个原始可用通道，轮毂风速/风向通过语义重建；
- 训练受限站和 5 min 测风站高度重合，不能把子集差异单独归因于采样频率。

关键审计产物：

~~~text
data_audit/round3_external14_data_audit.csv
data_audit/round3_external14_split_manifest.csv
data_audit/round3_raw_data_audit.csv
data_audit/round3_timestamp_semantics.csv
data_audit/round3_time_alignment_diagnostics.csv
data_audit/round3_power_reference_protocol.json
data_audit/round3_power_reference_table.csv
data_audit/round3_regime_config_validation.csv
data_audit/round3_training_feasibility.csv
manifests/feature_schema_v1.json
round3_preprocess_bundle_complete.json
~~~

### 28.4 十五模型矩阵

共 14 个可训练模型和 1 个无需训练的基础基线：

| 类别 | 模型 | 场景适配要点 | 参数量 |
| --- | --- | --- | ---: |
| 原生强基线 | PatchTST | 原生 patch encoder 适配 96→16、45 通道 | 210,960 |
| 循环/卷积 | BiLSTM | 双向 LSTM 历史编码 | 107,920 |
| 循环/卷积 | CNN-LSTM | 局部卷积后 LSTM | 70,480 |
| 循环/卷积 | CNN-ResNet-GRU | 残差卷积 + GRU | 118,544 |
| 卷积 | WaveNet | 膨胀因果卷积 | 940,560 |
| Transformer | Transformer | 标准历史 token 自注意 | 858,512 |
| Transformer | Informer | 稀疏注意思想适配 | 484,240 |
| 分解 Transformer | Autoformer | 序列分解与自相关思想适配 | 212,737 |
| 历史复杂模型 | HR-MoE FeTS-PatchTST B6/v5ab | long/mid/short/Persistence 四专家 + horizon-regime router | 885,395 |
| 最终方法 | WindPRISM F7/G0 | Persistence + causal residual + P+H+D 门控 | 20,969 |
| 现代强基线 | iTransformer | variate-as-token、instance normalization、2 层 encoder，d_model=512 | 6,363,664 |
| 现代强基线 | TimesNet | FFT top-5 periods、TimesBlock、2D Inception-like conv | 4,709,917 |
| 现代强基线 | TimeMixer | 多尺度 96/48/24/12、PDM、RevIN-like normalization | 61,017 |
| 现代轻量基线 | DLinear | moving-average=25 的 trend/seasonal 分解，shared 96→16 linear | 3,104 |
| 无训练基线 | Persistence | 最后历史功率重复 16 步 | 0 |

iTransformer、TimesNet、TimeMixer 和 DLinear 均参考其官方思想改写为 Keras，
并适配本项目 4 h 风电任务；它们不是官方 PyTorch 仓库的逐位数值复现。适配时
没有加入 WindPRISM 专属 residual、candidate 或工况路由，以保持基线独立性。

### 28.5 训练协议、完成度与防重复机制

外部 14 站主训练共同设置：

- seed=2026；
- batch=192；
- Adam，lr=5e-4，clipnorm=1；
- Huber 主损失；
- 模型专属最大 epochs 沿用既定代码，现代强基线为 60，WindPRISM/HR-MoE
  保留相应既定预算；
- 每站每模型独立从零训练，不复用原 5 站权重；
- 只以 validation checkpoint 选择最佳 epoch；
- 每个可训练模型都在自身首次正式测试预测前恢复并冻结最佳 validation
  checkpoint；旧 10 模型先完成，之后 iTransformer、TimesNet、TimeMixer、
  DLinear 和 Persistence 作为一个冻结扩展批次追加。WindPRISM 及新增模型均未
  因已见测试结果再调参，但 `all_models_frozen_before_first_formal_test_prediction`
  按严格 marker 为 false。

完成度：

- 14 个可训练模型 × 14 站 = **196/196**；
- 15 个预测模型 × 14 站 = **210/210**；
- batch=192 的 196 个任务全部成功，OOM fallback=0、retry=0；
- 训练 `.keras`、weights、history CSV/PNG、validation/overfit JSON 均完整；
- 预测包括逐样本、逐站、逐 horizon、诊断、加权曲线和图片；
- prediction output inventory 共 1,776 项并通过完成校验；
- 可视化共 872 张，另含 14 站 WindPRISM gate diagnostics 和 14 站 HR-MoE
  router diagnostics。

### 28.6 验证集结果

14 站等权 Macro 验证结果；Persistence 无需训练，因此无验证项：

| 验证排名 | 模型 | Val trNRMSE | Val trNMAE | Val R² |
| ---: | --- | ---: | ---: | ---: |
| 1 | WindPRISM | **0.119764** | **0.076407** | **0.710754** |
| 2 | HR-MoE | 0.122165 | 0.077099 | 0.698169 |
| 3 | PatchTST | 0.122620 | 0.079565 | 0.697338 |
| 4 | DLinear | 0.124669 | 0.080689 | 0.686421 |
| 5 | TimeMixer | 0.126536 | 0.081130 | 0.677832 |
| 6 | CNN-LSTM | 0.128737 | 0.086946 | 0.664100 |
| 7 | iTransformer | 0.129217 | 0.082344 | 0.663782 |
| 8 | TimesNet | 0.131467 | 0.086092 | 0.652597 |
| 9 | CNN-ResNet-GRU | 0.131959 | 0.088071 | 0.649329 |
| 10 | Transformer | 0.133595 | 0.091545 | 0.639758 |
| 11 | BiLSTM | 0.134059 | 0.091783 | 0.638875 |
| 12 | Informer | 0.135039 | 0.092773 | 0.633182 |
| 13 | WaveNet | 0.145001 | 0.102231 | 0.577452 |
| 14 | Autoformer | 0.146967 | 0.104818 | 0.570348 |

WindPRISM 在验证集和最终测试集的首位一致，降低了“只在 test 偶然翻转”的风险。
验证指标用于 checkpoint 和记录，不与不同训练 loss 的原始 `val_loss` 直接混排。

### 28.7 测试集最终十五模型主表

选择规则预先固定为：

1. 最小等权 Macro trNRMSE；
2. 再比较 Macro trNMAE；
3. 再比较逐站平均 NRMSE 排名；
4. 最后比较参数量。

| 排名 | 模型 | Macro trNRMSE | Macro trNMAE | Macro R² | 平均排名 | 单站第一数 | 参数 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **WindPRISM** | **0.128801** | **0.082437** | **0.769094** | **2.214** | **6** | 20,969 |
| 2 | DLinear | 0.130812 | 0.086161 | 0.763759 | 2.786 | 3 | 3,104 |
| 3 | PatchTST | 0.131618 | 0.085978 | 0.759770 | 3.357 | 1 | 210,960 |
| 4 | HR-MoE FeTS-PatchTST | 0.132615 | 0.084063 | 0.760356 | 3.500 | 3 | 885,395 |
| 5 | TimeMixer | 0.133456 | 0.086388 | 0.753212 | 4.714 | 0 | 61,017 |
| 6 | Persistence | 0.136696 | 0.082602 | 0.738429 | 6.429 | 1 | 0 |
| 7 | iTransformer | 0.139122 | 0.089145 | 0.733530 | 7.357 | 0 | 6,363,664 |
| 8 | CNN-LSTM | 0.141931 | 0.096025 | 0.725519 | 7.857 | 0 | 70,480 |
| 9 | TimesNet | 0.144317 | 0.095298 | 0.713931 | 9.643 | 0 | 4,709,917 |
| 10 | CNN-ResNet-GRU | 0.150481 | 0.100588 | 0.693351 | 10.714 | 0 | 118,544 |
| 11 | BiLSTM | 0.152963 | 0.106070 | 0.687698 | 10.714 | 0 | 107,920 |
| 12 | Transformer | 0.153482 | 0.105506 | 0.681811 | 10.643 | 0 | 858,512 |
| 13 | Informer | 0.157688 | 0.107953 | 0.663779 | 12.143 | 0 | 484,240 |
| 14 | WaveNet | 0.169573 | 0.120056 | 0.613785 | 13.643 | 0 | 940,560 |
| 15 | Autoformer | 0.170210 | 0.121373 | 0.611507 | 14.286 | 0 | 212,737 |

WindPRISM 的 Micro pooled-points 结果也全部第一：

- trNRMSE 0.132537；
- trNMAE 0.081770；
- R² 0.834648；
- RMSE 18.005323 MW；
- MAE 9.493768 MW。

相对 Macro trNRMSE 改善：

| 对手 | WindPRISM 相对改善 |
| --- | ---: |
| DLinear | 1.537% |
| PatchTST | 2.140% |
| HR-MoE | 2.876% |
| TimeMixer | 3.488% |
| Persistence | 5.776% |
| iTransformer | 7.418% |
| TimesNet | 10.751% |

### 28.8 逐站和逐 horizon 解释

单站 NRMSE 第一：

- WindPRISM：JSFD001、003、004、009、011、013，共 6 站；
- DLinear：JSFD005、007、010，共 3 站；
- HR-MoE：JSFD002、008、012，共 3 站；
- PatchTST：JSFD014；
- Persistence：JSFD006。

WindPRISM 的 14 站名次分布为 6 次第 1、5 次第 2、1 次第 4、1 次第 5、
1 次第 6。最差场站为 JSFD008，trNRMSE 0.204121；这说明宏平均最优并不意味着
每个场站都最优。

逐 horizon NRMSE：

- H1（15 min）：DLinear 0.040179 第 1，WindPRISM 0.041036 第 3；
- H2（30 min）：DLinear 0.063319 第 1，WindPRISM 0.063721 第 2；
- H3–H16（45–240 min）：WindPRISM 连续 14 个 lead 第 1；
- H8（120 min）：WindPRISM 0.128062；
- H12（180 min）：WindPRISM 0.152196；
- H16（240 min）：WindPRISM 0.170022，优于 PatchTST 0.172477、
  DLinear 0.173470、HR-MoE 0.175222、TimeMixer 0.176001。

逐 horizon NMAE：

- Persistence 在 H1–H9 最低；
- WindPRISM 在 H10–H16 最低。

这构成最清晰的机制解释：Persistence 对极短 lead 的绝对误差非常强，
WindPRISM 的 residual/gate 价值主要体现在 45 min 以后，并随预测时距扩大。

### 28.9 W/T/L 与统计显著性

WindPRISM 的逐站 NRMSE W/T/L：

- 对 DLinear、PatchTST、HR-MoE：均为 10/0/4；
- 对 TimeMixer：11/0/3；
- 对 Persistence、iTransformer：均为 13/0/1；
- 对 TimesNet：14/0/0；
- 对其它 7 个旧基线：均为 14/0/0。

Wilcoxon signed-rank + Holm 校正与 farm bootstrap：

| 对手 | Mean Δ trNRMSE（WindPRISM-对手） | 95% CI | Holm p | 显著 |
| --- | ---: | ---: | ---: | --- |
| PatchTST | -0.002817 | [-0.005191, -0.000108] | 0.148315 | 否 |
| HR-MoE | -0.003814 | [-0.009729, 0.001202] | 0.156982 | 否 |
| DLinear | -0.002011 | [-0.004554, 0.000776] | 0.156982 | 否 |
| iTransformer | -0.010320 | [-0.014171, -0.006328] | 0.004272 | 是 |
| TimesNet | -0.015516 | [-0.019930, -0.011629] | 0.001709 | 是 |
| TimeMixer | -0.004654 | [-0.007306, -0.001659] | 0.042969 | 是 |
| Persistence | -0.007895 | [-0.011082, -0.004996] | 0.002197 | 是 |

总体为显著优于 14 个对手中的 11 个。最接近的 PatchTST、HR-MoE 和 DLinear
没有通过 Holm 校正，因此手稿只能写“描述性综合最优，并显著优于 11/14 个
比较模型”，不能写“显著优于所有强基线”。

### 28.10 WindPRISM 门控机制在外部站的表现

14 站平均 gate 约为 0.928833，表示绝大多数样本采用较强的 corrected candidate
权重，但仍保留 Persistence 锚点。

- fused 相对 corrected candidate：Macro trNRMSE 改善约 0.397%，
  trNMAE 改善约 1.249%；
- fused 相对 Persistence：trNRMSE 改善约 5.776%，trNMAE 改善约 0.200%。

因此最终性能的主要来源是轻量 residual 候选质量，G0 门控提供较小但稳定的
二次收益。论文不应把门控写成全部增益来源，也不应把没有晋级的 G1–G4
校准安全机制并入最终模型。

### 28.11 复杂度与 Pareto 结论

WindPRISM 参数量：

- 比 PatchTST 少 90.06%；
- 比 HR-MoE 少 97.63%；
- 比 TimeMixer 少 65.63%；
- 比 iTransformer 少 99.67%；
- 比 TimesNet 少 99.55%。

NRMSE–参数 Pareto 前沿为：

~~~text
Persistence → DLinear → WindPRISM
~~~

因此 WindPRISM 可以称“精度—参数 Pareto 高效”或“相对复杂深度基线极轻”，
但不能称“绝对最小”：DLinear 只有 3,104 参数，Persistence 为 0 参数。

196 个训练任务记录的 fit time 合计约 17.278 GPU h；WindPRISM 合计约
0.228 h，平均 58.73 s/站。12/196 个任务被启发式标记为 probable overfit，
WindPRISM 只有 JSFD005 一项。

旧 10 模型主要在 4090 服务器生成，新增现代基线主要在本机 3080 Ti Laptop
GPU 完成，因此现有 inference seconds 混有硬件差异。参数量和文件大小可以公平
比较；若论文强调实时部署速度，必须在同一硬件、同一 batch 和相同 warm-up 下
复测冻结模型。

### 28.12 敏感性分析

以下是从归档逐站 CSV 只读聚合的 post-hoc 敏感性，不是预注册主分析：

| 子集 | WindPRISM trNRMSE | 最优模型/结果 | 解释 |
| --- | ---: | --- | --- |
| 9 个 sufficient 站 | **0.114757** | WindPRISM；PatchTST 0.118742，DLinear 0.119325 | 数据充分时优势稳定 |
| 5 个 limited 站 | 0.154081 | DLinear 0.151490；HR-MoE 0.151739 | 训练受限是主要弱点 |
| 排除 JSFD006/012 | **0.129165** | WindPRISM；DLinear 0.131334，PatchTST 0.131383 | 排除强 CCF warning 后结论不变 |
| 10 个原始 45 通道站 | **0.128147** | WindPRISM；DLinear 0.131044 | 完整通道子集稳定 |
| 4 个重建通道站 | 0.130436 | DLinear 0.130233；WindPRISM trNMAE 更好 | 缺失重建子集 NRMSE 优势消失 |
| 排除 JSFD014 | **0.128233** | WindPRISM | 重复数据站不改变总排名 |

敏感性结论：

- 总排名对排除时间对齐 warning 和重复数据站较稳健；
- WindPRISM 的主要薄弱处是训练受限和需重建通道的站；
- 5 min 测风与 limited 站高度混杂，不能声称“5 min 采样导致模型失败”；
- 手稿应把这些结果放在鲁棒性/局限性，而不是选择性删除不利站点。

### 28.13 结果与图片路径

首选主文件：

~~~text
testdata_predict_output/round3_external14_test_final_selection.md
testdata_predict_output/round3_external14_test_macro_micro.csv
testdata_predict_output/round3_external14_test_metrics_per_farm.csv
testdata_predict_output/round3_external14_test_metrics_by_horizon.csv
testdata_predict_output/round3_external14_average_rank.csv
testdata_predict_output/round3_external14_win_tie_loss.csv
testdata_predict_output/round3_external14_significance.csv
testdata_predict_output/round3_external14_complexity.csv
validation_metrics/round3_external14_validation_metrics.csv
validation_metrics/round3_overfit_diagnostics.csv
~~~

完成标记：

~~~text
round3_preprocess_bundle_complete.json
round3_training_bundle_complete.json
testdata_predict_output/round3_external14_prediction_bundle_complete.json
testdata_predict_output/round3_external14_output_inventory.csv
~~~

论文优先图片：

~~~text
testdata_predict_output/visualizations/overview/macro_nrmse_nmae_bars.png
testdata_predict_output/visualizations/overview/nrmse_model_farm_heatmap.png
testdata_predict_output/visualizations/overview/average_farm_rank.png
testdata_predict_output/visualizations/overview/parameter_nrmse_pareto.png
visualizations/data_quality/round3_external14_preprocess_overview.png
~~~

各模型目录还保存：

- `.keras` 完整模型和 `.weights.h5` 最佳权重；
- 三子图训练 history；
- 逐样本预测 CSV；
- 单窗真实/预测曲线、加权聚合曲线；
- 每站/每 horizon 指标图；
- WindPRISM gate 与 HR-MoE router 诊断；
- 数据审计、复杂度、运行日志和 hash marker。

---

## 29. 工程复现、依赖、远程迁移与续跑归档

### 29.1 分支与关键提交

主线历史：

~~~text
0b72b49  Merge dev-FeTS-PatchTST into main
142f6b6  add Stage A TimeAlign-QDF experiments
a2cddc7  fix phased Stage A early-stop history
d98eedb  add fair F7/G0 strong-baseline comparison
13e0671  add leakage-free external14 benchmark pipeline
cc3af82  harden external14 evaluation protocol
f5280f1  add pinned requirements.txt
83c77fd  package reproducible Round 3 remote runtime
464e50f  add iTransformer/TimesNet/TimeMixer/DLinear/Persistence
e3431f3  allow strict local resume after SSH relocation
~~~

Part 3 位于 `dev-new-module`；截至本次文档更新前，代码 HEAD 与
`origin/dev-new-module` 均为 `e3431f3`。本次仅更新归档文档，除非另行要求，
不自动创建提交。

### 29.2 两份 requirements 为什么不合并

~~~text
requirements.txt
requirements-round3-lock-linux-py39-gpu.txt
~~~

二者职责不同：

- `requirements.txt`：项目直接依赖及项目认可的精确版本，供一般安装和代码审阅；
- `requirements-round3-lock-linux-py39-gpu.txt`：已验证 Linux/Python 3.9/GPU
  环境的完整传递依赖锁，用于尽量逐包复现远程环境。

把两者强行合并会把平台专属传递包、CUDA 相关约束和一般项目依赖混在一起，
降低跨平台可用性，因此保持分离是正确的。

已验证组合：

- Python 3.9.25；
- TensorFlow/Keras 2.14；
- CUDA 11.8；
- cuDNN 8.7。

租用镜像若只有 Python 3.8 + TensorFlow 2.9 + CUDA 11.2，即使配 5090，
也不能直接满足现有锁定环境：TensorFlow 2.9 对新 GPU/CUDA 组合和当前保存格式/
自定义层均存在兼容风险。正确做法是选择支持 Python 3.9、TF 2.14、CUDA 11.8
的镜像或容器，而不是仅为迁就旧镜像降低项目依赖。

### 29.3 远程训练包

根目录归档：

~~~text
ROUND3_REMOTE_TRAINING_README.md
wind_part3_round3_external14_remote_training_bundle_83c77fd.zip
~~~

压缩包约 143 MB，只包含三份 Round 3 代码的完整依赖闭包、14 站原始数据、
requirements、必要的上游模型定义和保持相对路径的目录。远程解压后按项目根
目录运行，避免仅上传三份入口代码导致自定义 Keras 层和超参数导入失败。

### 29.4 SSH 中断后严格续跑

基础 10 模型在远程 4090 环境完成；随后 SSH 频繁掉线，新增四个现代可训练
基线转到本机 3080 Ti Laptop GPU。提交 `e3431f3` 允许以下严格迁移：

- 只把已知旧根 `/root/digitalchina2026` 的路径按相同相对路径重定位到当前根；
- 继续校验文件大小、SHA、schema、模型×场站身份和 marker；
- 只接受精确允许的前一版代码 SHA；
- 不按 basename 全盘搜索；
- 不接受任意代码漂移；
- 不修改旧 marker，只新增当前运行的完成记录。

这不是“放宽哈希让任何结果都能混入”，而是对同一 artifact 的可审计根路径
迁移。

只训练新增四个可训练模型：

~~~bash
python wind_part3_round3_external14_all_models_train.py \
  --models itransformer,timesnet,timemixer,dlinear \
  --farms all \
  --resume
~~~

统一预测必须运行完整矩阵入口：

~~~bash
python wind_part3_round3_external14_all_models_predict.py --resume
~~~

旧模型会通过 marker/hash 只读复用；Persistence 自动生成，不进入训练。
`--partial/--smoke` 只用于调试，不能发布到正式结果目录代替完整 bundle。

---

## 30. 截至 2026-07-27 的一区创新判断与论文主线

### 30.1 最终可支撑的三项主要贡献

#### 贡献 1：Persistence 锚定的轻量因果修正主干

WindPRISM 不直接让大型 Transformer 学完整功率轨迹，而是：

1. 用最后历史功率形成低方差 Persistence 候选；
2. 用 18,416 参数级 causal residual 学习未来 16 步偏离；
3. 最终全模型只有 20,969 参数。

直接证据：

- B0/B1/B2/B6 最小有效结构搜索；
- Round 2 从零公平重训；
- Round 3 对 HR-MoE、PatchTST、iTransformer、TimesNet、TimeMixer、
  DLinear 和 Persistence 的统一比较；
- H3–H16 持续第 1，说明 residual 在较长 lead 的累积修正价值。

#### 贡献 2：显式风电工况驱动的 sample×horizon 双候选融合

36 维 P+H+D 工况上下文明确表示：

- P：历史功率形态与变化；
- H：轮毂高度风速及其统计；
- D：风向变化与转向。

工况编码器输出样本×horizon gate，在 Persistence 和 corrected candidate
之间动态融合。直接证据来自 R0–R6、F0–F8、FP0/FP4 和 G0–G4，而不是仅用
attention heatmap 解释。

#### 贡献 3：候选—门控—融合分层的可靠性和否定性评价框架

项目系统区分：

~~~text
candidate 是否变好
→ gate 是否可辨识/可校准
→ fused 是否真正获益
→ dynamic/ramp/逐站是否安全
~~~

通过 Frozen-Pair、soft oracle、Q90、Brier/ECE、regret/harm、Stage 4B、
X1R 和 Part 3 A0–A5 说明：

- 更好的 candidate 不保证最终 fused 更好；
- 更好的 calibration 不保证 NRMSE/ramp 更好；
- 更复杂时频、多尺度、对齐或 QDF 不应在守门失败后进入最终模型。

第三项属于“机制验证与可靠评价贡献”，不能误写成最终结构含有所有被否决模块。

### 30.2 为什么当前比早期 Q2 判断更强

| 早期短板 | 当前补证 |
| --- | --- |
| 只在原 5 站、测试参与选型 | 新增 14 站严格时序处理和冻结强基线比较 |
| 训练预算可能不公平 | Round 2 从零、batch=256 的公平重训 |
| 缺现代强基线 | iTransformer、TimesNet、TimeMixer、DLinear、Persistence 已补 |
| 缺统计 | 逐站 W/T/L、Wilcoxon-Holm、farm bootstrap CI 已完成 |
| 缺复杂度系统比较 | 15 模型参数、模型大小、Pareto 已完成 |
| 只报宏平均 | Macro/Micro、逐站、逐 horizon、子集敏感性均已分析 |
| 为创新可能继续堆结构 | A0–A5 守门失败后停止 Stage B，负结果完整 |

这些证据使论文从“方法线清楚但外部支撑不足的 Q2”提升为
**可合理冲击专业型 Q1 的完整实验稿**。

### 30.3 仍然限制 Q1 把握的事实

1. 对最接近的 DLinear、PatchTST、HR-MoE，Holm 校正后未显著；
2. 相对前三者的 trNRMSE 增益为 1.54%–2.88%，不是压倒性提升；
3. 外部 14 站是每站自身训练的 chronological holdout，不是未见站零样本迁移；
4. 14 站有历史数据暴露，因此 marker 明确不是 globally blind；
5. 14 站全部用训练段 q99.9，而不是经核验装机容量；
6. 训练受限/重建通道子集上 DLinear 的 NRMSE 略优；
7. 现有推理时间混合 4090 和 3080 Ti，不能直接支持统一硬件延迟主张；
8. Round 3 为 seed=2026；用户所述多 seed 证据在手稿引用前必须定位 artifact；
9. 最终结构创新是强领域问题驱动，而不是新的通用 Transformer backbone。

因此投稿判断是：

> WindPRISM 具备专业型 SCI/JCR Q1 的投稿基础，但属于 Q1 边缘/有竞争力尝试，
> 不是高把握顶级一区。若目标期刊偏好能源场景机制、轻量部署和多站系统证据，
> 匹配度较高；若更强调通用时序架构的基础新颖性，风险较大。

### 30.4 论文推荐结构

1. **Introduction**
   - 4 h 超短期预测中的 Persistence 强基线；
   - 非平稳风况下修正必要性；
   - 大模型复杂度和门控可靠性问题；
   - 三项贡献概述。
2. **Related Work**
   - 风电物理/工程先验；
   - Patch/Transformer/时频多尺度；
   - 动态门控、MoE 与可靠性评价；
   - 轻量时序预测与现代基线。
3. **Problem Formulation and Protocol**
   - 96→16、无未来 NWP；
   - 原 5 站与补充 14 站的证据分层；
   - 45 通道、available_at、70/15/15；
   - capacity-normalized 与 train-reference-normalized 指标区别。
4. **WindPRISM Method**
   - Persistence anchor；
   - lightweight causal residual；
   - P+H+D explicit regime encoder；
   - sample×horizon two-candidate fusion；
   - 参数和训练目标。
5. **Controlled Model Development**
   - B/R/F/FP 主消融；
   - candidate drift 控制；
   - 为什么选择 F7/G0。
6. **Experimental Setup**
   - 19 站两层数据；
   - 15 强基线及 Keras 适配；
   - 训练、显著性、复杂度和完成审计。
7. **Main Results**
   - 原 5 站公平结果；
   - 外部 14 站 Macro/Micro 主表；
   - 逐站、逐 horizon、W/T/L、统计。
8. **Mechanism, Reliability and Negative Ablations**
   - gate/candidate/Persistence；
   - G/T/D/X/X1R；
   - A0–A5 否定结果；
   - 参数 Pareto 和敏感性。
9. **Limitations**
   - 非 globally blind；
   - q99.9 reference；
   - limited/reconstructed 子集；
   - 单 seed 主归档和混合硬件时延。
10. **Conclusion**
    - 领域先验、轻量、跨站验证和适用边界。

### 30.5 摘要与正文可用的安全表述

推荐：

> Across 14 supplementary wind farms and 15 forecasting models, WindPRISM
> achieved the lowest equal-farm Macro and pooled Micro trNRMSE, ranked first
> on six farms, and retained the best Macro trNRMSE from 45 to 240 minutes,
> while using 20,969 trainable parameters.

> Paired station-level tests showed significant improvements over 11 of the
> 14 baselines after Holm correction; differences from DLinear, PatchTST and
> HR-MoE FeTS-PatchTST were not statistically significant.

不推荐：

- “state-of-the-art on every farm and every metric”；
- “statistically superior to all baselines”；
- “the smallest or fastest model”；
- “blind unseen-farm generalization”；
- “capacity-normalized error”用于 JSFD14；
- 把未晋级的 TimeAlign、QDF、时频、多尺度或安全门控写进最终计算图。

### 30.6 投稿建议如何理解

第 22 节列出的 IET Renewable Power Generation、Wind Energy、Energy Reports
是此前按较保守 Q2 证据给出的匹配/保底建议，不等于当前必须投这三本。若以 Q1
为目标，投稿前需按当年 JCR/中科院分区和 scope 重新核验 Renewable Energy、
Applied Energy、Energy、Energy Conversion and Management 等更高目标的适配性。
分区和审稿偏好会变化，本档案不把任何期刊写成“高概率录取”承诺。

---

## 31. 后续是否还需要补实验

### 31.1 已经完成，不建议重复的内容

- 原 5 站 B/R/F/FP/G/T/D/X/X1R 全链路；
- Part 3 A0–A5 高成功先验模块验证；
- 五站公平从零重训；
- 外部 14 站严格时序预处理；
- 15 模型强基线；
- Persistence 无训练基线；
- 逐站、逐 horizon、W/T/L、NRMSE 配对显著性；
- 参数量、模型大小、history、预测曲线和主可视化；
- batch=192 显存预检和 OOM 回退记录；
- 用户确认的时间戳语义核验，不需要再自动平移；
- 用户确认同模型多 seed 结论一致，不应为形式重复跑 Round 3 三 seed。

### 31.2 为什么不在外部 14 站再做 P/H/D 门控消融

P/H/D、候选漂移和门控机制已经由原 5 站 R/F/FP/G 系列直接验证。外部 14 站的
角色是补充数据上的冻结强基线比较。现在再根据其测试结果训练 F0–F8/G0–G4：

- 复制已回答的机制问题；
- 产生 14×多变体的大量重复训练；
- 让补充测试进一步参与结构选择；
- 降低其作为新增数据证据的可信度。

因此不补外部 14 站机制消融是合理设计，而不是实验缺失。论文应把原 5 站称为
development/mechanism cohort，把 14 站称为 supplementary benchmark cohort。

### 31.3 仍建议完成的无重训后处理

这些工作使用现有预测，不改变模型：

1. 对 NMAE 也执行配对 Wilcoxon-Holm、farm bootstrap CI 和效应量；
2. 增加 Friedman omnibus test 与 Nemenyi/平均排名 CD 图；
3. 把第 28.12 节 post-hoc 子集聚合固化成 CSV/JSON 和图片；
4. 生成 H1–H16 的相对提升/置信区间图；
5. 汇总 WindPRISM gate 在场站、horizon、工况上的分布；
6. 把多 seed 证据的精确代码、模型、CSV 和 seed 列表加入附录索引；
7. 归档用户确认时间戳无误的原始字段/业务说明；
8. 在同一 GPU 上只对冻结的 WindPRISM、DLinear、PatchTST、HR-MoE、
   TimeMixer 做统一 warm-up、batch、重复次数的推理速度/峰值显存测试；
9. 生成参数、模型文件、FLOPs、延迟、吞吐的统一硬件表。

这些是当前最具性价比的 Q1 手稿补强，不需要重新训练预测模型。

### 31.4 唯一可能显著抬高证据等级的新实验

若作者仍想进一步提高 Q1 把握，唯一高价值的新训练实验是：

- 完全未触碰的新场站或新时间段；
- 在看到标签前冻结预处理、模型、超参数和评价脚本；
- 一次性运行；
- 不再根据结果调模型；
- 最好同时给出跨站训练→未见站推理或 rolling-origin。

这属于新的确认性研究，不是当前论文“漏做的常规消融”。即使不做，现有稿件
也已具备投稿基础；做了且结果保持，才能更强地使用 independent confirmation
或 unseen-site generalization 表述。

### 31.5 当前终止决策

截至本档案更新：

~~~text
模型结构开发：停止
X2–X6：不启动
TimeAlign/QDF Stage B：不启动
外部14站P/H/D再消融：不启动
新增常规强基线：已完成
主要工作：统计后处理 → 图表冻结 → SCI Q1手稿写作
~~~

---

## 32. 新对话快速接手索引

### 32.1 建议阅读顺序

新对话若目标是写 SCI 一区手稿，按以下顺序读取：

1. 本文第 1–2 节：当前模型、双证据协议与术语；
2. 第 5、7–10 节：WindPRISM 的创新来源和直接消融；
3. 第 12–14 节：candidate→gate 闭环与失败边界；
4. 第 25–27 节：Part 3 的补充路线和公平重训；
5. 第 28 节：外部 14 站最终数据、15 模型主表和统计；
6. 第 30–31 节：创新主张、投稿边界和剩余工作；
7. `docs/WIND_FETS_PATCHTST_MODEL_DEVELOPMENT_CONTEXT.md`：更早的工程演进细节。

### 32.2 手稿主数据只读入口

原 5 站最终结构：

~~~text
wind_results/regime_encoder_patchtst/
└─ stage2_feature_screening_f0_f7/
~~~

原 5 站公平强基线：

~~~text
wind_results/part3_new_module_supplement/
└─ 02_strong_baseline_f7_g0_fair_training/
   └─ testdata_predict_output/
      └─ part3_round2_all_models_test_macro_comparison.csv
~~~

外部 14 站最终主结果：

~~~text
wind_results/part3_new_module_supplement/
└─ 03_external14_leakage_free_strong_baseline_benchmark/
   ├─ data_audit/
   ├─ validation_metrics/
   └─ testdata_predict_output/
      ├─ round3_external14_test_final_selection.md
      ├─ round3_external14_test_macro_micro.csv
      ├─ round3_external14_test_metrics_per_farm.csv
      ├─ round3_external14_test_metrics_by_horizon.csv
      ├─ round3_external14_significance.csv
      └─ round3_external14_complexity.csv
~~~

### 32.3 写作时必须固定的术语

| 概念 | 正确写法 |
| --- | --- |
| 最终模型 | WindPRISM (F7/G0)，20,969 parameters |
| 原五站指标 | capacity-normalized NRMSE/NMAE，若原 artifact 确实使用容量 |
| 外部十四站指标 | trNRMSE/trNMAE，train power q99.9 reference |
| 原五站角色 | development/mechanism cohort |
| 十四站角色 | supplementary within-station chronological benchmark |
| 测试性质 | descriptive frozen comparison, not globally blind |
| 最优表述 | best Macro/Micro descriptive performance |
| 显著性 | significant against 11/14 baselines, not against the nearest three |
| 复杂度 | parameter-efficient/Pareto-efficient, not the smallest |
| 最终结构不包含 | FeTS/PatchTST encoder、四专家、G1–G4、T1–T3、X1/X2–X6、A1–A5 |

### 32.4 当前一句话结论

> WindPRISM 以 Persistence 为物理锚点，通过轻量因果 residual 和 P+H+D
> 显式工况驱动的逐样本—逐 horizon 双候选融合，在原 5 站机制消融和公平重训中
> 定型，并在 14 个补充风场、15 个模型的严格时序统一比较中取得最佳 Macro/Micro
> trNRMSE、trNMAE 与 R²；其优势集中于 45–240 min lead 且参数仅 20,969，
> 但对 DLinear、PatchTST、HR-MoE 的差异未达到 Holm 校正显著，外部结果也不是
> globally blind，因此适合以严谨边界冲击专业型 SCI Q1，而非宣称无条件 SOTA。
