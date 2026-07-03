# 短期风电预测项目：模型工程上下文

> 项目根目录：`/mnt/d/Python/myprojects/digitalchina2026`
>
> 用途：在新对话中快速恢复本项目的数据、模型、训练、消融实验、预测和后续研究背景。
>
> 本文只记录与模型工程和模型优化直接相关的内容。

## 1. 项目任务

项目面向多个风电场的超短期功率预测：

- 数据频率：15 分钟。
- 历史输入：96 点，即过去 24 小时。
- 预测长度：16 点，即未来 4 小时。
- 预测目标：`功率`。
- 每个风电场独立训练、保存和测试。
- 主要指标：MAE、MSE、RMSE、R2、MAPE、SMAPE、容量归一化 MAE 和 NRMSE。
- 除窗口级指标外，还对重叠窗口进行指数加权，生成完整测试时间轴曲线。

当前主要研究问题是：

1. PatchTST 如何适配短期风电场景。
2. tuned PatchTST 如何在不同风电场上稳定泛化。
3. 如何同时改善 RMSE/NRMSE、MAE 和完整时间轴指标。
4. 如何形成具有 SCI 方法创新价值、而不只是训练技巧组合的模型。

## 2. 数据约定

数据目录：

```text
./wind_split
```

只读取目录第一层的人工命名文件，不递归读取任何子目录：

```text
wind_train_<farm_id>.csv
wind_test_<farm_id>.csv
```

五个场站：

```text
4081950112845135880
4081950112845135895
4081950112845135971
4081950112845135975
4081950112845136015
```

测试文件虽然包含实际未来功率，但必须满足：

- 模型输入只能使用预测起点之前已观测到的历史功率。
- 未来真实功率不得进入输入窗口。
- 测试文件中的未来功率只用于 `y_true`、评价指标和可视化。

该约束在 `wind_dl_model_predict.py::prepare_prediction_arrays()` 中实现。

## 3. 数据预处理

主要实现位于：

```text
wind_dl_model_train.py::load_and_preprocess()
```

其它深度学习模型复用同一流程：

1. 按 `时间` 排序、去重。
2. 恢复完整的 15 分钟时间索引。
3. 非数值内容转为缺失值。
4. 从 `装机` 获取容量中位数，随后从输入特征删除。
5. 修复少量气压/湿度疑似互换异常。
6. 清洗风速、温度、气压、湿度和功率物理异常。
7. 功率裁剪到 `[0, capacity]`。
8. 风向转换为 sin/cos。
9. 时间插值、前向填充、后向填充和零填充。
10. 添加日内、星期、年内和月份周期特征。
11. 添加风速平方、立方、轮毂高度风速差和比值等物理特征。
12. `StandardScaler` 分别缩放输入和目标功率。
13. 历史功率作为模型输入通道。

典型单场站预处理后约为：

```text
数据形状：(99360, 45)
滑动窗口：99249
训练窗口：84361
验证窗口：14888
```

## 4. 核心代码

### 4.1 `wind_dl_model_train.py`

原生 PatchTST 风电训练代码。

主要配置：

```text
history_len = 96
forecast_len = 16
patch_len = 16
patch_stride = 8
d_model = 64
n_heads = 4
n_layers = 3
d_ff = 128
dropout = 0.15
batch_size = 256
epochs = 80
validation_split = 0.15
```

结构：

- 每个变量独立切分 patch。
- patch 投影至 `d_model`。
- 可学习位置编码。
- 三层 Transformer encoder。
- 恢复 channel 维度。
- 拼接目标功率通道和全通道上下文。
- MLP 输出未来 16 个功率点。
- Huber loss，MAE 和 RMSE。

原生 PatchTST 已完成，训练入口默认关闭；只有显式设置
`WIND_PATCHTST_ENABLE_TRAINING=1` 才会重训。

### 4.2 `wind_dl_other_models_train.py`

训练以下基线：

```text
bilstm
cnn_lstm
cnn_resnet_gru
wavenet
transformer
informer
autoformer
```

统一配置大致为：

```text
d_model = 64
n_heads = 4
n_layers = 2
d_ff = 128
dropout = 0.15
epochs = 60
validation_split = 0.15
```

Informer 和 Autoformer 是面向本项目 TensorFlow/Keras 管线和单张 3080 Ti 的轻量实现，不是官方代码逐行复制，但保留核心机制。

Informer 保留：

- 时间和变量 embedding。
- ProbSparse Self-Attention。
- sampled keys 估计 query sparsity。
- top queries 全量注意力。
- 非 top query 使用初始 context。
- encoder distilling。

Autoformer 保留：

- moving-average series decomposition。
- encoder/decoder 多级分解。
- FFT series-wise auto-correlation。
- top-k delay aggregation。
- seasonal/trend 初始化和重构。

其它基线已完成，训练入口默认关闭；只有显式设置
`WIND_OTHER_MODELS_ENABLE_TRAINING=1` 才会重训。

### 4.3 `wind_dl_tuned_patchtst_train.py`

负责 tuned PatchTST、自蒸馏和前五轮消融实验。

当前活动代码包含的 variant：

```text
baseline
revin
revin_cnn_adapter
revin_cnn_adapter_balanced_loss
revin_cnn_adapter_balanced_loss_swa
revin_balanced_loss
revin_balanced_loss_multiseed
revin_rmse_balanced_loss
revin_rmse_balanced_loss_no_distill
```

“滚动验证 + NRMSE checkpoint + baseline 回退 + 验证集加权集成”的第六轮代码当前不在活动版本中，但其模型和测试结果仍保留，作为已完成的历史实验。

为避免误运行覆盖第六轮 canonical artifact，历史训练入口现在默认关闭。
只有显式设置 `WIND_TUNED_ENABLE_TRAINING=1` 才会进入训练，且
round 1～5 的执行开关默认全部为关闭状态。
当前结构验证阶段 `WIND_TUNED_MULTI_SEEDS` 的默认值仅为 `2026`；
若要严格复现历史第三轮，需要显式设置为 `2026,2027,2028`。

### 4.4 `wind_dl_model_predict.py`

所有深度学习模型的统一预测入口：

```text
patchtst
tuned_patchtst
tuned_patchtst_external_teacher
tuned_patchtst_ramp_trajectory
tuned_patchtst_ramp_gated
tuned_patchtst_ramp_persistence_gated
bilstm
cnn_lstm
cnn_resnet_gru
wavenet
transformer
informer
autoformer
```

当前活动版本对多 seed 模型采用等权平均，不读取第六轮的非均匀 `ensemble_weights`。

因此：

- 当前代码适合前五轮的等权多 seed 预测。
- 第六轮正式预测结果应直接分析已保存的第六轮 CSV。
- 若重新复现第六轮非均匀权重预测，需要重新加入相应权重读取逻辑。
- 若 canonical artifact 仍为第六轮且含非均匀权重，预测脚本默认跳过整个
  `tuned_patchtst` 模型族，防止用等权结果覆盖历史 CSV。

### 4.5 补充数据 teacher 实验

新增文件：

```text
wind_supplementary_preprocess.py
wind_dl_external_teacher_train.py
```

补充数据来自：

```text
./wind_split/supplementary_other_wind_data/JSFD001
...
./wind_split/supplementary_other_wind_data/JSFD014
```

这些场站与五个目标场站来源无关，只用于 teacher 预训练，不进入目标场站
验证集或测试集。

实验使用独立模型名和输出目录：

```text
tuned_patchtst_external_teacher
./models/tuned_patchtst_external_teacher/
./wind_results/tuned_patchtst_external_teacher/
```

因此不会覆盖第六轮 tuned PatchTST 模型和历史结果。

### 4.6 CNN ramp expert 结构消融

新增独立入口：

```text
wind_dl_ramp_expert_ablation_train.py
```

该入口只读复用前五轮 `revin_balanced_loss` 的验证指标，不提供父模型
重训开关；也不使用补充风场预训练、多 seed、SWA、滚动验证、第六轮 NRMSE
checkpoint / baseline 回退或验证集加权集成。三个新候选固定使用 seed
`2026`，并分别保存到独立模型命名空间，因此不会覆盖 `tuned_patchtst`
或第六轮产物。

## 5. 原生 PatchTST 的保存产物

每个场站保存：

```text
./models/patchtst_farm_<farm_id>.keras
./wind_results/patchtst/patchtst_farm_<farm_id>_best.weights.h5
./wind_results/patchtst/patchtst_farm_<farm_id>_preprocess.pkl
./wind_results/patchtst/patchtst_tail_farm_<farm_id>.csv
./wind_results/patchtst/history/
./wind_results/patchtst/tensorboard/
./wind_results/patchtst/patchtst_training_metrics.csv
```

完整模型保存为 `.keras`，checkpoint 仅保存 `.weights.h5`。

曾出现的 Keras 错误：

```text
ValueError: The following argument(s) are not supported with the native
Keras format: ['options']
```

处理原则：

- 原生 `.keras` 格式调用 `model.save(path)`，不传不支持的 `options`。
- `ModelCheckpoint` 使用 `save_weights_only=True`。

## 6. 统一预测与可视化

输出目录：

```text
./wind_results/<model_name>/testdata_predict_output/
```

每个场站保存：

- 所有窗口、所有 horizon 的预测长表。
- 总体和逐 horizon 指标。
- 单个 4 小时窗口的 16 点预测/实测 CSV 和图片。
- 完整测试集指数加权曲线 CSV 和图片。
- 模型和场站汇总指标。

完整时间轴聚合公式：

```text
weight(h) = 0.5 ** ((horizon_step - 1) / 4)
```

同一目标时刻可能被多个重叠窗口预测：

- 预测距离越近，权重越高。
- 预测距离越远，权重越低。
- 半衰期为 4 个步长，即 1 小时。

只预测 tuned PatchTST：

```bash
WIND_DL_MODEL_NAMES=tuned_patchtst \
  /home/samlai/anaconda3/envs/deeplearning/bin/python wind_dl_model_predict.py
```

只预测补充 teacher 实验：

```bash
WIND_DL_MODEL_NAMES=tuned_patchtst_external_teacher \
  /home/samlai/anaconda3/envs/deeplearning/bin/python wind_dl_model_predict.py
```

预测全部模型：

```bash
/home/samlai/anaconda3/envs/deeplearning/bin/python wind_dl_model_predict.py
```

## 7. tuned PatchTST 结构

### 7.1 与原生 PatchTST 的区别

tuned 消融中的 `baseline` 不是 `wind_dl_model_train.py` 的原生 PatchTST。

tuned baseline 额外包含：

- Gaussian input noise。
- channel dropout。
- persistence baseline。
- 零初始化 residual forecast head。
- AdamW 和 cosine decay。
- gradient clipping。
- mixed precision。
- 两阶段训练和自蒸馏。
- 功率物理边界及平滑约束。

### 7.2 Persistence residual

```text
forecast = repeat(last_observed_power) + learned_residual
```

残差 head 零初始化，使模型初始预测等价于 persistence forecast。

该先验适合功率连续性较强的场站，但在快速爬坡、骤降或持续性弱的场站可能产生偏置。

### 7.3 Power RevIN

只对历史功率通道执行窗口级可逆归一化：

- 每个窗口计算功率均值和标准差。
- 模型在归一化功率空间预测。
- 输出恢复到原窗口尺度。

优势是缓解功率分布漂移；风险是移除绝对功率水平、趋势以及功率和气象变量之间的尺度关系。

### 7.4 零初始化 CNN Adapter

选择功率和多高度风速通道，通过 separable Conv1D 提取局部模式，再以零初始化 gate 接入主 head。

该模块在部分场站改善 NRMSE，但综合评分不稳定，因此没有进入此前最佳组合。

### 7.5 Loss

`TunedPatchTSTLoss`：

- horizon 衰减 Huber loss。
- 置信度蒸馏。
- 功率上下界 penalty。
- 输出平滑 penalty。

`BalancedTunedPatchTSTLoss`：

- Huber supervised/distillation。
- ramp loss。
- 容量归一化相对误差。
- 物理边界 penalty。
- 不再对远端 horizon 过度降权。

`RMSEBalancedTunedPatchTSTLoss`：

- 在 balanced loss 上增加 MSE。
- 增强远端 horizon 权重。

## 8. DeepSeek 论文启发的训练技巧

参考 DeepSeek-R1 的蒸馏和训练稳定性思想，但没有使用大规模强化学习或超大教师模型。

本项目采用：

1. Cold-start 纯监督训练。
2. Cold-start 最优模型生成 teacher prediction。
3. 按 teacher 样本 MAE 过滤，只保留较可信的 70% teacher 样本。
4. 第二阶段同时使用真实标签和蒸馏目标。
5. AdamW、cosine decay、gradient clipping。
6. Gaussian noise、channel dropout、mixed precision。
7. 风电功率边界和 ramp 约束。

当前 teacher 是同一个模型的早期版本，不是外部强教师。因此蒸馏可能复制 teacher 偏差，并偏向容易样本。

## 9. 前五轮消融实验

综合评分越低越好，训练阶段主要由以下部分组成：

```text
0.45 * NRMSE
+ 0.30 * normalized MAE
+ 0.15 * stable SMAPE
+ 0.10 * R2 penalty
```

窗口级评分占 70%，指数加权曲线评分占 30%。

### 9.1 第一轮

顺序：

```text
baseline
+ RevIN
+ zero-init CNN Adapter
+ balanced loss
+ SWA
```

| Variant | 五场站平均综合评分 | 综合评分改善场站 | NRMSE 改善场站 |
| --- | ---: | ---: | ---: |
| baseline | 0.137562 | - | - |
| revin | 0.132582 | 5/5 | 0/5 |
| revin_cnn_adapter | 0.133029 | 2/5 | 4/5 |
| revin_cnn_adapter_balanced_loss | 0.129835 | 5/5 | 0/5 |
| revin_cnn_adapter_balanced_loss_swa | 0.129775 | 2/5 | 1/5 |

结论：

- RevIN 和 balanced loss 改善综合评分，但没有稳定改善 NRMSE。
- CNN Adapter 更偏向改善 NRMSE，但综合指标不稳定。
- SWA 增益很小。

### 9.2 第二轮

```text
revin_balanced_loss
```

- 移除 CNN Adapter。
- 相对 `revin`，综合评分 5/5 改善。
- 平均综合评分 0.129771。
- NRMSE 仍未稳定改善。

### 9.3 第三轮

```text
revin_balanced_loss_multiseed
```

随机种子：

```text
2026, 2027, 2028
```

结果：

- 综合评分 5/5 改善。
- NRMSE 5/5 改善。
- 平均综合评分 0.129402。
- 是前五轮中唯一同时稳定改善综合评分和 NRMSE 的模块。
- 此前被五个场站全部选中。

### 9.4 第四轮

```text
revin_rmse_balanced_loss
```

- 综合评分仅 1/5 场站改善。
- NRMSE 仅 2/5 改善。
- 平均综合评分 0.130180。
- 不优于第三轮。

### 9.5 第五轮

```text
revin_rmse_balanced_loss_no_distill
```

- 综合评分 0/5 改善。
- NRMSE 仅 2/5 改善。
- 平均综合评分 0.130350。
- 纯监督对照没有优于其父模型。

### 9.6 前五轮保留结论

建议保留：

- persistence residual。
- 输入噪声和 channel dropout。
- RevIN。
- balanced loss。
- 两阶段置信度自蒸馏。
- 多随机种子。
- 等权集成或验证集选择。

暂不作为默认：

- CNN Adapter。
- SWA。
- RMSE-balanced loss。
- 关闭蒸馏。

负收益模块不必从研究代码删除。SCI 消融不要求所有尝试都产生正增益，关键是公平实验、完整证据和与结论一致。

## 10. 已完成的第六轮历史实验

实验组合：

```text
滚动验证
+ NRMSE checkpoint
+ baseline 回退
+ 验证集加权集成
```

这些新增内容没有修改单个 PatchTST 成员的网络主干：

| 模块 | 类型 | 是否为网络结构 |
| --- | --- | --- |
| 滚动验证 | 时间序列验证协议 | 否 |
| NRMSE checkpoint | 参数选择策略 | 否 |
| baseline 回退 | 部署决策策略 | 否 |
| 验证集加权集成 | 输出层决策融合 | 不是 PatchTST 主干结构 |

### 10.1 滚动验证

- 3 个 expanding-window folds。
- 训练和验证之间 purge 15 个窗口。
- 每个 fold 的 scaler 只在训练前缀拟合。
- 每个 fold 训练三个 seed，共 9 个成员/场站。

### 10.2 NRMSE checkpoint

- 物理尺度 RMSE 按装机容量归一化。
- EarlyStopping 和 ModelCheckpoint 监控 `val_actual_nrmse`。

### 10.3 验证集加权集成

- SLSQP 优化三个 seed 权重。
- 权重非负且和为 1。
- 目标包含平均 fold 误差、最差 fold 和 L2 正则。
- 在优化权重、等权和单 seed 中选择。

实际验证中，优化权重相对等权的平均 NRMSE 改善仅约 `0～4.6e-5`，接近验证噪声，未证明非均匀权重具有稳定价值。

### 10.4 Baseline 回退

候选必须同时满足：

```text
candidate NRMSE <= baseline NRMSE
candidate composite score <= baseline composite score
```

否则部署 tuned baseline。

最终：

- `...5880`：baseline fallback。
- `...5895`：滚动加权候选。
- `...5971`：baseline fallback。
- `...5975`：滚动加权候选。
- `...6015`：baseline fallback。

## 11. 第六轮最终测试结果

最终测试汇总：

```text
./wind_results/tuned_patchtst/testdata_predict_output/
  tuned_patchtst_test_metrics_summary.csv
```

分 horizon：

```text
./wind_results/tuned_patchtst/testdata_predict_output/
  tuned_patchtst_test_metrics_by_horizon_all.csv
```

相对第三轮此前最佳 tuned PatchTST：

| 指标 | 五场站平均变化 | 结论 |
| --- | ---: | --- |
| RMSE | -2.05% | 改善 |
| NRMSE | -0.75% | 改善 |
| SMAPE | -0.24% | 小幅改善 |
| R2 | +0.0021 | 改善 |
| MAE | +3.19% | 退化 |
| NMAE | +0.93% | 退化 |
| 完整曲线 RMSE | -1.73% | 改善 |
| 完整曲线 NRMSE | +0.16% | 轻微退化 |
| 完整曲线 MAE | +3.19% | 退化 |

场站 RMSE：

| Farm | 第三轮 tuned | 第六轮 | 变化 | 第六轮策略 |
| --- | ---: | ---: | ---: | --- |
| `...5880` | 9.9154 | 9.6893 | -2.28% | baseline fallback |
| `...5895` | 1.6623 | 1.6742 | +0.71% | weighted/equal ensemble |
| `...5971` | 5.5079 | 5.4780 | -0.54% | baseline fallback |
| `...5975` | 7.2491 | 7.2277 | -0.30% | weighted ensemble |
| `...6015` | 48.1448 | 46.9264 | -2.53% | baseline fallback |

分 horizon 结果：

- RMSE 在 16 个 horizon 全部改善。
- 改善幅度从第 1 步约 0.27% 增长到第 16 步约 2.67%。
- MAE 在 16 个 horizon 全部退化。

说明 NRMSE checkpoint 更重视大误差和远期误差，但牺牲了一般样本的绝对误差。

### 11.1 第六轮模块取舍

建议保留：

1. 滚动验证，作为稳健时间序列验证协议。
2. NRMSE checkpoint，但应与 MAE/综合评分 checkpoint 形成 Pareto 选择。
3. 回退机制，但回退对象应改为该场站此前最佳 champion，而不是固定 tuned baseline。

不建议默认启用：

1. 非均匀验证集权重。
2. 仅按极小验证差异决定复杂权重。

推荐策略：

```text
滚动验证
+ NRMSE/综合评分双 checkpoint
+ 回退到场站历史最佳 champion
+ 多 seed 等权集成
```

第六轮四项是组合实验，不能据此分别宣称每个模块具有独立正增益。若用于论文，需要逐项单变量消融。

## 12. 原生 PatchTST、tuned PatchTST 与 CNN-LSTM

使用现有测试结果：

| Farm | 原生 PatchTST RMSE | CNN-LSTM RMSE | 第六轮 tuned RMSE | 最优 |
| --- | ---: | ---: | ---: | --- |
| `...5880` | 9.3514 | 9.6257 | 9.6893 | PatchTST |
| `...5895` | 1.6380 | 1.7002 | 1.6742 | PatchTST |
| `...5971` | 5.5710 | 5.6035 | 5.4780 | tuned |
| `...5975` | 6.6808 | 6.8170 | 7.2277 | PatchTST |
| `...6015` | 54.6712 | 62.2907 | 46.9264 | tuned |

五场站平均：

| Model | RMSE | NRMSE | MAE | R2 |
| --- | ---: | ---: | ---: | ---: |
| CNN-LSTM | 17.2074 | 0.12458 | 12.8382 | 0.83836 |
| 原生 PatchTST | 15.5825 | 0.11973 | 10.4904 | 0.84780 |
| 第六轮 tuned | 14.1991 | 0.12022 | 8.6869 | 0.84059 |

解释：

- tuned 平均原始 RMSE 更低，主要受高容量场站 `...6015` 大幅改善影响。
- 平均 NRMSE 仍略差于原生 PatchTST，说明跨场站稳定性没有全面超过原生模型。
- `...5880`、`...5975` 的短时局部模式更适合简单模型或原生 PatchTST。
- CNN-LSTM 的卷积和递归归纳偏置可直接捕捉局部爬坡和顺序趋势。
- 当前 `patch_len=16` 覆盖 4 小时，可能压缩 15 分钟级局部变化。
- Persistence residual 在快速变化工况下可能引入错误先验。
- RevIN 可能移除有用绝对尺度。
- 自蒸馏 teacher 不够强，可能复制自身偏差。
- 多个 loss 项之间可能发生目标冲突。
- 所有场站共用同一超参数，无法适配地形、容量、风切变和功率持续性差异。

## 13. 模型保存和加载约定

每个可复现模型至少保留：

1. `.keras` 完整模型。
2. `.weights.h5` 最佳权重。
3. `*_preprocess.pkl`：
   - input/feature columns。
   - scaler_x/scaler_y。
   - capacity。
   - target index。
   - history/forecast 长度。
   - 模型与权重路径。
   - tuned variant 和 ensemble 元数据。
4. 多 seed 模型对应的全部成员文件。
5. 训练 history、TensorBoard 和指标 CSV。

其它模型通常保存到：

```text
./wind_results/<model_name>/
  models/
  weights/
  preprocess/
  history/
  tensorboard/
  tails/
```

加载 tuned、Informer 和 Autoformer 时必须注册自定义 layer、loss 和 metric。统一预测代码中的 `get_custom_objects()` 负责该工作。

## 14. 外部风电数据 teacher 预训练

### 14.1 价值预估

JSFD001～JSFD014 均能提供多高度风速/风向、历史功率和部分温湿压信息。
清洗后每站拥有 `11,143～64,940` 个连续 112 点有效窗口，因此有足够数据
学习跨场站共有的持续性、ramp 和风速—功率动态。

预期有中等提升潜力，但不能在正式目标场站验证/测试完成前宣称已提升：

- 正向因素：约两年、多场站、15 分钟功率和多高度风速，适合增强 teacher
  的通用时序表示。
- 负向因素：来源、容量、气候和传感器分布不同；部分场站存在负功率、
  气象零值、缺高度和限电/故障记录。
- 短期功率预测高度依赖本场站历史功率，外部气象关系可能产生负迁移。

因此采用受控迁移和验证回退，而不是直接拼接。

### 14.2 预处理

`wind_supplementary_preprocess.py`：

1. 只读取 Excel 中有效时间行，跳过百万行格式空尾。
2. 映射不同中文字段名。
3. 功率和气象统一至 15 分钟；风向采用圆周平均。
4. 使用功率高分位数和运行记录稳健估算容量。
5. 功率转换前裁剪到物理边界。
6. 排除运行记录覆盖时段和功率—风速明显矛盾点。
7. 缺高度使用短时插值、跨高度和站内统计量补齐。
8. 仅允许历史 96 点和未来 16 点全部通过质量掩码的窗口。
9. 保存不依赖 TensorFlow/sklearn 版本的压缩 NPZ。

当前报告：

```text
./wind_split/supplementary_other_wind_data/processed_npz/
  supplementary_preprocess_report.csv
  JSFD001_15min.npz
  ...
  JSFD014_15min.npz
```

预处理环境需有 `openpyxl`。当前 base Python 可执行：

```bash
python wind_supplementary_preprocess.py --force
```

### 14.3 Teacher 训练协议

`wind_dl_external_teacher_train.py` 使用：

```text
外部场站功率 per-unit
→ 映射至目标场站容量和 scaler 空间
→ 每站等量抽取 8,192 个窗口
→ 3 epoch 外部有监督 teacher 预训练
→ 目标场站 cold-start 微调
→ 70% 置信 teacher 样本自蒸馏
→ 仅训练 seed=2026
→ 不构建 seed 集成
```

补充数据只提供初始化；teacher checkpoint 仍由目标场站验证集选择。
本轮以验证代码和模型结构为主，第三轮父模型指标只作为对照记录，不执行
父模型回退，避免最终预测实际加载旧结构。待结构和数据链路验证完成后，
再通过显式实验恢复多 seed、集成或 champion fallback。

正式训练默认关闭。执行：

```bash
WIND_EXTERNAL_TEACHER_ENABLE_TRAINING=1 \
  /home/samlai/anaconda3/envs/deeplearning/bin/python \
  wind_dl_external_teacher_train.py
```

在得到测试结果前，该实验只能称为“补充数据 teacher 候选”，不能写成已
证明提升。其它基线模型同样可以使用外部预训练，并非 tuned PatchTST 独有。

## 15. 建议的模型结构创新方向

第六轮主要是训练和部署技巧，不属于 PatchTST 主干结构创新。

更适合后续 SCI 主要创新的方向：

```text
短尺度 CNN ramp expert
+ 多尺度 PatchTST long-context expert
+ persistence expert
+ 工况和不确定性驱动的可学习 gating
```

可进一步加入：

- 多尺度 patch，例如 4/8/16 点并行分支。
- 多高度风速、风向和风切变物理交互编码。
- adaptive RevIN，而不是所有窗口固定启用。
- 场站 embedding 或轻量 station adapter。
- 不确定性输出和可学习回退。
- 有未来数值天气预报时，引入未来气象 decoder。

滚动验证、双 checkpoint 和 champion fallback 可以作为新结构的稳健训练框架，而不是论文唯一创新。

### 15.1 已实现的 CNN ramp expert 消融

这不是先前 CNN Adapter 的重复实验。旧 Adapter 对完整历史做普通卷积后
使用全局平均池化，只生成一个经标量门控注入共享 head 的静态表示；新的
ramp expert 则保留最近 32 点的时间位置，使用 dilation `1/2/4/8` 的 causal
CNN，直接预测未来 16 步功率增量，再累加为完整轨迹。

本轮按以下增量链比较：

```text
A  revin_balanced_loss（前五轮完成结果，只读复用）
B  A + causal/dilated CNN ramp trajectory residual
C  B + PatchTST/ramp 的样本级、逐 horizon gating
D  C + persistence 第三 expert 的逐 horizon gating
```

为隔离结构变量，B～D 保留 A 已有的 RevIN、balanced loss 和自蒸馏训练
协议；它们不是本轮新增贡献。本轮不再叠加新的训练或部署 trick。

三种新候选分别用于预测：

```text
tuned_patchtst_ramp_trajectory
tuned_patchtst_ramp_gated
tuned_patchtst_ramp_persistence_gated
```

总训练开关默认关闭。训练缺失的新结构：

```bash
WIND_RAMP_EXPERT_ENABLE_TRAINING=1 \
  /home/samlai/anaconda3/envs/deeplearning/bin/python \
  wind_dl_ramp_expert_ablation_train.py
```

候选级开关为：

```text
WIND_RAMP_EXPERT_TRAIN_TRAJECTORY
WIND_RAMP_EXPERT_TRAIN_GATED
WIND_RAMP_EXPERT_TRAIN_PERSISTENCE_GATED
```

已完成候选会在检查这些开关之前优先复用；因此完成后可将相应候选开关设为
`0`，不会重复训练。`WIND_RAMP_EXPERT_REUSE_COMPLETED=1` 为默认值。

只预测三个结构候选：

```bash
WIND_DL_MODEL_NAMES=tuned_patchtst_ramp_trajectory,tuned_patchtst_ramp_gated,tuned_patchtst_ramp_persistence_gated \
  /home/samlai/anaconda3/envs/deeplearning/bin/python \
  wind_dl_model_predict.py
```

该实验可以解释 B、C、D 相对各自直接父项的增量差异；正式结论仍必须基于
五个场站统一验证和测试结果，不能仅凭模型结构设计宣称提升。

## 16. 关键结果文件

前五轮消融：

```text
./wind_results/tuned_patchtst/ablation/
  tuned_patchtst_ablation_metrics_all_farms.csv
  tuned_patchtst_ablation_module_summary.csv
```

当前合并明细曾由第六轮代码追加 round 6。读取前五轮时必须显式筛选
`round <= 5`，不能把整个合并 CSV 直接当作前五轮。

第六轮验证实验：

```text
./wind_results/tuned_patchtst/ablation/
  tuned_patchtst_ablation_round6_metrics_all_farms.csv
  tuned_patchtst_ablation_round6_module_summary.csv
```

第六轮测试结果：

```text
./wind_results/tuned_patchtst/testdata_predict_output/
  tuned_patchtst_test_metrics_summary.csv
  tuned_patchtst_test_metrics_by_horizon_all.csv
```

补充数据预处理和 external-teacher 候选：

```text
./wind_split/supplementary_other_wind_data/processed_npz/
  supplementary_preprocess_report.csv

./wind_results/tuned_patchtst_external_teacher/
  tuned_patchtst_external_teacher_candidate_metrics.csv
  tuned_patchtst_external_teacher_training_metrics.csv
  preprocess/
  selection/
  testdata_predict_output/
```

CNN ramp expert 结构消融：

```text
./wind_results/ramp_expert_ablation/
  ramp_expert_ablation_metrics_all_farms.csv
  ramp_expert_ablation_summary.csv
  ramp_expert_promoted_models.csv

./wind_results/tuned_patchtst_ramp_trajectory/
./wind_results/tuned_patchtst_ramp_gated/
./wind_results/tuned_patchtst_ramp_persistence_gated/
```

此前全部深度学习模型比较：

```text
./wind_results/wind_dl_all_models_test_metrics_summary.csv
./wind_results/wind_dl_all_models_test_metrics_by_horizon_all.csv
```

原生 PatchTST 历史测试：

```text
./wind_results/patchtst/第2轮训练结果/testdata_predict_output/
  patchtst_test_metrics_summary.csv
```

注意：

- `wind_dl_all_models_test_metrics_summary.csv` 中的 tuned 结果对应第六轮之前。
- 第六轮最终 tuned 结果应以模型专属的 `tuned_patchtst_test_metrics_summary.csv` 为准。
- 做论文横向比较时，应统一重新预测全部模型，避免混用不同代码阶段的结果。

## 17. 参考项目与论文

### PatchTST

- GitHub：<https://github.com/PatchTST/PatchTST>
- 论文：*A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*
- arXiv：<https://arxiv.org/abs/2211.14730>

### Informer

- GitHub：<https://github.com/zhouhaoyi/Informer2020>
- 论文：*Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*
- arXiv：<https://arxiv.org/abs/2012.07436>

### Autoformer

- GitHub：<https://github.com/thuml/Autoformer>
- 论文：*Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*
- arXiv：<https://arxiv.org/abs/2106.13008>

### DeepSeek-R1

- 论文：*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- arXiv：<https://arxiv.org/abs/2501.12948>
- 本地补充版：

```text
/mnt/f/05-个人校外文件/【待撰写】2026数字中国创新大赛/
参考论文/2501.12948v2-nature论文补充版.pdf
```

## 18. 新对话建议开场文本

```text
请先阅读 docs/WIND_FORECASTING_PROJECT_CONTEXT.md，并检查
wind_dl_model_train.py、wind_dl_other_models_train.py、
wind_dl_tuned_patchtst_train.py 和 wind_dl_model_predict.py。

当前活动代码保留 tuned PatchTST 前五轮消融和等权多 seed 预测；
第六轮“滚动验证 + NRMSE checkpoint + baseline 回退 +
验证集加权集成”已完成实验但不在活动代码中，其模型和结果仍保留。

继续工作时请区分：
1. 当前活动代码；
2. 前五轮消融结果；
3. 第六轮历史实验结果；
4. 原生 PatchTST 和其它基线测试结果。

不要把第六轮组合实验直接解释为四个模块各自独立有效。
```
