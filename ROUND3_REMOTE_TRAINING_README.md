# Part 3 Round 3 远程训练包使用说明

本压缩包是 JSFD001–JSFD014 补充数据集的最小可运行闭包。请把压缩包
完整解压到远程服务器的一个新目录，并始终在该目录根路径执行命令。不要只
上传三份 Round 3 入口脚本，也不要把数据目录改名或单独移动。

## 1. 已验证环境

- Linux x86_64
- CPython 3.9.25
- TensorFlow 2.14.0 / Keras 2.14.0
- TensorFlow 对应 CUDA 11.8、cuDNN 8.7 用户态库
- NVIDIA 驱动必须兼容 CUDA 11.8

推荐在干净的 Python 3.9.25 环境中安装完整锁文件：

```bash
python -m pip install -r requirements-round3-lock-linux-py39-gpu.txt
python -m pip check
```

`requirements.txt` 是精确锁定的项目级直接依赖；`requirements-round3-lock-
linux-py39-gpu.txt` 还锁定了已验证环境中的传递依赖。不要使用 Python 3.13
或 Keras 3 运行本轮代码。

安装后验证 TensorFlow 与 GPU：

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.test.is_built_with_cuda()); print(tf.config.list_physical_devices('GPU'))"
```

输出应包含 TensorFlow `2.14.0`、`True`，并至少列出一块 GPU。

## 2. 必须保持的目录层级

```text
<解压根目录>/
├── requirements.txt
├── requirements-round3-lock-linux-py39-gpu.txt
├── ROUND3_REMOTE_TRAINING_README.md
├── wind_part3_round3_external14_preprocess.py
├── wind_part3_round3_external14_all_models_train.py
├── wind_part3_round3_external14_all_models_predict.py
├── wind_dl_model_train.py
├── wind_dl_other_models_train.py
├── wind_FeTS_PatchTST_train.py
├── wind_FeTS_PatchTST_min_train.py
├── wind_RegimeEncoder_PatchTST_train.py
├── wind_RegimeEncoder_PatchTST_feature_screen_train.py
└── wind_split/
    └── supplementary_other_wind_data/
        ├── JSFD001/
        ├── ...
        └── JSFD014/
```

数据目录只包含从原始 Excel 出发的 14 个场站数据。旧
`processed_npz/`既不在包内，也不得复制回该路径作为本轮输入。

## 3. 执行顺序

先完成 14 个场站的无泄漏预处理和特征工程：

```bash
python wind_part3_round3_external14_preprocess.py --farms all
```

再运行最重模型的全局单 epoch 显存预检。若某个重模型在
`batch_size=192` 下真实 GPU OOM，代码会为该模型冻结 14 场站统一使用
`batch_size=128` 的策略：

```bash
python wind_part3_round3_external14_all_models_train.py --preflight-only
```

启动 10 模型 × 14 场站正式训练；`--resume`允许 SSH 中断后继续，并只复用
通过身份与哈希校验的已完成任务：

```bash
python wind_part3_round3_external14_all_models_train.py --resume
```

全部 140 个训练任务冻结后，最后一次性执行测试集预测、比较、统计检验和
可视化：

```bash
python wind_part3_round3_external14_all_models_predict.py --resume
```

正式结果会自动保存到：

```text
wind_results/part3_new_module_supplement/
└── 03_external14_leakage_free_strong_baseline_benchmark/
```

该目录无需预先创建，也不应从本地旧实验结果复制进远程训练目录。
