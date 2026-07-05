# 机器学习课程项目 PDF 验收文档

来源：仅读取 `ml_exam.pdf`，共 3 页。本文档只整理 PDF 中明确出现的要求，不依据仓库现状补充要求。

## PDF 要求总览

- 项目名称：2026 年专硕机器学习课程项目。（第 1 页）
- 提交截止时间：2026 年 7 月 15 日中午 12 点之前；晚于此时间视为没交。（第 1 页）
- 提交链接：https://docs.qq.com/form/page/DT3pqV3pNcGV6TG1z （第 1 页）
- 合作方式：最多 2 人组队；报告需详细列明作者贡献和各自所属研究领域；借鉴其他团队或网上作者内容必须在参考文献中明确标注，否则视为剽窃。（第 1 页）
- 任务主题：家庭电力消耗多变量时间序列预测，预测未来每一天的总有功功率。（第 1-2 页）
- 主要数据：UCI Machine Learning Repository 的 `Individual household electric power consumption` 数据集；法国一户家庭，2006 年 12 月至 2010 年 11 月，分钟级记录。（第 1 页）
- 可融合外部天气因素；天气信息来自 data.gouv.fr 的月度基础气候数据。（第 1 页）
- 预测设置：基于过去 90 天数据曲线，分别预测未来 90 天短期曲线和未来 365 天长期曲线；短期、长期需分别训练，长期模型参数不能用于短期预测。（第 2 页）
- 方法设置：LSTM、Transformer、自行提出的改进模型三部分。（第 2 页）
- 训练与测试：数据集主要分为 train 和 test 两部分，PDF 原文写作 “train.csv” 和 “tes.csv”。（第 2 页）
- 指标：MSE 和 MAE；至少五轮实验；报告平均值和标准差 std。（第 2 页）
- 报告结构：问题介绍、模型、结果与分析、讨论四部分；同时提交代码并给出 Github 链接。（第 3 页）
- 图表要求：结果截图贴入报告，并绘制 power 预测与 Ground Truth 曲线对比图；注意比较三种方法。（第 3 页）
- 参考文献与工具披露：务必注明参考文献；允许使用 ChatGPT、DEEPSEEK 一类工具撰写报告，但仅限撰写部分，并需注明。（第 3 页）

## 验收清单

### 1. 提交与合作要求

- [ ] 在截止时间 2026 年 7 月 15 日中午 12 点之前提交。（第 1 页）
- [ ] 通过 PDF 给出的提交链接提交。（第 1 页）
- [ ] 如组队，团队人数不超过 2 人。（第 1 页）
- [ ] 报告中详细列明各位作者的贡献。（第 1 页）
- [ ] 报告中注明各位作者各自所属的研究领域。（第 1 页）
- [ ] 如借鉴其他团队、网上作者或外部内容，在参考文献中明确标注。（第 1 页）

### 2. 数据来源与变量要求

- [ ] 使用 UCI `Individual household electric power consumption` 数据集。（第 1 页）
- [ ] 数据来源链接对应 PDF 给出的 UCI 地址：https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption （第 1 页）
- [ ] 数据背景说明包含：采集自法国一户家庭。（第 1 页）
- [ ] 数据时间跨度说明包含：2006 年 12 月到 2010 年 11 月。（第 1 页）
- [ ] 数据粒度说明包含：每一分钟一条记录。（第 1 页）
- [ ] 输入变量覆盖或说明全屋有功功率、无功功率、电流、电压、各路子表能耗等多变量信息。（第 1 页）
- [ ] 通常以每天为单位对原始数据进行汇总。（第 1 页）
- [ ] 可融合天气等外部因素作为输入变量，构建多变量时间序列预测模型。（第 1 页）
- [ ] 天气数据以月为基础汇总，然后提取并添加相应信息。（第 1 页）
- [ ] 天气信息来源对应 PDF 给出的 data.gouv.fr 地址：https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles （第 1 页）
- [ ] 数据字段包含或说明 `global_active_power`：全局有功功率，单位 kW。（第 1 页）
- [ ] 数据字段包含或说明 `global_reactive_power`：全局无功功率，单位 kW。（第 1 页）
- [ ] 数据字段包含或说明 `voltage`：平均电压，单位 V。（第 1 页）
- [ ] 数据字段包含或说明 `global_intensity`：平均电流强度，单位 A。（第 1 页）
- [ ] 数据字段包含或说明 `sub_metering_1`：厨房区域有功能量消耗，单位 Wh。（第 1 页）
- [ ] 数据字段包含或说明 `sub_metering_2`：洗衣房区域有功能量消耗，单位 Wh。（第 2 页）
- [ ] 数据字段包含或说明 `Sub_metering_3`：气候控制系统有功能量消耗，单位 Wh。（第 2 页）
- [ ] 天气字段包含或说明 `RR`：月累计降水高度，单位为毫米的十分之一，记录值需除以 10。（第 2 页）
- [ ] 天气字段包含或说明 `NBJRR1`、`NBJRR5`、`NBJRR10`：当月日降水 >= 1/5/10 mm 的天数。（第 2 页）
- [ ] 天气字段包含或说明 `NBJBROU`：当月雾出现的天数。（第 2 页）
- [ ] 根据 PDF 提示计算 `sub_metering_remainder = (global_active_power * 1000 / 60) - (sub_metering_1 + sub_metering_2 + sub_metering_3)`。（第 2 页）
- [ ] 注意并处理数据项缺失；PDF 说明真实数据有缺失是正常现象。（第 3 页）
- [ ] 注意数据基本时间单位为分钟，需要自行处理。（第 3 页）

### 3. 数据处理要求

- [ ] `global_active_power` 按天取总和。（第 2 页）
- [ ] `global_reactive_power` 按天取总和。（第 2 页）
- [ ] `sub_metering_1` 按天取总和。（第 2 页）
- [ ] `sub_metering_2` 按天取总和。（第 2 页）
- [ ] `voltage` 按天取平均。（第 2 页）
- [ ] `global_intensity` 按天取平均。（第 2 页）
- [ ] `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU` 取当天的任意一个数据。（第 2 页）
- [ ] 一个 Sample 的大小为 input + output。（第 3 页）

### 4. 预测任务要求

- [ ] 预测问题表述为：根据最近的电力消耗情况，预测接下来的预期电力消耗。（第 2 页）
- [ ] 预测目标是接下来每一天的总有功功率。（第 2 页）
- [ ] 基于过去 90 天的数据曲线作为输入。（第 2 页）
- [ ] 完成未来 90 天短期预测。（第 2 页）
- [ ] 完成未来 365 天长期预测。（第 2 页）
- [ ] 短期预测和长期预测需要分别训练。（第 2 页）
- [ ] 长期预测的模型参数不能用于短期预测。（第 2 页）

### 5. 模型要求

- [ ] 使用 LSTM 模型进行预测。（第 2 页）
- [ ] 使用 Transformer 模型进行预测。（第 2 页）
- [ ] 使用自己提出的改进模型进行预测。（第 2 页）
- [ ] 改进模型结构不限。（第 2 页）
- [ ] 改进模型可采用但不限于：结合卷积层提取局部特征后接 Transformer 编码，以改进长期依赖建模能力。（第 2 页）

### 6. 训练、测试与评价要求

- [ ] 数据集主要分为 train 和 test 两部分。（第 2 页）
- [ ] 按 PDF 原文所述，具体见文件 “train.csv” 和 “tes.csv”。（第 2 页）
- [ ] 使用均方误差 MSE 进行测试评价。（第 2 页）
- [ ] 使用平均绝对误差 MAE 进行测试评价。（第 2 页）
- [ ] 至少进行五轮实验。（第 2 页）
- [ ] 对实验结果取平均值。（第 2 页）
- [ ] 提供标准差 std，以评估结果稳定性。（第 2 页）

### 7. 报告与代码交付要求

- [ ] 提交实验报告。（第 3 页）
- [ ] 报告包含第 1 部分：问题介绍。（第 3 页）
- [ ] 报告包含第 2 部分：模型。（第 3 页）
- [ ] 模型部分可以包含少量伪代码。（第 3 页）
- [ ] 报告包含第 3 部分：结果与分析。（第 3 页）
- [ ] 报告包含第 4 部分：讨论。（第 3 页）
- [ ] 提交代码。（第 3 页）
- [ ] 必须给出 Github 链接。（第 3 页）
- [ ] 结果需以截图形式贴在报告中。（第 3 页）
- [ ] 绘制电量 power 预测曲线与真实值 Ground Truth 曲线的对比图。（第 3 页）
- [ ] 对 LSTM、Transformer、自行提出的改进模型三种方法进行比较。（第 3 页）
- [ ] 务必注明参考文献。（第 3 页）
- [ ] 如使用 ChatGPT、DEEPSEEK 一类工具撰写报告，需要在报告中注明。（第 3 页）
- [ ] 若使用 ChatGPT、DEEPSEEK 一类工具，仅限于撰写部分。（第 3 页）

## 评分/注意事项（如 PDF 有）

- [ ] 前两部分为基础题，第三部分为开放题，三部分各占总分的三分之一。（第 2 页）
- [ ] 自行提出的改进模型部分以原理的新颖程度为首要评价标准，性能为次要评价标准。（第 2 页）
- [ ] 如果自行提出的方法新颖但性能不佳，只要原因分析有力，同样可以获得较高分数。（第 3 页）
- [ ] 未在参考文献中明确标注借鉴内容，将被视为剽窃。（第 1 页）
- [ ] 未注明参考文献，将视为抄袭。（第 3 页）
- [ ] 晚于 2026 年 7 月 15 日中午 12 点提交，视为没交。（第 1 页）
- [ ] PDF 给出 3 个博客链接，供不清楚窗口大小和步长等概念的同学参考。（第 3 页）

## 待仓库核对项

以下项目需要后续读取仓库代码、输出和报告内容后核对；本 PDF 子任务不执行仓库核对。

- [ ] 仓库是否实现并使用 UCI 家庭电力消耗数据。
- [ ] 仓库是否处理或说明天气数据来源与字段。
- [ ] 仓库是否按 PDF 要求完成日级汇总。
- [ ] 仓库是否计算 `sub_metering_remainder`。
- [ ] 仓库是否处理分钟级数据和缺失值。
- [ ] 仓库是否构造过去 90 天输入窗口。
- [ ] 仓库是否分别完成 90 天和 365 天预测任务。
- [ ] 仓库是否保证短期、长期模型分别训练，参数不混用。
- [ ] 仓库是否实现 LSTM、Transformer、自行提出的改进模型。
- [ ] 仓库是否划分 train/test，或说明与 PDF 中 `train.csv` / `tes.csv` 的对应关系。
- [ ] 仓库是否对每种方法和预测长度至少运行五轮实验。
- [ ] 仓库是否计算 MSE 和 MAE。
- [ ] 仓库是否报告均值和标准差 std。
- [ ] 报告是否包含问题介绍、模型、结果与分析、讨论四部分。
- [ ] 报告是否包含结果截图。
- [ ] 报告是否包含 power 预测与 Ground Truth 对比曲线。
- [ ] 报告是否比较三种方法。
- [ ] 报告是否包含代码 Github 链接。
- [ ] 报告是否注明参考文献。
- [ ] 报告是否披露 ChatGPT、DEEPSEEK 等工具使用情况（如有）。
- [ ] 鎶ュ憡鏄惁鎶湶 ChatGPT銆丏EEPSEEK 绛夊伐鍏蜂娇鐢ㄦ儏鍐碉紙濡傛湁锛夈€?

---

## 仓库现状核对与报告草稿

> 本节由仓库盘点与报告 subagent 追加；依据当前仓库文件、代码和输出核对，不重新读取 `ml_exam.pdf`。核对时间：2026-06-20。

### 1. 仓库现状核对表

| 主要要求 | 当前状态 | 当前证据 | 缺口与下一步 |
|---|---|---|---|
| 使用 UCI Individual Household Electric Power Consumption 数据集 | 已满足 | `src/constants.py` 定义 `UCI_POWER_URL` / `UCI_FALLBACK_URL`；`data/raw/household_power_consumption.zip` 和 `data/raw/household_power_consumption.txt` 已存在；`src/data.py` 的 `download_uci_power()`、`extract_uci_power()`、`prepare_daily_data()` 支持下载、解压、处理。 | 报告中需写清数据来源、采集地点、时间范围、分钟级粒度和缺失值情况。 |
| 将分钟级数据聚合为日级数据 | 已满足 | `src/data.py` 的 `aggregate_raw_to_daily()` 按日期聚合；`data/processed/daily_power.csv` 已生成，共 1433 行日级样本。 | 报告中需说明聚合策略。 |
| `global_active_power`、`global_reactive_power`、`sub_metering_1/2/3` 按天求和 | 已满足 | `src/data.py` 的 `aggregate_raw_to_daily()` 对上述字段累加。 | 无代码缺口；报告需列明。 |
| `voltage`、`global_intensity` 按天取平均 | 已满足 | `src/data.py` 中维护 `voltage_count`、`global_intensity_count` 并在输出时除以计数。 | 无代码缺口；报告需列明。 |
| 计算 `sub_metering_remainder` | 已满足 | `src/data.py` 使用 `(gap * 1000.0 / 60.0) - (sm1 + sm2 + sm3)` 累加；`data/processed/daily_power.csv` 含该列。 | 无代码缺口；报告需列明单位换算。 |
| 处理缺失值 | 部分满足 | `src/data.py` 的 `_parse_float()` 跳过 `?` 或空值；`fill_missing_array()` 用列均值填充 NaN，全 NaN 列用 0。 | 报告需说明处理策略；如要更严谨，可统计缺失比例。 |
| 接入 data.gouv.fr 月度天气数据及字段 `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU` | 部分满足 | `src/constants.py` 有 `WEATHER_URL`、`WEATHER_RAW_CSV_PATH`、`WEATHER_COLUMNS`；`src/data.py` 的 `_read_weather_by_month()` 可读取 `data/raw/weather_monthly.csv`；`daily_power.csv` 含天气列。 | 当前 `data/raw/weather_monthly.csv` 不存在，已处理数据中的天气列为 0 fallback。下一步需实现或手动补充真实天气 CSV，并重新生成 `daily_power.csv`；若来不及，需要在讨论中披露未能接入真实天气。 |
| 基于过去 90 天输入窗口构造样本 | 已满足 | `src/constants.py` 定义 `INPUT_DAYS = 90`；`src/data.py` 的 `build_windows()` 使用 `input_days=INPUT_DAYS`。 | 报告中说明 sample = input window + output horizon。 |
| 分别完成未来 90 天和 365 天预测 | 部分满足 | `src/train.py` 支持 `--horizon`；`outputs/metrics` 中存在 `flow-matching_h90_seed42.csv`、`flow-matching_h365_seed42.csv`、`spectral-patch_h90_seed42.csv`、`spectral-patch_h365_seed42.csv`，以及 smoke 的 90 天结果。 | LSTM 和 Transformer 当前未见 365 天完整输出；也未见所有模型、所有 horizon 的完整实验汇总。下一步运行 3 个主模型在 90/365 两个 horizon 上的完整实验。 |
| 短期和长期模型分别训练，参数不混用 | 部分满足 | `src/train.py` 保存 checkpoint 名称含 `h{horizon}`；已有 `flow-matching_h90_seed42.pt` 和 `flow-matching_h365_seed42.pt` 等独立文件。 | 仍需完整覆盖 LSTM、Transformer、自设计模型的 90/365 两种 horizon，并在报告中说明分别训练。 |
| 实现 LSTM | 已满足 | `src/models.py` 中 `LSTMForecaster`；`src.models.build_model()` 支持 `lstm`；`outputs/metrics/smoke_metrics.csv` 有 `lstm,90,42`。 | 需完整实验与报告描述。 |
| 实现 Transformer | 已满足 | `src/models.py` 中 `TransformerForecaster`；`build_model()` 支持 `transformer`；`outputs/metrics/smoke_metrics.csv` 有 `transformer,90,42`。 | 需完整实验与报告描述。 |
| 实现自行提出的改进模型 | 已满足 | `src/models.py` 中 `FlowMatchingForecaster`，`build_model()` 支持 `flow-matching`；`README.md` 和 `report/model_research.md` 描述自设计模型思路；`outputs/metrics/smoke_metrics.csv` 有 `flow-matching,90,42`。 | 报告中应把 Flow Matching 作为主改进模型；`conv-transformer` 和 `spectral-patch` 可作为备选或消融，不建议和主要求混淆。 |
| 划分 train/test | 已满足 | `src/data.py` 的 `split_and_scale_windows()` 使用顺序 80/20 切分；`src.data.find_local_train_test()` 可检测 `train.csv` / `test.csv` / `tes.csv`。 | 报告需说明当前仓库使用滑窗后按时间顺序切分，而非随机打乱；如课程提供 `train.csv`/`tes.csv`，需确认是否应优先使用。 |
| 训练集统计量用于标准化，避免测试泄露 | 已满足 | `src/data.py` 的 `split_and_scale_windows()` 用 `train_x` 计算 `feature_mean` 和 `feature_std`，再应用到 train/test。 | 报告可作为实验设置的一部分说明。 |
| 使用 MSE 和 MAE 评价 | 已满足 | `src/train.py` 的 `_evaluate_scaled()` 在原始尺度计算 MSE、MAE；`outputs/metrics/smoke_metrics.csv` 包含 `mse`、`mae`。 | 完整实验仍需汇总均值和标准差。 |
| 每种方法和预测长度至少五轮实验 | 未满足 | `src/constants.py` 定义 `FULL_SEEDS = (11, 22, 33, 44, 55)`，但没有发现 `src/run_experiments.py` 或 `full_metrics.csv`。 | 新增完整实验脚本，循环模型、horizon、seed；运行至少 3 x 2 x 5 = 30 个实验。 |
| 报告均值和标准差 | 未满足 | 当前只有 `outputs/metrics/smoke_metrics.csv` 和 `smoke_summary.json`，未见 `outputs/metrics/full_summary.csv` 或 `full_summary.json`。 | 完成五轮实验后按 model+horizon 计算 MSE/MAE 的 mean/std。 |
| 绘制 Prediction vs Ground Truth 曲线 | 部分满足 | `src/train.py` 的 `_plot_prediction()` 保存对比图；`outputs/figures` 下已有 `lstm_h90_seed42.png`、`transformer_h90_seed42.png`、`flow-matching_h90_seed42.png`、`flow-matching_h365_seed42.png` 等。 | 需补齐每个主模型、每个 horizon 的代表性图，并在报告中插入。 |
| 比较 LSTM、Transformer、自设计模型三种方法 | 部分满足 | smoke 指标已包含三种方法在 90 天、seed 42、1 epoch 下的比较。 | smoke 结果不能作为最终结论；需完整实验后比较 mean/std。 |
| 提交实验报告，包含问题介绍、模型、结果与分析、讨论 | 部分满足 | `report/model_research.md` 已有自设计模型调研；本文件追加了报告草稿。 | 仍需生成正式 `report/report.md` 或将本文件扩展成正式报告，并填入最终结果表和图。 |
| 给出代码 Github 链接 | 未满足 | 当前 README 未见 Github 链接。 | 用户创建或提供仓库后，加入报告。 |
| 注明参考文献 | 部分满足 | `report/model_research.md` 列出 Flow Matching、PatchTST、FEDformer 等参考。 | 正式报告还需补 UCI、data.gouv.fr、LSTM、Transformer 参考文献。 |
| 披露 ChatGPT/Codex 等工具使用 | 部分满足 | 本节包含披露草稿。 | 正式报告需保留工具使用披露，并说明工具仅用于代码/报告辅助。 |

### 2. 当前仓库证据摘要

- 代码入口：
  - `src/data.py`：数据下载、解压、日级聚合、天气 CSV 读取、缺失值处理、滑窗构造、顺序 train/test 切分与标准化。
  - `src/models.py`：`LSTMForecaster`、`TransformerForecaster`、`ConvTransformerForecaster`、`SpectralDecompPatchForecaster`、`FlowMatchingForecaster`。
  - `src/train.py`：单模型训练、CLI 参数、自动设备选择、MSE/MAE 评价、checkpoint 保存、预测图生成。
  - `src/run_smoke.py`：CPU 友好的 smoke test，当前只跑 `lstm`、`transformer`、`flow-matching` 的 90 天任务。
- 数据文件：
  - `data/raw/household_power_consumption.zip`
  - `data/raw/household_power_consumption.txt`
  - `data/processed/daily_power.csv`，包含 `date`、电力字段、`sub_metering_remainder` 和天气字段。
- 已有输出：
  - `outputs/metrics/smoke_metrics.csv`
  - `outputs/metrics/smoke_summary.json`
  - `outputs/metrics/flow-matching_h90_seed42.csv`
  - `outputs/metrics/flow-matching_h365_seed42.csv`
  - `outputs/metrics/spectral-patch_h90_seed42.csv`
  - `outputs/metrics/spectral-patch_h365_seed42.csv`
  - `outputs/figures/*.png`
  - `outputs/models/*.pt`
- 当前 smoke 指标仅用于可运行性检查，不应作为最终报告结果：

| model | horizon | seed | epochs | device | MSE | MAE |
|---|---:|---:|---:|---|---:|---:|
| lstm | 90 | 42 | 1 | cpu | 339254.0 | 447.3593 |
| transformer | 90 | 42 | 1 | cpu | 309591.25 | 426.2964 |
| flow-matching | 90 | 42 | 1 | cpu | 1285776.125 | 1002.9897 |

### 3. 主要缺口与下一步建议

1. 补完整实验编排：新增 `src/run_experiments.py`，循环 `lstm`、`transformer`、`flow-matching`，horizon 为 90 和 365，seed 使用 `FULL_SEEDS = (11, 22, 33, 44, 55)`。
2. 保存完整结果：生成 `outputs/metrics/full_metrics.csv`、`outputs/metrics/full_summary.csv`、`outputs/metrics/full_summary.json`，其中 summary 按 `model+horizon` 汇总 MSE/MAE 的 mean/std。
3. 补齐 365 天实验：当前未见 LSTM、Transformer 的 365 天结果，需优先补齐。
4. 增强训练可靠性：加入 validation split 或早停，保存 best checkpoint；保存 loss curve，便于报告分析。
5. 接入真实天气：实现 data.gouv.fr 月度天气资源下载，或手动准备 `data/raw/weather_monthly.csv`，再重新运行 `python -m src.data --force`；若失败，在讨论中如实说明使用零填充天气 fallback。
6. 生成正式报告：可以以本文件的“报告草稿”为基础，另存或整理为 `report/report.md`，补最终结果表、曲线图、Github 链接、参考文献和工具披露。
7. 若 GPU 可用：使用 GPU 环境跑完整实验；记录 `torch.cuda.is_available()` 和 GPU 名称，避免把 CPU smoke 指标误当最终结果。

### 4. 可作为课程报告基础的内容草稿

#### 4.1 问题介绍

本项目研究家庭电力消耗的多变量时间序列预测问题。给定过去 90 天的日级电力消耗及相关特征，目标是预测未来每天的 `global_active_power` 曲线。实验包含两个预测长度：短期预测未来 90 天，长期预测未来 365 天。该任务具有明显的时间依赖、周期性、长期趋势和缺失值处理挑战，因此适合比较循环神经网络、Transformer 以及自设计改进模型在多步预测中的表现。

数据主要来自 UCI Machine Learning Repository 的 Individual Household Electric Power Consumption 数据集。原始记录为法国一户家庭的分钟级用电数据，字段包括全局有功功率、无功功率、电压、电流以及多个分表能耗。当前仓库将分钟级数据聚合为日级数据，并构造过去 90 天输入、未来 horizon 天输出的滑动窗口样本。

#### 4.2 数据说明

当前数据处理流程位于 `src/data.py`。代码首先下载并解压 UCI 原始 zip 文件，然后读取 `household_power_consumption.txt`。对每一天的数据，`global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 使用求和；`voltage` 和 `global_intensity` 使用平均值；同时计算：

```text
sub_metering_remainder = (global_active_power * 1000 / 60) - (sub_metering_1 + sub_metering_2 + sub_metering_3)
```

缺失或非法数值通过 `_parse_float()` 识别，在日级矩阵中使用列均值填充；如果某列全为缺失，则填充为 0。模型训练前，代码按时间顺序划分 train/test，并仅使用训练集统计量进行标准化，以减少测试集信息泄露。

天气数据方面，仓库已经预留 `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU` 五个字段，并支持读取 `data/raw/weather_monthly.csv` 后按月份连接到日级样本。但当前仓库未发现真实天气 CSV，处理结果中的天气字段为 0 fallback。因此正式报告需要说明该限制，或在最终实验前补齐真实天气数据。

#### 4.3 模型描述

**LSTM。** `LSTMForecaster` 使用 LSTM 编码过去 90 天的多变量输入序列，并取最后一层隐状态经过多层感知机直接输出未来 horizon 天的 `global_active_power`。该模型作为经典循环神经网络基线，用于检验序列递归建模能力。

**Transformer。** `TransformerForecaster` 先将输入特征投影到 `d_model` 维空间，加入正弦位置编码，再经过 Transformer Encoder 建模时间维依赖。最后对时间维表示做平均池化，并通过预测头输出多步预测结果。该模型用于比较自注意力机制在长距离依赖建模中的效果。

**自设计改进模型：Flow Matching Forecaster。** `FlowMatchingForecaster` 将未来用电曲线视为条件生成轨迹，而不是单一确定性回归向量。模型先用 patch Transformer 编码历史 90 天信息，生成 trend-guided prior future curve，再通过 rectified-flow / conditional flow matching 形式学习从先验未来曲线到真实未来曲线的 velocity field。推理时从先验曲线出发，经 Euler steps 积分得到预测轨迹，并可对多次采样取平均以计算 MSE/MAE。该设计的创新点在于把多步电力预测建模为条件轨迹生成问题，能够自然表达未来曲线的不确定性，同时仍输出点预测用于课程要求的指标计算。

#### 4.4 实验设置

实验输入窗口长度为 90 天，预测 horizon 分别为 90 天和 365 天。每个模型在每个 horizon 上单独训练，模型参数不跨 horizon 复用。数据按时间顺序划分 train/test，训练时使用 AdamW 优化器和 MSE 类训练目标；评价时将预测值反标准化回原始日级 `global_active_power` 尺度，计算 MSE 和 MAE。

当前仓库已经完成 CPU smoke test：`python -m src.run_smoke`。该测试只用于验证代码链路，包括数据读取、模型前向、训练、评价、checkpoint 和图像输出。最终报告应使用完整实验结果：3 个主模型 x 2 个预测长度 x 5 个随机种子，并报告均值和标准差。

建议正式实验命令形态如下：

```powershell
python -m src.data --force
python -m src.run_experiments --models lstm transformer flow-matching --horizons 90 365 --seeds 11 22 33 44 55
```

注：上述 `src.run_experiments` 当前尚未实现，是下一步建议新增的完整实验入口。

#### 4.5 结果与分析占位

最终结果表应按 `model` 和 `horizon` 分组，报告 MSE mean、MSE std、MAE mean、MAE std。建议表格如下：

| Horizon | Model | MSE mean | MSE std | MAE mean | MAE std |
|---:|---|---:|---:|---:|---:|
| 90 | LSTM | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 |
| 90 | Transformer | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 |
| 90 | Flow Matching | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 |
| 365 | LSTM | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 |
| 365 | Transformer | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 |
| 365 | Flow Matching | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 | 待完整实验填入 |

分析时建议分别讨论：

- 90 天预测中，三种模型对短期趋势和局部波动的拟合能力。
- 365 天预测中，误差随 horizon 变长是否明显上升。
- Transformer 与 LSTM 在长期依赖建模上的差异。
- Flow Matching 模型是否表现出更平滑或更稳定的未来曲线；若指标不优，也应从数据规模、训练轮数、采样步数、模型复杂度等角度解释。
- 天气字段若仍为 0 fallback，需要说明外部变量未充分发挥作用。

需要插入的图包括每个模型、每个 horizon 至少一张 Prediction vs Ground Truth 曲线。当前可用图位于 `outputs/figures/`，但最终应使用完整实验生成的代表性图。

#### 4.6 讨论

本项目的关键挑战包括：原始数据粒度为分钟级，需要聚合为日级；真实数据存在缺失值；用过去 90 天预测未来 365 天时，输入信息相对不足，长期趋势和季节性难以完全捕捉。LSTM 能够建模顺序依赖，但在长 horizon 预测中可能受限于隐状态容量。Transformer 通过自注意力机制增强了长距离依赖建模能力，但在小数据场景中可能需要更充分的正则化和调参。Flow Matching Forecaster 将未来曲线作为条件生成轨迹，具备表达多种可能未来形态的潜力，但训练和采样成本更高，也更依赖完整实验和超参数选择。

当前仓库仍处于 smoke/prototype 状态，已有结果只能证明代码链路可运行，不能直接作为最终课程结论。后续需要在 GPU 环境或更充分训练设置下运行完整实验，并将均值、标准差和可视化曲线纳入报告。

#### 4.7 参考文献占位

正式报告至少应包含以下参考：

- UCI Machine Learning Repository: Individual household electric power consumption dataset.
- data.gouv.fr: Données climatologiques de base mensuelles.
- Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory.
- Vaswani et al. Attention Is All You Need.
- Flow Matching for Generative Modeling.
- Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting 或其他与时间序列 flow matching 相关文献。
- 如报告使用 PatchTST、FEDformer、Conv-Transformer 作为灵感或对比，也应列入参考文献。

#### 4.8 工具使用披露草稿

本项目开发与报告整理过程中使用了 ChatGPT/Codex 作为辅助工具，用于代码结构梳理、实验脚本建议、报告草稿撰写和验收清单核对。最终实验运行、结果解释、图表选择和报告结论由作者检查并负责。AI 工具未被用于伪造实验结果或替代必要的实验验证。
