# 自定义模型调研报告: Flow Matching 用电轨迹生成模型

## 候选方向对比

| 方向 | 代表方法 | 核心思想 | 适配本项目 | 结论 |
|---|---|---|---|---|
| Conv + Transformer | Informer 类变体、局部卷积编码 | 先提局部模式，再做序列编码 | 实现简单，但创新性不足 | 仅作为 baseline |
| Patch / 频域 / 分解 | PatchTST、FEDformer、TimesNet | 将序列切片、分解趋势、建模低频周期 | 适合用电预测，但仍主要是确定性回归 | 可作为辅助思想 |
| Diffusion forecasting | TimeGrad、CSDI 等 | 从噪声逐步去噪生成未来轨迹 | 概率预测能力强，但训练/采样成本偏高 | CPU smoke 不优先 |
| Conditional Flow Matching | Flow Matching、OT-CFM、TSFlow、AFM | 学习从先验轨迹到真实未来轨迹的连续 velocity field | 直接把未来用电曲线建模为条件生成分布，创新性更强 | 推荐采用 |

## 推荐方案

第三个模型实现为 `FlowMatchingForecaster`，命令名为 `flow-matching`。

该模型不再把未来曲线视为单个确定性回归向量，而是建模条件分布:

```text
p(y_future | x_history)
```

其中 `x_history` 是过去 90 天多变量输入，`y_future` 是未来 90 或 365 天 `global_active_power` 曲线。

模型采用 Conditional Flow Matching / Rectified Flow 形式:

1. 使用 patch Transformer 编码过去 90 天历史，得到 context。
2. 用 context 和历史目标列生成 trend-guided prior:
   `z0 = prior_mean(context, history_target) + sigma(context) * eps`。
3. 训练时采样 `t ~ U(0, 1)`，构造直线路径:
   `z_t = (1 - t) z0 + t y`。
4. velocity network 输入 `z_t`、flow time `t`、horizon position 和 context，预测速度场:
   `v_theta(z_t, t, context)`。
5. 训练目标为:
   `MSE(v_theta, y - z0) + lambda * MSE(prior_mean, y)`。
6. 推理时从 `z0` 出发，用 Euler steps 积分到 `t=1`，多次采样后取均值作为点预测。

## 创新点

- 从确定性回归升级为条件生成式预测，可自然表达未来用电轨迹的不确定性。
- Flow Matching 训练阶段不需要反向传播穿过 ODE solver，比 diffusion 类方法更适合当前 CPU smoke 约束。
- trend-guided prior 将时间序列结构注入 source distribution，避免从纯白噪声生成整条用电曲线导致训练不稳定。
- velocity field 学习的是“从先验未来曲线到真实未来曲线的修正方向”，比直接输出未来 365 个点更具建模解释性。
- 采样均值可继续用于课程要求的 MSE/MAE，后续还可以报告预测区间或样本标准差。

## 风险与边界

- smoke 只跑 1 epoch，小样本指标不能代表最终性能。
- 365 天预测从 90 天历史外推，本身信息不足，生成模型也无法消除数据限制。
- Euler steps 和 samples 设得较小是为了 CPU 可运行，完整实验可增大 `FLOW_EULER_STEPS` 和 `FLOW_NUM_SAMPLES`。
- 报告中应表述为“将 Conditional Flow Matching 应用于家庭用电轨迹预测的自设计轻量模型”，不要声称提出了通用新理论。

## 关键参考

- Flow Matching for Generative Modeling. https://arxiv.org/abs/2210.02747
- Improving and generalizing flow-based generative models with minibatch optimal transport. https://arxiv.org/abs/2302.00482
- Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting. https://openreview.net/forum?id=uxVBbSlKQ4
- Probabilistic Forecasting via Autoregressive Flow Matching. https://arxiv.org/html/2503.10375v2
- PatchTST: A Time Series is Worth 64 Words. https://arxiv.org/abs/2211.14730
- FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. https://arxiv.org/abs/2201.12740
