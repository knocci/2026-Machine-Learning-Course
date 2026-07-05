"""Forecasting models for the household power project."""

from __future__ import annotations

import math

import torch
from torch import nn

from .constants import (
    FLOW_EULER_STEPS,
    FLOW_NOISE_MAX,
    FLOW_NOISE_MIN,
    FLOW_NUM_SAMPLES,
    FLOW_PRIOR_LOSS_WEIGHT,
    INPUT_DAYS,
)


class LSTMForecaster(nn.Module):
    """LSTM baseline with a direct multi-step prediction head."""

    def __init__(
        self,
        input_size: int,
        output_days: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_days),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """Transformer encoder baseline."""

    def __init__(
        self,
        input_size: int,
        output_days: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, output_days),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.position(self.projection(x)))
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class ConvTransformerForecaster(nn.Module):
    """Conv1D local pattern extractor followed by a Transformer encoder."""

    def __init__(
        self,
        input_size: int,
        output_days: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
        )
        self.residual = nn.Linear(input_size, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_days),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        encoded_input = local_features + self.residual(x)
        encoded = self.encoder(self.position(encoded_input))
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class RevIN(nn.Module):
    """Reversible instance normalization for non-stationary time series."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False).detach() + self.eps)
        normalized = (x - mean) / std
        return normalized * self.gamma + self.beta, mean, std

    def denormalize_target(
        self,
        y: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        target_index: int = 0,
    ) -> torch.Tensor:
        target_gamma = self.gamma[:, :, target_index].clamp_min(self.eps)
        target_beta = self.beta[:, :, target_index]
        restored = (y - target_beta) / target_gamma
        return restored * std[:, :, target_index] + mean[:, :, target_index]


class SeriesDecomposition(nn.Module):
    """Moving-average trend plus residual seasonal component."""

    def __init__(self, kernel_size: int = 15) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = (self.kernel_size - 1) // 2
        padded = torch.nn.functional.pad(series.unsqueeze(1), (pad, pad), mode="replicate")
        trend = self.pool(padded).squeeze(1)
        residual = series - trend
        return trend, residual


class SpectralDecompPatchForecaster(nn.Module):
    """Decomposition, frequency, and patch-token forecaster.

    The model combines three recent time-series ideas in a CPU-friendly form:
    reversible instance normalization, trend/seasonal decomposition, and
    patch-level Transformer tokens with an FFT residual branch.
    """

    def __init__(
        self,
        input_size: int,
        output_days: int,
        input_days: int = 90,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        patch_len: int = 14,
        stride: int = 7,
        moving_avg: int = 15,
        freq_modes: int = 8,
        target_index: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_days = output_days
        self.input_days = input_days
        self.patch_len = patch_len
        self.stride = stride
        self.freq_modes = freq_modes
        self.target_index = target_index

        augmented_size = input_size + 2
        self.revin = RevIN(input_size)
        self.decomposition = SeriesDecomposition(moving_avg)
        self.patch_projection = nn.Linear(augmented_size * patch_len, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.patch_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.patch_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, output_days),
        )
        self.trend_head = nn.Linear(input_days, output_days)
        self.residual_head = nn.Linear(input_days, output_days)
        self.frequency_head = nn.Sequential(
            nn.Linear(freq_modes * 2, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, output_days),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4),
        )

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < self.patch_len:
            raise ValueError("input sequence is shorter than patch_len")
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.flatten(start_dim=2)

    def _frequency_features(self, residual: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(residual, dim=1)
        available = max(spectrum.size(1) - 1, 0)
        use_modes = min(self.freq_modes, available)
        features = residual.new_zeros((residual.size(0), self.freq_modes * 2))
        if use_modes > 0:
            selected = spectrum[:, 1 : use_modes + 1]
            features[:, :use_modes] = selected.real
            features[:, self.freq_modes : self.freq_modes + use_modes] = selected.imag
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm, mean, std = self.revin.normalize(x)
        target = x_norm[:, :, self.target_index]
        trend, residual = self.decomposition(target)
        augmented = torch.cat((x_norm, trend.unsqueeze(-1), residual.unsqueeze(-1)), dim=-1)

        patch_tokens = self.patch_projection(self._patch_tokens(augmented))
        encoded = self.patch_encoder(self.position(patch_tokens))
        context = encoded.mean(dim=1)

        patch_forecast = self.patch_head(context)
        trend_forecast = self.trend_head(trend)
        residual_forecast = self.residual_head(residual)
        frequency_forecast = self.frequency_head(self._frequency_features(residual))
        weights = torch.softmax(self.gate(context), dim=-1)

        normalized_forecast = (
            weights[:, 0:1] * patch_forecast
            + weights[:, 1:2] * trend_forecast
            + weights[:, 2:3] * residual_forecast
            + weights[:, 3:4] * frequency_forecast
        )
        return self.revin.denormalize_target(normalized_forecast, mean, std, self.target_index)


class FlowMatchingForecaster(nn.Module):
    """Conditional flow-matching forecaster for future power trajectories.

    Training uses a rectified-flow objective in standardized target space. Given
    a history context, the model samples a condition-dependent prior trajectory
    and learns the velocity field that transports it to the observed future
    trajectory. Inference integrates the learned velocity with Euler steps and
    averages multiple generated trajectories for point metrics.
    """

    def __init__(
        self,
        input_size: int,
        output_days: int,
        input_days: int = INPUT_DAYS,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        patch_len: int = 14,
        stride: int = 7,
        prior_loss_weight: float = FLOW_PRIOR_LOSS_WEIGHT,
        noise_min: float = FLOW_NOISE_MIN,
        noise_max: float = FLOW_NOISE_MAX,
        target_index: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_days = output_days
        self.input_days = input_days
        self.patch_len = patch_len
        self.stride = stride
        self.prior_loss_weight = prior_loss_weight
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.target_index = target_index

        self.history_projection = nn.Linear(input_size * patch_len, d_model)
        self.history_position = PositionalEncoding(d_model)
        history_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(history_layer, num_layers=num_layers)
        self.context_norm = nn.LayerNorm(d_model)

        self.trend_prior = nn.Linear(input_days, output_days)
        self.prior_residual = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, output_days),
        )
        self.prior_scale = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, output_days),
        )
        nn.init.zeros_(self.trend_prior.weight)
        nn.init.zeros_(self.trend_prior.bias)
        nn.init.zeros_(self.prior_residual[-1].weight)
        nn.init.zeros_(self.prior_residual[-1].bias)

        self.horizon_position = nn.Parameter(torch.randn(1, output_days, d_model) * 0.02)
        self.trajectory_projection = nn.Linear(1, d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.context_projection = nn.Linear(d_model, d_model)
        velocity_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.velocity_encoder = nn.TransformerEncoder(velocity_layer, num_layers=num_layers)
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < self.patch_len:
            raise ValueError("input sequence is shorter than patch_len")
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.flatten(start_dim=2)

    def encode_history(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.history_projection(self._patch_tokens(x))
        encoded = self.history_encoder(self.history_position(tokens))
        return self.context_norm(encoded.mean(dim=1))

    def prior_parameters(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history_target = x[:, :, self.target_index]
        last_level = history_target[:, -1:].expand(-1, self.output_days)
        mean = last_level + self.trend_prior(history_target) + self.prior_residual(context)
        scale_raw = self.prior_scale(context)
        scale = self.noise_min + (self.noise_max - self.noise_min) * torch.sigmoid(scale_raw)
        return mean, scale

    def velocity(self, z_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        trajectory_tokens = self.trajectory_projection(z_t.unsqueeze(-1))
        time_tokens = self.time_embedding(t).unsqueeze(1)
        context_tokens = self.context_projection(context).unsqueeze(1)
        tokens = trajectory_tokens + self.horizon_position + time_tokens + context_tokens
        encoded = self.velocity_encoder(tokens)
        return self.velocity_head(encoded).squeeze(-1)

    def training_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        context = self.encode_history(x)
        prior_mean, prior_scale = self.prior_parameters(x, context)
        z0 = prior_mean + prior_scale * torch.randn_like(y)
        t = torch.rand(y.size(0), 1, device=y.device, dtype=y.dtype)
        z_t = (1.0 - t) * z0 + t * y
        target_velocity = y - z0
        predicted_velocity = self.velocity(z_t, t, context)
        flow_loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        prior_loss = torch.nn.functional.mse_loss(prior_mean, y)
        return flow_loss + self.prior_loss_weight * prior_loss

    def sample_forecast(
        self,
        x: torch.Tensor,
        num_samples: int = FLOW_NUM_SAMPLES,
        steps: int = FLOW_EULER_STEPS,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        context = self.encode_history(x)
        prior_mean, prior_scale = self.prior_parameters(x, context)

        batch_size = x.size(0)
        repeated_context = context.repeat(num_samples, 1)
        z = prior_mean.repeat(num_samples, 1)
        z = z + prior_scale.repeat(num_samples, 1) * torch.randn_like(z)

        dt = 1.0 / float(steps)
        for step in range(steps):
            t_value = step * dt
            t = z.new_full((z.size(0), 1), t_value)
            z = z + dt * self.velocity(z, t, repeated_context)
        return z.view(num_samples, batch_size, self.output_days).mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample_forecast(x, num_samples=1, steps=FLOW_EULER_STEPS)


class GeGLU(nn.Module):
    """Gated GELU activation for better gradient flow (LLaMA-style)."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class SpectralPatchV2Forecaster(nn.Module):
    """Improved forecaster with channel-split encoding, core token fusion (SOFTS),
    learnable frequency MLP (FreTS), and increased capacity (GeGLU, DropPath)."""

    def __init__(
        self,
        input_size: int,
        output_days: int,
        input_days: int = INPUT_DAYS,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        droppath_prob: float = 0.1,
        patch_len: int = 14,
        stride: int = 7,
        moving_avg: int = 15,
        num_core_tokens: int = 6,
        d_channel: int = 12,
        target_index: int = 0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_days = output_days
        self.input_days = input_days
        self.patch_len = patch_len
        self.stride = stride
        self.target_index = target_index
        self.use_revin = use_revin

        self.revin = RevIN(input_size)
        self.decomposition = SeriesDecomposition(moving_avg)
        augmented_size = input_size + 2

        # --- Channel-Split Encoder ---
        k3_out = d_channel // 3
        k7_out = d_channel // 3
        k15_out = d_channel - k3_out - k7_out
        self.ch_conv_k3 = nn.Conv1d(1, k3_out, kernel_size=3, padding=1)
        self.ch_conv_k7 = nn.Conv1d(1, k7_out, kernel_size=7, padding=3)
        self.ch_conv_k15 = nn.Conv1d(1, k15_out, kernel_size=15, padding=7)
        self.channel_proj = nn.Linear(d_channel, d_model)
        self.channel_norm = nn.LayerNorm(d_model)

        # --- Core Token Fusion (SOFTS-inspired) ---
        self.core_tokens = nn.Parameter(torch.randn(1, num_core_tokens, d_model) * 0.02)
        self.core_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.core_self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.channel_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )

        # --- Temporal Encoder ---
        self.patch_projection = nn.Linear(augmented_size * patch_len, d_model)
        self.position = PositionalEncoding(d_model, max_len=128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Frequency MLP (FreTS-inspired) ---
        n_freq = input_days // 2 + 1
        self.freq_real = nn.Sequential(
            nn.Linear(n_freq, dim_feedforward),
            GeGLU(dim_feedforward, dim_feedforward),
        )
        self.freq_imag = nn.Sequential(
            nn.Linear(n_freq, dim_feedforward),
            GeGLU(dim_feedforward, dim_feedforward),
        )
        self.freq_fusion = nn.Sequential(
            nn.Linear(dim_feedforward * 2, d_model),
            nn.LayerNorm(d_model),
            GeGLU(d_model, d_model),
        )

        # --- Prediction Head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            GeGLU(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_days),
        )

        self.residual_skip = nn.Linear(input_days, output_days)

        self._init_weights()
        self._init_droppath()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_droppath(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                pass
        self._droppath_enabled = True

    def _match_core(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.core_tokens.expand(batch_size, -1, -1).to(device)

    def _channel_encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        B, L, C = x_norm.shape
        x = x_norm.transpose(1, 2).reshape(B * C, 1, L)
        f3 = self.ch_conv_k3(x).mean(dim=-1)
        f7 = self.ch_conv_k7(x).mean(dim=-1)
        f15 = self.ch_conv_k15(x).mean(dim=-1)
        tokens = torch.cat([f3, f7, f15], dim=-1)
        tokens = tokens.reshape(B, C, -1)
        tokens = self.channel_norm(self.channel_proj(tokens))
        return tokens

    def _core_fusion(self, channel_tokens: torch.Tensor) -> torch.Tensor:
        B = channel_tokens.size(0)
        core = self._match_core(B, channel_tokens.device)
        core, _ = self.core_cross_attn(core, channel_tokens, channel_tokens)
        core, _ = self.core_self_attn(core, core, core)
        channel_tokens, _ = self.channel_cross_attn(channel_tokens, core, core)
        return channel_tokens, core.mean(dim=1)

    def _freq_features(self, series: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(series, dim=1)
        real_features = self.freq_real(spectrum.real)
        imag_features = self.freq_imag(spectrum.imag)
        return self.freq_fusion(torch.cat([real_features, imag_features], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_revin:
            x_norm, mean, std = self.revin.normalize(x)
        else:
            x_norm, mean, std = x, None, None
        target = x_norm[:, :, self.target_index]
        trend, residual = self.decomposition(target)

        channel_tokens = self._channel_encode(x_norm)
        enriched_channels, channel_context = self._core_fusion(channel_tokens)

        augmented = torch.cat(
            (x_norm, trend.unsqueeze(-1), residual.unsqueeze(-1)), dim=-1
        )
        patches = augmented.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous().flatten(start_dim=2)
        patch_emb = self.patch_projection(patches)
        patch_emb = patch_emb + channel_context.unsqueeze(1)
        encoded = self.temporal_encoder(self.position(patch_emb))
        temporal_context = encoded.mean(dim=1)

        freq_context = self._freq_features(target)
        fused = temporal_context + freq_context

        nonlinear_forecast = self.head(fused)
        linear_forecast = self.residual_skip(target)
        normalized_forecast = nonlinear_forecast + linear_forecast
        if self.use_revin:
            return self.revin.denormalize_target(
                normalized_forecast, mean, std, self.target_index
            )
        return normalized_forecast


class DLinearPlusForecaster(nn.Module):
    """Multi-scale decomposition + channel-independent linear maps + learnable fusion.

    Inspired by DLinear (AAAI 2023), iTransformer, and SOFTS.
    Uses moving-average decomposition at multiple kernel sizes, applies
    per-channel shared linear projections independently, then fuses via
    learned channel-wise and scale-wise attention.
    """

    def __init__(
        self,
        input_size: int,
        output_days: int,
        input_days: int = INPUT_DAYS,
        moving_avg_kernels: tuple[int, ...] = (3, 7, 15, 25),
        target_index: int = 0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_days = output_days
        self.input_days = input_days
        self.target_index = target_index
        self.ma_kernels = moving_avg_kernels
        self.use_revin = use_revin
        num_scales = len(moving_avg_kernels) + 1

        self.revin = RevIN(input_size)

        self.pools = nn.ModuleList()
        for k in moving_avg_kernels:
            pad = (k - 1) // 2
            self.pools.append(nn.AvgPool1d(kernel_size=k, stride=1, padding=pad))

        self.channel_linears = nn.ModuleList()
        for _ in range(num_scales):
            self.channel_linears.append(nn.Linear(input_days, output_days))

        self.channel_attn_weight = nn.Parameter(torch.ones(1, num_scales, input_size) * 0.1)
        self.scale_fusion = nn.Parameter(torch.ones(1, num_scales) / num_scales)
        self.residual_skip = nn.Linear(input_days, output_days)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _decompose(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: (B, C, L) — per-channel processing
        B, C, L = x.shape
        x_flat = x.reshape(B * C, 1, L)
        components: list[torch.Tensor] = []
        for pool in self.pools:
            trend = pool(x_flat).reshape(B, C, L)
            components.append(trend)
        last_trend = components[-1] if components else None
        residual = x - last_trend if last_trend is not None else x
        components.append(residual)
        return components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if self.use_revin:
            x_norm, mean, std = self.revin.normalize(x)
        else:
            x_norm, mean, std = x, None, None

        components = self._decompose(x_norm.transpose(1, 2))

        channel_weights = torch.softmax(self.channel_attn_weight, dim=-1)
        scale_weights = torch.softmax(self.scale_fusion, dim=-1)

        predictions = torch.zeros(B, self.output_days, device=x.device)
        for s, (comp, linear) in enumerate(zip(components, self.channel_linears)):
            ch_preds = linear(comp.reshape(B * C, L)).reshape(B, C, self.output_days)
            weighted = (ch_preds * channel_weights[:, s, :, None]).sum(dim=-2)
            predictions = predictions + scale_weights[0, s] * weighted

        target = x[:, :, self.target_index]
        predictions = predictions + self.residual_skip(target)
        if self.use_revin:
            return self.revin.denormalize_target(predictions, mean, std, self.target_index)
        return predictions


class ConvTransformerV3Forecaster(nn.Module):
    """Conv-Transformer + SOFTS core tokens + FreTS frequency MLP + GeGLU + residual skip.

    Proven Conv1D all-channel mixing backbone enhanced with core token fusion for
    cross-temporal pattern sharing and learnable frequency features. No RevIN.
    """

    def __init__(
        self,
        input_size: int,
        output_days: int,
        input_days: int = INPUT_DAYS,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        kernel_size: int = 5,
        num_core_tokens: int = 6,
        target_index: int = 0,
    ) -> None:
        super().__init__()
        self.input_days = input_days
        self.target_index = target_index

        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
        )
        self.residual = nn.Linear(input_size, d_model)
        self.position = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.core_tokens = nn.Parameter(torch.randn(1, num_core_tokens, d_model) * 0.02)
        self.core_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.core_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.temporal_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        n_freq = input_days // 2 + 1
        self.freq_real = nn.Sequential(
            nn.Linear(n_freq, dim_feedforward), GeGLU(dim_feedforward, dim_feedforward),
        )
        self.freq_imag = nn.Sequential(
            nn.Linear(n_freq, dim_feedforward), GeGLU(dim_feedforward, dim_feedforward),
        )
        self.freq_fusion = nn.Sequential(
            nn.Linear(dim_feedforward * 2, d_model), nn.LayerNorm(d_model),
            GeGLU(d_model, d_model),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            GeGLU(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_days),
        )
        self.dlinear_skip = nn.Linear(input_days, output_days)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _core_fusion(self, temporal: torch.Tensor) -> torch.Tensor:
        B = temporal.size(0)
        core = self.core_tokens.expand(B, -1, -1)
        core, _ = self.core_cross_attn(core, temporal, temporal)
        core, _ = self.core_self_attn(core, core, core)
        temporal, _ = self.temporal_cross_attn(temporal, core, core)
        return temporal

    def _freq_features(self, series: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(series, dim=1)
        return self.freq_fusion(torch.cat([
            self.freq_real(spectrum.real), self.freq_imag(spectrum.imag),
        ], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        encoded_input = conv_features + self.residual(x)

        enriched = self._core_fusion(encoded_input)
        encoded = self.encoder(self.position(enriched))
        temporal_context = encoded.mean(dim=1)

        target = x[:, :, self.target_index]
        freq_context = self._freq_features(target)
        fused = temporal_context + freq_context

        nonlinear_forecast = self.head(fused)
        linear_forecast = self.dlinear_skip(target)
        return nonlinear_forecast + linear_forecast


def build_model(model_name: str, input_size: int, output_days: int) -> nn.Module:
    name = model_name.lower().replace("_", "-")
    if name == "lstm":
        return LSTMForecaster(input_size=input_size, output_days=output_days)
    if name == "transformer":
        return TransformerForecaster(input_size=input_size, output_days=output_days)
    if name in {"conv-transformer", "convtransformer", "conv"}:
        return ConvTransformerForecaster(input_size=input_size, output_days=output_days)
    if name in {"spectral-patch", "spectralpatch", "decomp-patch", "decomposition-patch"}:
        return SpectralDecompPatchForecaster(input_size=input_size, output_days=output_days)
    if name in {"spectral-patch-v2", "spectralpatchv2", "spv2"}:
        return SpectralPatchV2Forecaster(input_size=input_size, output_days=output_days)
    if name in {"spectral-patch-v2-norevin", "spv2-nr"}:
        return SpectralPatchV2Forecaster(input_size=input_size, output_days=output_days, use_revin=False)
    if name in {"flow-matching", "flowmatching", "fm"}:
        return FlowMatchingForecaster(input_size=input_size, output_days=output_days)
    if name in {"dlinear-plus", "dlinearplus", "dlinear"}:
        return DLinearPlusForecaster(input_size=input_size, output_days=output_days)
    if name in {"dlinear-plus-norevin", "dlinear-nr"}:
        return DLinearPlusForecaster(input_size=input_size, output_days=output_days, use_revin=False)
    if name in {"conv-transformer-v3", "conv-v3", "ctv3"}:
        return ConvTransformerV3Forecaster(input_size=input_size, output_days=output_days)
    raise ValueError(f"Unknown model: {model_name}")
