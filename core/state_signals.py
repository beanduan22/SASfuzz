from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set

_SIGMA_HEURISTIC: Dict[str, list[str]] = {
    "gradient_tracking": [
        "torch.nn.BCELoss", "torch.nn.BCEWithLogitsLoss",
        "torch.nn.CrossEntropyLoss", "torch.nn.CTCLoss",
        "torch.nn.L1Loss", "torch.nn.MSELoss", "torch.nn.NLLLoss",
        "torch.nn.KLDivLoss", "torch.nn.PoissonNLLLoss",
        "torch.nn.GaussianNLLLoss", "torch.nn.HuberLoss",
        "torch.nn.SmoothL1Loss", "torch.nn.HingeEmbeddingLoss",
        "torch.nn.MarginRankingLoss", "torch.nn.TripletMarginLoss",
        "torch.nn.CosineEmbeddingLoss", "torch.nn.MultiLabelMarginLoss",
        "torch.nn.SoftMarginLoss", "torch.nn.MultiMarginLoss",
        "torch.nn.AdaptiveLogSoftmaxWithLoss",
        "torch.nn.functional.binary_cross_entropy",
        "torch.nn.functional.binary_cross_entropy_with_logits",
        "torch.nn.functional.nll_loss", "torch.nn.functional.cross_entropy",
        "torch.nn.functional.ctc_loss", "torch.nn.functional.kl_div",
        "torch.nn.functional.l1_loss", "torch.nn.functional.mse_loss",
        "torch.nn.functional.huber_loss", "torch.nn.functional.smooth_l1_loss",
        "torch.nn.functional.poisson_nll_loss",
        "torch.nn.functional.softmax", "torch.nn.functional.log_softmax",
        "torch.nn.Embedding", "torch.nn.EmbeddingBag",
        "torch.linalg.", "torch.autograd.",
        "torch.reciprocal", "torch.log", "torch.sqrt", "torch.pow",
        "torch.Tensor.reciprocal", "torch.Tensor.log", "torch.Tensor.sqrt",
        "torch.gather", "torch.scatter", "torch.index_select",
        "torch.nn.functional.embedding",
        "tf.GradientTape", "tf.gradients", "tf.stop_gradient",
        "tf.math.reciprocal", "tf.math.log", "tf.math.sqrt",
        "tf.keras.losses.", "tf.losses.",
        "tf.nn.sigmoid_cross_entropy_with_logits",
        "tf.nn.softmax_cross_entropy_with_logits",
        "tf.nn.sparse_softmax_cross_entropy_with_logits",
    ],
    "execution_mode": [
        "torch.nn.BatchNorm1d", "torch.nn.BatchNorm2d", "torch.nn.BatchNorm3d",
        "torch.nn.LazyBatchNorm1d", "torch.nn.LazyBatchNorm2d", "torch.nn.LazyBatchNorm3d",
        "torch.nn.InstanceNorm1d", "torch.nn.InstanceNorm2d", "torch.nn.InstanceNorm3d",
        "torch.nn.GroupNorm", "torch.nn.LayerNorm", "torch.nn.RMSNorm",
        "torch.nn.Dropout", "torch.nn.Dropout1d", "torch.nn.Dropout2d",
        "torch.nn.Dropout3d", "torch.nn.AlphaDropout", "torch.nn.FeatureAlphaDropout",
        "torch.nn.LSTM", "torch.nn.GRU", "torch.nn.RNN",
        "torch.nn.LSTMCell", "torch.nn.GRUCell", "torch.nn.RNNCell",
        "torch.nn.MultiheadAttention",
        "torch.nn.functional.batch_norm", "torch.nn.functional.dropout",
        "torch.nn.functional.layer_norm", "torch.nn.functional.instance_norm",
        "torch.nn.functional.group_norm", "torch.nn.functional.alpha_dropout",
        "tf.function", "tf.autograph.",
        "tf.keras.layers.BatchNormalization", "tf.keras.layers.Dropout",
        "tf.keras.layers.LayerNormalization",
        "tf.nn.batch_normalization", "tf.nn.fused_batch_norm",
    ],
    "distribution_strategy": [
        "torch.nn.SyncBatchNorm",
        "torch.nn.parallel.", "torch.distributed.",
        "tf.distribute.", "tf.keras.callbacks.BackupAndRestore",
    ],
}

@dataclass
class StateSignals:
    """
    Holds σ_i(d) used in Eq. 1 of the paper:

        s_i(d) = (1 + σ_i(d)) / (u_i + 1)

    σ_i(d) is the documentation-derived state-relevance signal: it is 1 when
    a dimension-specific keyword for d matches a tokenized documentation
    field of API i, and 0 otherwise.

    sigma: dimension → set of API names/prefixes with σ=1
    """
    sigma: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "StateSignals":
        data = json.loads(Path(path).read_text())
        sigma = {d: set(apis) for d, apis in data.get("sigma", {}).items()}
        return cls(sigma=sigma)

    @classmethod
    def from_heuristics(cls) -> "StateSignals":
        sigma: Dict[str, Set[str]] = {
            d: set(prefixes) for d, prefixes in _SIGMA_HEURISTIC.items()
        }
        return cls(sigma=sigma)

    @classmethod
    def load(cls, signals_file: str | Path | None = None) -> "StateSignals":
        if signals_file is not None:
            p = Path(signals_file)
            if p.exists():
                return cls.from_file(p)
        default = Path(__file__).parent.parent / "state_signals.json"
        if default.exists():
            return cls.from_file(default)
        return cls.from_heuristics()

    def _match(self, api: str, dim_set: Set[str]) -> bool:
        if api in dim_set:
            return True
        return any(api.startswith(p) for p in dim_set if p.endswith(".") or p.endswith("_"))

    def get_sigma(self, api: str, dimension: str) -> int:
        if not dimension:
            return 0
        return int(self._match(api, self.sigma.get(dimension, set())))
