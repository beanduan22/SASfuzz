from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Skeleton:
    skeleton_id: str
    framework: str
    dimension: str
    description: str
    harness: str
    offline_slots: Dict[str, str] = field(default_factory=dict)

    @property
    def template(self) -> str:
        scaffold = _SCAFFOLD_PT if self.framework == "pytorch" else _SCAFFOLD_TF
        shown = scaffold.format(
            LAYER='        "<<LAYER_SLOT: define layers used by the body>>"',
            BODY='        "<<BODY_SLOT: chain candidate-pool APIs into h>>"',
            INPUT='    "<<INPUT_SLOT: build a randomly initialised tensor x>>"',
        )
        return shown + "\n\nFROZEN STATE HARNESS (do not modify):\n" + self.harness



_SCAFFOLD_PT = '''import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
{LAYER}

    def forward(self, x):
        h = x
{BODY}
        return h


def make_inputs():
{INPUT}
    return [x]
'''

_PT_DEFAULT_SLOTS = {
    "LAYER": "        self.lin = nn.Linear(8, 8)",
    "BODY": "        h = torch.relu(self.lin(h))",
    "INPUT": "    x = torch.randn(4, 8)",
}

_PT_BN_SLOTS = {
    "LAYER": "        self.lin = nn.Linear(8, 8)\n        self.bn = nn.BatchNorm1d(8)",
    "BODY": "        h = self.bn(self.lin(h))",
    "INPUT": "    x = torch.randn(4, 8)",
}

_PT_G1_HARNESS = '''def sas_run(device, inputs=None):
    torch.manual_seed(42)
    if inputs is None:
        inputs = make_inputs()
    model = Model().to(device).double().train()
    xs = []
    for t in inputs:
        if isinstance(t, torch.Tensor):
            t = t.to(device)
            if t.is_floating_point():
                t = t.detach().double().requires_grad_(True)
        xs.append(t)
    out = model(*xs)
    outs = out if isinstance(out, (list, tuple)) else [out]
    targets = [t for t in xs if isinstance(t, torch.Tensor) and t.requires_grad]
    loss = sum(o.double().sum() for o in outs if isinstance(o, torch.Tensor))
    grads = torch.autograd.grad(loss, targets, allow_unused=True) if targets else []
    result = {}
    for i, o in enumerate(outs):
        if isinstance(o, torch.Tensor):
            result["out%d" % i] = o.detach()
    for i, g in enumerate(grads):
        if isinstance(g, torch.Tensor):
            result["grad%d" % i] = g.detach()
    return result
'''

_PT_G2_HARNESS = '''def sas_run(device, inputs=None):
    torch.manual_seed(42)
    if inputs is None:
        inputs = make_inputs()
    model = Model().to(device).double().eval()
    xs = [t.to(device).double() if isinstance(t, torch.Tensor) and t.is_floating_point()
          else (t.to(device) if isinstance(t, torch.Tensor) else t) for t in inputs]
    result = {}
    with torch.no_grad():
        out = model(*xs)
    outs = out if isinstance(out, (list, tuple)) else [out]
    for i, o in enumerate(outs):
        if isinstance(o, torch.Tensor):
            result["out%d" % i] = o.detach()
    return result
'''

_PT_M1_HARNESS = '''def sas_run(device, inputs=None):
    torch.manual_seed(42)
    if inputs is None:
        inputs = make_inputs()
    model = Model().to(device).double()
    xs = [t.to(device).double() if isinstance(t, torch.Tensor) and t.is_floating_point()
          else (t.to(device) if isinstance(t, torch.Tensor) else t) for t in inputs]
    result = {}
    for mode in ("train", "eval"):
        getattr(model, mode)()
        with torch.no_grad():
            out = model(*xs)
        outs = out if isinstance(out, (list, tuple)) else [out]
        for i, o in enumerate(outs):
            if isinstance(o, torch.Tensor):
                result["%s_out%d" % (mode, i)] = o.detach()
    return result
'''

_PT_M2_HARNESS = '''def sas_run(device, inputs=None):
    torch.manual_seed(42)
    if inputs is None:
        inputs = make_inputs()
    model = Model().to(device).double().eval()
    xs = [t.to(device).double() if isinstance(t, torch.Tensor) and t.is_floating_point()
          else (t.to(device) if isinstance(t, torch.Tensor) else t) for t in inputs]
    result = {}
    with torch.no_grad():
        out_eager = model(*xs)
        traced = torch.jit.trace(model, tuple(xs), check_trace=False)
        out_traced = traced(*xs)
    for name, out in (("eager", out_eager), ("traced", out_traced)):
        outs = out if isinstance(out, (list, tuple)) else [out]
        for i, o in enumerate(outs):
            if isinstance(o, torch.Tensor):
                result["%s_out%d" % (name, i)] = o.detach()
    return result
'''

_PT_D1_HARNESS = '''def sas_run(device, inputs=None):
    import tempfile
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.manual_seed(42)
    if inputs is None:
        inputs = make_inputs()
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        idx = torch.device(device).index or 0
        device = "cuda:%d" % idx
    backend = "nccl" if is_cuda else "gloo"
    if not dist.is_initialized():
        _f = tempfile.NamedTemporaryFile(delete=False)
        _f.close()
        dist.init_process_group(backend=backend, init_method="file://" + _f.name,
                                rank=0, world_size=1)
    model = Model().to(device).double()
    ddp = DDP(model, device_ids=[torch.device(device).index] if is_cuda else None)
    xs = []
    for t in inputs:
        if isinstance(t, torch.Tensor):
            t = t.to(device)
            if t.is_floating_point():
                t = t.detach().double().requires_grad_(True)
        xs.append(t)
    out = ddp(*xs)
    outs = out if isinstance(out, (list, tuple)) else [out]
    loss = sum(o.double().sum() for o in outs if isinstance(o, torch.Tensor))
    loss.backward()
    result = {}
    for i, o in enumerate(outs):
        if isinstance(o, torch.Tensor):
            result["out%d" % i] = o.detach()
    for n, p in model.named_parameters():
        if p.grad is not None:
            result["grad_%s" % n] = p.grad.detach()
    return result
'''

_PT_D2_HARNESS = '''def sas_run(device, inputs=None):
    import tempfile
    import torch.distributed as dist
    torch.manual_seed(42)
    if inputs is None:
        inputs = make_inputs()
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        idx = torch.device(device).index or 0
        device = "cuda:%d" % idx
    backend = "nccl" if is_cuda else "gloo"
    if not dist.is_initialized():
        _f = tempfile.NamedTemporaryFile(delete=False)
        _f.close()
        dist.init_process_group(backend=backend, init_method="file://" + _f.name,
                                rank=0, world_size=1)
    model = Model().to(device).double().eval()
    xs = [t.to(device).double() if isinstance(t, torch.Tensor) and t.is_floating_point()
          else (t.to(device) if isinstance(t, torch.Tensor) else t) for t in inputs]
    with torch.no_grad():
        out = model(*xs)
    outs = out if isinstance(out, (list, tuple)) else [out]
    result = {}
    for i, o in enumerate(outs):
        if isinstance(o, torch.Tensor):
            red = o.clone().contiguous()
            dist.all_reduce(red, op=dist.ReduceOp.SUM)
            result["reduced%d" % i] = red.detach()
    return result
'''

PYTORCH_SKELETONS = [
    Skeleton("PT-G1", "pytorch", "gradient_tracking",
             "requires_grad + backward", _PT_G1_HARNESS, dict(_PT_DEFAULT_SLOTS)),
    Skeleton("PT-G2", "pytorch", "gradient_tracking",
             "torch.no_grad scope", _PT_G2_HARNESS, dict(_PT_DEFAULT_SLOTS)),
    Skeleton("PT-M1", "pytorch", "execution_mode",
             "train()/eval() switch", _PT_M1_HARNESS, dict(_PT_BN_SLOTS)),
    Skeleton("PT-M2", "pytorch", "execution_mode",
             "torch.jit.trace", _PT_M2_HARNESS, dict(_PT_DEFAULT_SLOTS)),
    Skeleton("PT-D1", "pytorch", "distribution_strategy",
             "DistributedDataParallel wrap", _PT_D1_HARNESS, dict(_PT_DEFAULT_SLOTS)),
    Skeleton("PT-D2", "pytorch", "distribution_strategy",
             "torch.distributed collective", _PT_D2_HARNESS, dict(_PT_DEFAULT_SLOTS)),
]


_SCAFFOLD_TF = '''import numpy as np
import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
{LAYER}

    def call(self, x, training=False):
        h = x
{BODY}
        return h


def make_inputs():
{INPUT}
    return [x]
'''

_TF_DEFAULT_SLOTS = {
    "LAYER": "        self.dense = tf.keras.layers.Dense(8)",
    "BODY": "        h = tf.nn.relu(self.dense(h))",
    "INPUT": "    x = np.random.randn(4, 8).astype(np.float64)",
}

_TF_BN_SLOTS = {
    "LAYER": "        self.dense = tf.keras.layers.Dense(8)\n"
             "        self.bn = tf.keras.layers.BatchNormalization()",
    "BODY": "        h = self.bn(self.dense(h), training=training)",
    "INPUT": "    x = np.random.randn(4, 8).astype(np.float64)",
}


def _tf_dev(device: str) -> str:
    return device


_TF_G1_HARNESS = '''def sas_run(device, inputs=None):
    tf.random.set_seed(42)
    if inputs is None:
        inputs = make_inputs()
    result = {}
    with tf.device(device):
        model = Model()
        x = tf.constant(np.asarray(inputs[0], dtype=np.float64))
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = model(x, training=False)
        grad = tape.gradient(out, x)
        result["out"] = np.asarray(out, dtype=np.float64)
        if grad is not None:
            result["grad"] = np.asarray(grad, dtype=np.float64)
    return result
'''

_TF_G2_HARNESS = '''def sas_run(device, inputs=None):
    tf.random.set_seed(42)
    if inputs is None:
        inputs = make_inputs()
    result = {}
    with tf.device(device):
        model = Model()
        x = tf.constant(np.asarray(inputs[0], dtype=np.float64))
        with tf.GradientTape() as tape:
            tape.watch(x)
            h = model(x, training=False)
            h = tf.stop_gradient(h)
            out = h
        grad = tape.gradient(out, x)
        result["out"] = np.asarray(out, dtype=np.float64)
        result["grad_is_none"] = np.asarray([1.0 if grad is None else 0.0])
    return result
'''

_TF_M1_HARNESS = '''def sas_run(device, inputs=None):
    tf.random.set_seed(42)
    if inputs is None:
        inputs = make_inputs()
    result = {}
    with tf.device(device):
        model = Model()
        x = tf.constant(np.asarray(inputs[0], dtype=np.float64))
        y_eager = model(x, training=False)
        graph_fn = tf.function(model.__call__)
        y_graph = graph_fn(x, training=False)
        result["eager"] = np.asarray(y_eager, dtype=np.float64)
        result["graph"] = np.asarray(y_graph, dtype=np.float64)
    return result
'''

_TF_M2_HARNESS = '''def sas_run(device, inputs=None):
    tf.random.set_seed(42)
    if inputs is None:
        inputs = make_inputs()
    result = {}
    with tf.device(device):
        model = Model()
        x = tf.constant(np.asarray(inputs[0], dtype=np.float64))
        y_eager = model(x, training=False)

        @tf.function(jit_compile=True)
        def xla_fn(z):
            return model(z, training=False)

        try:
            y_xla = xla_fn(x)
            result["xla"] = np.asarray(y_xla, dtype=np.float64)
        except Exception:
            pass
        result["eager"] = np.asarray(y_eager, dtype=np.float64)
    return result
'''

_TF_D1_HARNESS = '''def sas_run(device, inputs=None):
    tf.random.set_seed(42)
    if inputs is None:
        inputs = make_inputs()
    result = {}
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Model()
    x = tf.constant(np.asarray(inputs[0], dtype=np.float64))
    out = strategy.run(lambda z: model(z, training=False), args=(x,))
    result["out"] = np.asarray(strategy.experimental_local_results(out)[0], dtype=np.float64)
    return result
'''

_TF_D2_HARNESS = '''def sas_run(device, inputs=None):
    tf.random.set_seed(42)
    if inputs is None:
        inputs = make_inputs()
    result = {}
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Model()
    x = tf.constant(np.asarray(inputs[0], dtype=np.float64))

    def step(z):
        h = model(z, training=False)
        ctx = tf.distribute.get_replica_context()
        return ctx.all_reduce(tf.distribute.ReduceOp.SUM, h)

    out = strategy.run(step, args=(x,))
    result["reduced"] = np.asarray(strategy.experimental_local_results(out)[0], dtype=np.float64)
    return result
'''

TENSORFLOW_SKELETONS = [
    Skeleton("TF-G1", "tensorflow", "gradient_tracking",
             "GradientTape + gradient", _TF_G1_HARNESS, dict(_TF_DEFAULT_SLOTS)),
    Skeleton("TF-G2", "tensorflow", "gradient_tracking",
             "stop_gradient inside tape", _TF_G2_HARNESS, dict(_TF_DEFAULT_SLOTS)),
    Skeleton("TF-M1", "tensorflow", "execution_mode",
             "tf.function tracing", _TF_M1_HARNESS, dict(_TF_BN_SLOTS)),
    Skeleton("TF-M2", "tensorflow", "execution_mode",
             "tf.function(jit_compile=True)", _TF_M2_HARNESS, dict(_TF_DEFAULT_SLOTS)),
    Skeleton("TF-D1", "tensorflow", "distribution_strategy",
             "MirroredStrategy scope", _TF_D1_HARNESS, dict(_TF_DEFAULT_SLOTS)),
    Skeleton("TF-D2", "tensorflow", "distribution_strategy",
             "tf.distribute collective", _TF_D2_HARNESS, dict(_TF_DEFAULT_SLOTS)),
]



def build_model_code(skeleton: Skeleton, layer: str, body: str, input_code: str) -> str:
    scaffold = _SCAFFOLD_PT if skeleton.framework == "pytorch" else _SCAFFOLD_TF
    filled = scaffold.format(LAYER=layer, BODY=body, INPUT=input_code)
    return filled + "\n\n" + skeleton.harness


def fill_offline(skeleton: Skeleton, api_list: list[str]) -> str:
    s = skeleton.offline_slots
    return build_model_code(skeleton, s["LAYER"], s["BODY"], s["INPUT"])


def get_skeletons(framework: str) -> list[Skeleton]:
    if framework == "pytorch":
        return list(PYTORCH_SKELETONS)
    if framework == "tensorflow":
        return list(TENSORFLOW_SKELETONS)
    raise ValueError(f"unsupported framework: {framework}")
