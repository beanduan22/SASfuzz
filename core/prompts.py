SYNTHESIS_TEMPLATE = """Task:
  Fill in the typed slots of the given skeleton with API calls drawn from the candidate pool, producing one complete, executable program.

Inputs:
  - Target library: {target_lib}
  - Skeleton template:
{skeleton}
  - Candidate API pool:
{api_pool}

Requirements:
  1) Preserve the skeleton. Do not modify the Model scaffold, the state-installing constructs, or the driver logic.
  2) Fill the body slot. Compose APIs drawn from the candidate pool into a dependency chain within the model body, using each API at most once.
  3) Fill the input slot. Provide a randomly initialized tensor matching the body's expected input shape and dtype.
  4) The resulting program must be syntactically valid and runnable as a standalone file.

Output:
  - Return ONLY the complete program code.
"""


REPAIR_TEMPLATE = """The program failed with the following error: {error_brief}

Repair instructions:
  1. Fix the error with minimal edits.
  2. Do not alter the skeleton scaffold: Model class structure, state-installing constructs, and driver logic must be preserved.
  3. Changes are allowed only within the body slot and the input slot (e.g., adjusting API arguments, input shapes or dtypes, or replacing a problematic API call with another from the original candidate pool). The skeleton scaffold must remain unchanged.

Original program:
{original_code}

Return the FULL corrected program only.
"""


def _format_pool(api_list: list[str]) -> str:
    return "\n".join(f"    - {a}" for a in api_list)


def _format_skeleton(skeleton: str) -> str:
    return "\n".join("    " + line for line in skeleton.splitlines())


def build_synthesis_prompt(
    api_list: list[str],
    skeleton: str,
    target_lib: str = "PyTorch",
) -> str:
    return SYNTHESIS_TEMPLATE.format(
        target_lib=target_lib,
        skeleton=_format_skeleton(skeleton),
        api_pool=_format_pool(api_list),
    )


def build_repair_prompt(original_code: str, error_brief: str) -> str:
    return REPAIR_TEMPLATE.format(
        error_brief=error_brief[:1500],
        original_code=original_code,
    )


def build_tf_synthesis_prompt(api_list: list[str], skeleton: str) -> str:
    return build_synthesis_prompt(api_list, skeleton, target_lib="TensorFlow 2.x")


def build_tf_repair_prompt(original_code: str, error_brief: str) -> str:
    return build_repair_prompt(original_code, error_brief)


# ── Ablation: w/o Skeleton ─────────────────────────────────────────────────
FREE_FORM_TEMPLATE = """Task:
  Write a complete, executable {target_lib} test program that defines a Model class
  and a make_inputs() function.

Inputs:
  - Target library: {target_lib}
  - Candidate API pool (use as many as possible, at most once each):
{api_pool}

Requirements:
  1) Define class Model(nn.Module) with a forward(self, x) method that chains
     API calls from the candidate pool into a dependency graph.
  2) Define make_inputs() -> list[torch.Tensor] that returns example inputs.
  3) At the end of the file, create an instance, run it, and (optionally) include
     autograd, mode switches, or distributed constructs if they fit naturally.
  4) The program must be syntactically valid and runnable as a standalone file.

Output:
  - Return ONLY the complete program code.
"""

NO_SCAFFOLD_TEMPLATE = """Task:
  Fill in the typed slots of the given skeleton with API calls drawn from the candidate pool.
  The skeleton structure is preserved but state-installing constructs are NOT pre-written;
  YOU must add them inside the body slot or driver to exercise the target state.

Inputs:
  - Target library: {target_lib}
  - Target state dimension: {dimension}
  - State hint: {state_hint}
  - Skeleton template (state scaffold removed):
{skeleton}
  - Candidate API pool:
{api_pool}

Requirements:
  1) Fill the body slot with API calls from the candidate pool.
  2) Fill the input slot with a concrete tensor.
  3) Add the state-installing constructs described in the state hint inside the
     body slot or after model instantiation (do not change the Model class signature).
  4) The resulting program must be syntactically valid and runnable.

Output:
  - Return ONLY the complete program code.
"""

_STATE_HINTS = {
    "gradient_tracking": (
        "Enable autograd: set requires_grad=True on input tensors and call "
        ".backward() or use torch.autograd.gradcheck on the model output."
    ),
    "execution_mode": (
        "Exercise train/eval mode: call model.train() and model.eval() and compare "
        "outputs, or trace the model with torch.jit.trace."
    ),
    "distribution_strategy": (
        "Use distributed execution: wrap the model with DistributedDataParallel or "
        "call torch.distributed collective ops (all_reduce, broadcast) on tensors."
    ),
}

# Generic base template for w/o Scaffold ablation (state constructs stripped)
_BASE_PT_TEMPLATE = """import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def forward(self, x):
        h = x
        # BODY_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
out = model(x)


def make_inputs():
    return [x]
"""

_BASE_TF_TEMPLATE = """import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def call(self, x):
        h = x
        # BODY_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
out = model(x)
"""


def build_free_form_prompt(api_list: list[str], target_lib: str = "PyTorch") -> str:
    return FREE_FORM_TEMPLATE.format(
        target_lib=target_lib,
        api_pool=_format_pool(api_list),
    )


def build_no_scaffold_prompt(
    api_list: list[str],
    dimension: str,
    target_lib: str = "PyTorch",
) -> str:
    base_tpl = _BASE_PT_TEMPLATE if "PyTorch" in target_lib else _BASE_TF_TEMPLATE
    hint = _STATE_HINTS.get(dimension, "Add appropriate state constructs.")
    return NO_SCAFFOLD_TEMPLATE.format(
        target_lib=target_lib,
        dimension=dimension,
        state_hint=hint,
        skeleton=_format_skeleton(base_tpl),
        api_pool=_format_pool(api_list),
    )
