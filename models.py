from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

def _encoder_hidden_states(args, kwargs):
    v = kwargs.get("encoder_hidden_states")
    if v is None and len(args) > 1:
        v = args[1]
    if torch.is_tensor(v):
        return v.data_ptr()
    if isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
        return v[0].data_ptr()
    raise RuntimeError(
        "could not identify the CFG stream from encoder_hidden_states; "
        "the pipeline's transformer call signature may have changed"
    )

@dataclass(frozen=True)
class Adapter:
    name: str
    repo_id: str
    pipeline_cls: str
    blocks_attr: str
    cfg_style: str
    steps: int
    guidance: float
    guidance_kwarg: str       # the pipeline's name for the CFG scale
    height: int = 1024
    width: int = 1024
    extra_call_kwargs: dict = field(default_factory=dict)

    @property
    def n_streams(self) -> int:
        return 2 if self.cfg_style == "two_call" else 1

    @property
    def stream_key_fn(self) -> Callable | None:
        return _encoder_hidden_states if self.cfg_style == "two_call" else None

    def blocks(self, pipe) -> torch.nn.ModuleList:
        return getattr(pipe.transformer, self.blocks_attr)

    def call_kwargs(self, prompt, negative_prompt, steps, guidance,
                    height, width, generator) -> dict[str, Any]:
        kw = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=generator,
            **self.extra_call_kwargs,
        )
        kw[self.guidance_kwarg] = guidance
        return kw

ADAPTERS: dict[str, Adapter] = {
    "krea2-raw": Adapter(
        name="krea2-raw",
        repo_id="krea/Krea-2-Raw",
        pipeline_cls="Krea2Pipeline",
        blocks_attr="transformer_blocks",
        cfg_style="two_call",
        steps=52, guidance=3.5, guidance_kwarg="guidance_scale",
    ),
    "z-image": Adapter(
        name="z-image",
        repo_id="Tongyi-MAI/Z-Image",
        pipeline_cls="ZImagePipeline",
        blocks_attr="layers",
        cfg_style="batched",
        steps=50, guidance=4.0, guidance_kwarg="guidance_scale",
        extra_call_kwargs={"cfg_normalization": False},
    ),
    "qwen-image-2512": Adapter(
        name="qwen-image-2512",
        repo_id="Qwen/Qwen-Image-2512",
        pipeline_cls="QwenImagePipeline",
        blocks_attr="transformer_blocks",
        cfg_style="two_call",
        steps=50, guidance=4.0, guidance_kwarg="true_cfg_scale",
        height=1328, width=1328,
    ),
}

def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise SystemExit(
            f"unknown model {name!r}; choose from {sorted(ADAPTERS)}") from None

def load_pipeline(adapter: Adapter, device="cuda", dtype=torch.bfloat16,
                  offload: str = "none"):
    import diffusers

    cls = getattr(diffusers, adapter.pipeline_cls)
    pipe = cls.from_pretrained(adapter.repo_id, torch_dtype=dtype)
    if offload == "none":
        pipe.to(device)
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"unknown offload mode {offload!r}")
    pipe.set_progress_bar_config(disable=True)
    return pipe
