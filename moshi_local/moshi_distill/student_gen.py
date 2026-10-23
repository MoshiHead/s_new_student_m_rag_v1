"""
student_gen.py — the 6.25 Hz student streaming backbone for LMGen
=================================================================

`StudentLMGen` is LMGen with the student's Temporal Transformer as the main
backbone: one 6.25 Hz step per pair of 12.5 Hz Mimi frames, upsampled back to
the 12.5 Hz conditioning the FROZEN teacher heads (out_norm / text_linear /
DepFormer) consume.

Kept separate from the web server (server_student.py) so it can be imported and
unit-tested without aiohttp.

Serving schedule (see models/student.py for the training side)
--------------------------------------------------------------
At step o the LMGen cache input is delayed-stream position p = o - 1:

    p odd  : the pair (p-1, p) completes now -> the state a_p is available
    p even : the pair (p, p+1) is still pending -> the state a_(p-1) is reused

With `streaming_fix="augment"` the reused (even) slots additionally receive the
current frame's embedding, exactly as `StudentLM.forward_train` does, so the
training objective and the serving behaviour cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models.lm import LMGen, _LMGenState
from .models.student import StudentLM
from .utils.compile import CUDAGraphed


@dataclass
class _StudentLMGenState(_LMGenState):
    """_LMGenState + the student's compressor pair buffer."""

    student_pending: torch.Tensor | None = None
    student_last_state: torch.Tensor | None = None

    def reset(self):
        super().reset()
        self.student_pending = None
        if self.student_last_state is not None:
            self.student_last_state.zero_()


class StudentLMGen(LMGen):
    """LMGen whose Temporal backbone is a distilled `StudentLM` at 6.25 Hz.

    The frame that completes a compressor pair consumes the freshest upsampled
    state; the next frame reuses it while its own pair is still pending (with
    the per-phase frame injection when the checkpoint enables the serving fix).
    Everything else (delay cache, DepFormer sampling, voice/text prompts) is
    inherited unchanged.
    """

    def __init__(self, student: StudentLM, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student = student

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        state = super()._init_streaming_state(batch_size)
        # Conditional pair-buffer mapping is not CUDA-graph safe; the frozen
        # DepFormer step stays graphed as in the teacher path.
        state.graphed_main = CUDAGraphed(self._student_forward_codes, disable=True)
        state.__class__ = _StudentLMGenState
        state.student_pending = None
        dtype = next(iter(self.lm_model.parameters())).dtype
        state.student_last_state = torch.zeros(
            batch_size, 1, self.lm_model.dim,
            device=self.lm_model.device, dtype=dtype)
        return state

    @torch.no_grad()
    def _student_forward_codes(
        self, input_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop-in for `LMModel.forward_codes` for one stream step."""
        state = self._streaming_state
        m = self.student.embed(input_tokens)                       # [B, 1, Ds]
        if state.student_pending is None:
            state.student_pending = m                              # even frame
        else:
            pair = torch.cat([state.student_pending, m], dim=1)    # [B, 2, Ds]
            z = self.student.compressor(pair)                      # [B, 1, Ds]
            u = self.student.transformer(z)                        # 6.25 Hz step
            s = self.student.out_norm(u)
            a = self.student.upsampler(s)                          # [B, 2, Dt]
            state.student_pending = None
            state.student_last_state = a[:, 1:2]                   # exact slot
        # Frozen teacher heads, exactly as in StudentLM.forward_train.
        transformer_out = self.lm_model.out_norm(state.student_last_state)
        # 6.25 Hz serving fix: on even slots the state above is one pair late;
        # inject the current frame's embedding, exactly as forward_train does.
        if getattr(self.student.cfg, "streaming_fix", "off") == "augment":
            phase = (state.offset - 1) % 2      # embed position p = offset - 1
            transformer_out = transformer_out + self.student.streaming_fix.phase_add(
                phase, m)
        text_logits = self.lm_model.text_linear(transformer_out)
        return transformer_out, text_logits[:, None]

    def load_voice_prompt_embeddings(self, path: str):
        raise NotImplementedError(
            "Pre-saved .pt voice prompt embeddings are not supported for the "
            "student: they were captured with the teacher pipeline and must "
            "not be replayed. The server falls back to a .wav prompt.")
