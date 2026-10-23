"""
ref_adapter.py — reference-text -> latent ref-slot conditioning (prefill-only)
==============================================================================

Design (locked in from RAG_PLAN.md):

  * Reference text is encoded **once** per retrieval into K vectors using the
    LM's own `text_emb` table + a small MLP.  No external encoder, no per-step
    module, no `context_gate_net`.
  * K = clamp(ceil(n_tokens / tokens_per_slot), 1, n_slots) — the same rule the
    corpus builder used for the slot spans — and slot j encodes the j-th
    contiguous token chunk of the reference (not the whole passage), so two
    different passages map to different vectors.
  * Every emitted vector has a fixed L2 norm (`out_norm`, ~ the mean `text_emb`
    row norm).  A learnable output scale initialised small collapses to zero
    before the LM learns to read the slots, and then no gradient ever reaches
    the encoder again.
  * The K vectors are written into the K reserved ref-slot frames
    (`<ref_start> [slots...] <ref_end>`, placed right after `<lookup>`).  The
    model consumes those frames at the normal 80 ms cadence, so the context
    lands in the KV cache and body generation attends to it via the ordinary
    causal attention.

Budget: the adapter runs once per retrieval (during the think-pause); a
steady-state step adds **zero** compute.

Usage (training)
----------------
    adapter = RefAdapter(dim=lm.dim, text_emb=lm.text_emb, n_slots=16)
    vecs = adapter(tok_ids, tok_mask)              # [B, K, dim]
    embedded = RefAdapter.inject(embedded, spans, vecs, mode="add")

where `spans` is a list of `(batch_idx, start_frame, end_frame)` ref-slot spans
(one per live reference).  Rows without a reference are simply omitted.

Usage (serving)
---------------
    vecs = adapter.encode_batch([reference], tokenizer, device)   # [1, K, dim]
    # then step the K slot frames through lm_gen with the vectors injected
    # (see server_student.py integration).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefAdapter(nn.Module):
    """Reference text -> K conditioning vectors from the LM's own text_emb.

    Parameters
    ----------
    dim : int
        LM hidden size (output dim of every vector).
    text_emb : nn.Module
        The LM's text embedding table (shared, frozen). Must accept [.., T]
        token ids and return [.., T, dim]. `ScaledEmbedding` qualifies.
    n_slots : int
        Number of latent slot frames the adapter fills (K).
    hidden : int | None
        MLP hidden width (defaults to `dim`).
    tokens_per_slot : int
        Slot budget: K = clamp(ceil(n_tokens / tokens_per_slot), 1, n_slots),
        the exact rule the corpus builder used for the slot spans.
    out_norm : float
        Fixed L2 norm of every emitted vector (default 1.0, i.e. about the
        mean norm of a `text_emb` row). Fixed on purpose: a learnable scale
        initialised small collapses to ~0 before the LM learns to read the
        slots, which kills the gradient path for good.
    """

    def __init__(self, dim: int, text_emb: nn.Module, n_slots: int = 16,
                 hidden: int | None = None, tokens_per_slot: int = 4,
                 out_norm: float = 1.0):
        super().__init__()
        # Shared with the LM: register it *outside* this module's parameter
        # tree so `adapter.state_dict()` / `adapter.parameters()` do not include
        # the 32k x 4096 embedding table (which would double the checkpoint and
        # put the whole table in the adapter's optimizer group).
        object.__setattr__(self, "text_emb", text_emb)
        self.dim = dim
        self.n_slots = n_slots
        self.tokens_per_slot = tokens_per_slot
        self.out_norm = out_norm
        hidden = hidden or dim

        self.norm_in = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        # One projection shared by every slot: slot identity comes from the
        # *content* of its token chunk, not from a per-slot row (a per-slot
        # 65k x 4096 matrix memorised slot noise instead of the passage).
        self.fc2 = nn.Linear(hidden, dim)

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    # -- core ---------------------------------------------------------------

    def forward(self, token_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """token_ids [B, T], attention_mask [B, T] bool/0-1 -> [B, K, dim].

        The reference is split into K contiguous token chunks (K from the same
        `ceil(n / tokens_per_slot)` rule the slot spans were built with) and
        each slot encodes *its own chunk*, so different passages produce
        different vectors.  Mean-pooling the whole passage into every slot made
        all references nearly collinear (cos ~0.97), which is why the LM
        ignored them.
        """
        n_vocab = getattr(self.text_emb, "num_embeddings", None)
        if n_vocab is not None:
            token_ids = token_ids.clamp(0, n_vocab - 1)
        emb = self.text_emb(token_ids)                     # [B, T, dim]
        # The shared table may be bf16 while the adapter is fp32 (or the other
        # way round, depending on how the checkpoint was saved / served).
        emb = emb.to(self.fc1.weight.dtype)
        B, T = token_ids.shape

        if attention_mask is not None:
            m = attention_mask.to(emb.dtype)                # [B, T]
        else:
            m = torch.ones(B, T, dtype=emb.dtype, device=emb.device)
        n = m.sum(dim=1).clamp(min=1.0)                     # [B]
        k = torch.clamp(torch.ceil(n / self.tokens_per_slot).long(),
                        1, self.n_slots)                    # [B]
        pos = torch.arange(T, device=emb.device)[None, :]
        chunk = (pos * k[:, None]) // n[:, None].long()     # [B, T] in 0..K-1

        slots = []
        for j in range(self.n_slots):
            mj = m * (chunk == j).to(emb.dtype)             # [B, T]
            slots.append((emb * mj.unsqueeze(-1)).sum(dim=1)
                         / mj.sum(dim=1, keepdim=True).clamp(min=1e-6))
        x = torch.stack(slots, dim=1)                       # [B, K, dim]

        x = self.norm_in(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                                     # [B, K, dim]
        return F.normalize(x, dim=-1) * self.out_norm

    # -- convenience --------------------------------------------------------

    @torch.no_grad()
    def encode_batch(self, references: list[str], tokenizer, device,
                     max_tokens: int = 256) -> torch.Tensor:
        """Encode references ('' -> zero slots) -> [B, K, dim]."""
        B = len(references)
        out = torch.zeros(B, self.n_slots, self.dim, device=device,
                          dtype=self.fc1.weight.dtype)
        live_ids, live_rows = [], []
        for i, ref in enumerate(references):
            if not ref:
                continue
            ids = tokenizer.encode(ref)[:max_tokens]
            if not ids:
                continue
            live_rows.append(i)
            live_ids.append(ids)

        for i, ids in zip(live_rows, live_ids):
            id_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            mask = torch.ones_like(id_t)
            out[i] = self.forward(id_t, mask)[0]
        return out

    # -- injection ----------------------------------------------------------

    @staticmethod
    def inject(embedded: torch.Tensor,
               spans: list[tuple[int, int, int]],
               vectors: torch.Tensor,
               mode: str = "add") -> torch.Tensor:
        """Write `vectors[i]` into frames `[s, e)` of batch row `b` for each
        span `(b, s, e)` in `spans`. Returns a NEW tensor (autograd-safe).

        mode="add"     : embed += context          (slot frames keep their
                         silence/sine audio embeddings)
        mode="replace" : embed  = context
        """
        if not spans or vectors.numel() == 0:
            return embedded
        B, T, D = embedded.shape
        assert vectors.shape[-1] == D
        device = embedded.device

        b_idx, t_idx, flat = [], [], []
        for i, (b, s, e) in enumerate(spans):
            if b < 0 or b >= B or s < 0 or s >= T:
                continue
            k = min(e - s, vectors.shape[1], T - s)
            if k <= 0:
                continue
            b_idx.extend([b] * k)
            t_idx.extend(range(s, s + k))
            flat.append(vectors[i, :k])
        if not flat:
            return embedded

        flat = torch.cat(flat, dim=0)
        idx = (torch.as_tensor(b_idx, device=device, dtype=torch.long) * T
               + torch.as_tensor(t_idx, device=device, dtype=torch.long))
        out = embedded.clone()
        out.view(B * T, D).index_put_((idx,), flat, accumulate=(mode == "add"))
        return out

    @staticmethod
    def spans_from_metas(seg_metas, reference_texts, offset: int = 0,
                         kind: str = "ref_slot") -> list[tuple[int, int, int]]:
        """Collect (batch_idx, start+offset, end+offset) for every segment
        that has a `<ref_slot>` span and a non-empty reference."""
        spans = []
        for meta, ref in zip(seg_metas, reference_texts):
            if meta is None or not ref:
                continue
            slots = meta.spans_of(kind)
            if not slots:
                continue
            s, e = slots[0].start_frame, slots[0].end_frame
            spans.append((meta.batch_idx, s + offset, e + offset))
        return spans


class RefAddAdapter(nn.Module):
    """Fact card -> one **additive** vector per spoken frame (MoshiRAG eq. 2).

    Unlike :class:`RefAdapter` nothing is written into reserved prefill frames:
    vector ``j`` is added to the input embedding of the ``j``-th frame the model
    is already speaking, so the reference costs **zero extra frames** and zero
    silence on the wire.

        vec_j = scale * normalize(fc2(gelu(fc1(LayerNorm(text_emb[tok_j])))))

    One vector per *token* (no chunk pooling): the trainer and the server both
    feed the fact card from :func:`rag_ref_select.select_ids`, so slot ``j`` is
    always a real token of the passage -- usually the digits of a year.

    ``scale`` starts small but **not** at 0 (0.1 -> vectors of ~0.1x a text
    embedding row, ~0.09 on the S20 student).  A hard zero would make step 0
    bit-identical to the no-reference model, but it also starves ``fc1``/``fc2``
    of gradient (the direction only learns in proportion to ``scale``), which
    is fatal on a ~1k-step LoRA budget: on the tiny twin the gradient on
    ``scale`` is ~3e-4, so at ``lr=1e-3`` the gate would need >1e5 steps to
    open.  0.1 keeps the perturbation negligible while letting the direction
    learn from the first batch.
    """

    def __init__(self, dim: int, text_emb: nn.Module, n_slots: int = 16,
                 hidden: int | None = None, init_scale: float = 0.1,
                 out_norm: float = 1.0):
        super().__init__()
        # Shared with the LM, kept out of this module's parameter tree (see
        # RefAdapter for why).
        object.__setattr__(self, "text_emb", text_emb)
        self.dim = dim
        self.n_slots = n_slots
        hidden = hidden or dim

        self.norm_in = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)
        self.out_norm = float(out_norm)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, token_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """token_ids [B, K] -> [B, K, dim] (rows with mask 0 -> zeros)."""
        n_vocab = getattr(self.text_emb, "num_embeddings", None)
        if n_vocab is not None:
            token_ids = token_ids.clamp(0, n_vocab - 1)
        emb = self.text_emb(token_ids)                     # [B, K, dim]
        emb = emb.to(self.fc1.weight.dtype)
        x = self.fc2(F.gelu(self.fc1(self.norm_in(emb))))
        # fixed L2 norm (like RefAdapter): the model then learns *which*
        # direction to inject and a single scalar for *how much*, instead of
        # having to discover a sane magnitude from an unconstrained linear
        # layer whose raw output norm is ~100x a text embedding.
        x = F.normalize(x, dim=-1) * self.out_norm
        if attention_mask is not None:
            x = x * attention_mask.to(x.dtype).unsqueeze(-1)
        return x * self.scale

    @torch.no_grad()
    def encode_ids(self, ids: list[int], device) -> torch.Tensor:
        """One fact card -> [K, dim] (no batch dim)."""
        id_t = torch.tensor([list(ids)], dtype=torch.long, device=device)
        return self.forward(id_t, torch.ones_like(id_t))[0]


class RefBagAdapter(nn.Module):
    """Fact card -> one **bag** vector per carrier frame after ``<lookup>``.

    Why not one learned vector per token: the temporal transformer reads

        emb[t] = sum_i audio_emb_i(audio_i[t]) + text_emb(token[t])

    and the only channel it has ever been trained to read *semantics* from is
    the text slot -- the 130-frame persona prompt, and S23's 18-frame card
    block (body CE +0.10).  A random 4096-d vector added to the audio sum is a
    new modality: adapter and LM would have to agree on a code within a ~1k
    step LoRA budget.  S24 tried it and the with/without generations were
    character-identical (body CE delta +-0.002).

    So the carrier carries the **native** representation:

        bag = sum_j w_j * text_emb(tok_j) / sqrt(sum_j w_j)
        vec = scale * (alpha * bag + bounded_mlp(bag)) * decay**j

    dividing by ``sqrt(K)`` keeps the magnitude of one word regardless of card
    length, and with ``alpha=1`` + zero-init ``fc2`` a one-token card injects
    *exactly* that token's text embedding -- a word the LM already reads.
    ``alpha`` and the MLP then learn the "use it, do not say it" transform.

    Frame 0 (the ``<lookup>`` frame) gets the full bag: measured
    ``|dCE_body/dv|`` is 0.75 there versus 0.09-0.17 on the lead, and the
    server discards that frame's audio, so a strong vector cannot be heard.
    The spoken frames after it decay geometrically (``decay=0.85`` -> 0.09 by
    frame 15), because there the audio CE is listening.
    """

    def __init__(self, dim: int, text_emb: nn.Module, n_slots: int = 16,
                 hidden: int | None = None, alpha: float = 1.0,
                 decay: float = 0.85, init_scale: float = 1.0,
                 res_norm: float = 1.0):
        super().__init__()
        # Shared with the LM, kept out of this module's parameter tree (see
        # RefAdapter for why).
        object.__setattr__(self, "text_emb", text_emb)
        self.dim = dim
        self.n_slots = n_slots
        hidden = hidden or dim

        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        # zero-init residual: at step 0 the vector is the pure native bag
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.w = nn.Parameter(torch.ones(n_slots))
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.decay = float(decay)
        self.res_norm = float(res_norm)

    def forward(self, token_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """token_ids [B, K] -> [B, K, dim]; row b is the bag for frame j scaled
        by ``decay ** j`` (all ``j`` of one row carry the same bag)."""
        n_vocab = getattr(self.text_emb, "num_embeddings", None)
        if n_vocab is not None:
            token_ids = token_ids.clamp(0, n_vocab - 1)
        emb = self.text_emb(token_ids).to(self.fc2.weight.dtype)   # [B, K, dim]
        B, K = token_ids.shape
        if attention_mask is None:
            m = torch.ones(B, K, device=emb.device, dtype=emb.dtype)
        else:
            m = attention_mask.to(emb.dtype)
        w = self.w[:K].view(1, K) * m                              # [B, K]
        k = w.sum(1, keepdim=True).clamp(min=1.0)                  # [B, 1]
        bag = (emb * w.unsqueeze(-1)).sum(1) / k.sqrt()            # [B, dim]
        # No LayerNorm on the input: the bag is already the scale of one word,
        # while LN would push it to ~sqrt(dim) and make the zero-init fc2 take
        # 3e-3-sized steps from a 80-norm hidden state (measured: ref_norm
        # exploded to 1750 in 120 steps on the first gate attempt).
        r = self.fc2(F.gelu(self.fc1(bag)))
        # ... and the residual is *bounded*: |res| < res_norm no matter what
        # fc2 learns, so an injected vector can never swamp the input stream
        # (the whole stream is |emb| ~ 2).
        rn = r.norm(dim=-1, keepdim=True)
        res = self.res_norm * r / (rn + self.res_norm)
        v = self.alpha * bag + res                                 # [B, dim]
        prof = torch.tensor([self.decay ** j for j in range(K)],
                            device=emb.device, dtype=emb.dtype)
        out = (v.unsqueeze(1) * prof.view(1, K, 1)) * self.scale
        # an empty card (all masks 0) must inject exactly zero, even after
        # fc1 has learned a bias
        has = (m.sum(1, keepdim=True) > 0).to(out.dtype)
        return out * has.unsqueeze(-1)

    @torch.no_grad()
    def encode_ids(self, ids: list[int], device) -> torch.Tensor:
        """One fact card -> [K, dim], one carrier vector per upcoming frame."""
        id_t = torch.tensor([list(ids)], dtype=torch.long, device=device)
        return self.forward(id_t, torch.ones_like(id_t))[0]


def spans_and_rows(seg_metas, has_ref, kind: str = "ref_slot"):
    """``(spans, rows)`` with ``rows[i]`` the batch row of ``spans[i]``.

    ``RefAdapter.inject`` indexes ``vectors`` by *span* order, so a batch that
    mixes reference and no-reference rows must gather ``vecs[rows]`` first --
    otherwise row 0's vector is injected into row 1's frames.
    """
    spans, rows = [], []
    for meta, h in zip(seg_metas, has_ref):
        if not h:
            continue
        sl = meta.spans_of(kind)
        if not sl:
            continue
        s, e = sl[0].start_frame, sl[0].end_frame
        if e <= s:
            continue
        spans.append((meta.batch_idx, s, e))
        rows.append(meta.batch_idx)
    return spans, rows


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    class _DummyTok:
        def encode(self, s):
            return list(range(1, min(len(s.split()) + 1, 10)))

    torch.manual_seed(0)
    dim, K = 32, 8
    emb = nn.Embedding(1000, dim)
    adapter = RefAdapter(dim=dim, text_emb=emb, n_slots=K)

    ids = torch.randint(1, 1000, (3, 17))
    mask = torch.ones_like(ids, dtype=torch.bool)
    vecs = adapter(ids, mask)
    assert vecs.shape == (3, K, dim), vecs.shape
    print("adapter output:", tuple(vecs.shape))

    # zero-slot path
    z = adapter.encode_batch(["", "hello there", ""], tokenizer=_DummyTok(), device="cpu")
    assert z.shape == (3, K, dim)
    assert torch.allclose(z[0], torch.zeros_like(z[0]))
    assert not torch.allclose(z[1], torch.zeros_like(z[1]))
    print("encode_batch zero-slot behaviour OK")

    # injection is autograd-safe and writes the right frames
    embedded = torch.randn(2, 12, dim, requires_grad=True)
    spans = [(0, 3, 6), (1, 8, 11)]
    out = RefAdapter.inject(embedded, spans, vecs[:2])
    assert out.shape == embedded.shape
    assert torch.allclose(out[0, 3:6], embedded[0, 3:6] + vecs[0, :3])
    assert torch.allclose(out[1, 8:10], embedded[1, 8:10] + vecs[1, :2])
    out.sum().backward()
    assert embedded.grad is not None and adapter.fc2.weight.grad is not None
    print("inject + backward OK")
    print("REF_ADAPTER_OK")
