"""rag_ref_select.py — what goes into the K ref-slot frames.

The cache sized the block for the *latent* path (K = ceil(n_tokens / 4) chunks).
In text mode the trainer wrote `ref_ids[:K]`, i.e. the first K tokens of a
passage that is typically 66 tokens long, so only ~1/4 of the passage ever
reached the model.  Measured over the whole cache: 36% of the body's digit
tokens were in the block, and only **15% of the body's years** were in it
whole, even though 98.6% of those years are somewhere in the passage.  That is
why the model "answers from its prior": for most samples the number was never
in the block.

`select_ids` fits the facts into the same K frames:

  1. every run of digit tokens (a year like 1960 is 4 tokens: 1,9,6,0),
     kept whole, as long as it fits the budget;
  2. then content words (no stopwords), in passage order;
  3. emitted in passage order, so the block still reads left to right.

Same frame budget, same embedding path, no new tokens:

    verbatim   digit 36%  year 15%   (old behaviour)
    facts      digit 83%  year 39%   (whole-number aware, +4 frames: 45%)

Used by both the trainer (`--ref-cards`) and the server
(`prepare_reference`), so training and serving see exactly the same block.
"""

from __future__ import annotations

import re

STOP = set("""
a an the this that these those it its he she they them his her their him
is are was were be been being am do does did have has had having
of to in on at by for from with without into onto over under about after
before during between among through across against along around behind
and or but nor so than then as if because while although though since
not no nor only just also very more most much many few some any all both
each other another such own same can could may might must shall should will
would i you we me my our your us one two three first second new
there here where when who whom whose which what how why
named called known used made
""".split())


def _word(sp, t: int) -> str:
    return sp.id_to_piece(int(t)).replace("\u2581", " ").strip().lower()


def _is_digit(sp, t: int) -> bool:
    p = sp.id_to_piece(int(t))
    return any(ch.isdigit() for ch in p)


def select_ids(sp, ids, k: int) -> list[int]:
    """Pick at most ``k`` ids from ``ids``: whole numbers first, then content."""
    ids = [int(t) for t in ids if int(t) > 0]
    k = int(k)
    if k <= 0 or not ids:
        return []
    if len(ids) <= k:
        return ids

    # runs of consecutive digit tokens == one number (1960 -> 1,9,6,0)
    runs, cur = [], []
    for i, t in enumerate(ids):
        if _is_digit(sp, t):
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    keep: list[int] = []
    for r in runs:                         # whole numbers, earliest first
        if len(keep) + len(r) <= k:
            keep += r
        elif not keep:                     # a single number longer than k
            keep += r[:k]
    for i, t in enumerate(ids):            # then content words
        if len(keep) >= k:
            break
        if i in keep or _is_digit(sp, t):
            continue
        w = _word(sp, t)
        if w not in STOP and len(w) > 2:
            keep.append(i)
    if len(keep) < k:                      # backfill with the dropped words
        for i in range(len(ids)):
            if len(keep) >= k:
                break
            if i not in keep:
                keep.append(i)
    return [ids[i] for i in sorted(keep)][:k]


# ---------------------------------------------------------------------------
# Window selection (S28 checkpoint)
# ---------------------------------------------------------------------------
#
# S28 was trained on *contiguous sentence windows*
# (`build_cards_win2.py`: 4-gram Jaccard against the answer, sentence-start
# bonus 0.6, word-start bonus 0.2), not on the keyword bag above.  Serving must
# produce the same shape of block or the model sees a distribution it never
# trained on:
#
#     bag                 first 16 tokens of the passage   -> 19% answer coverage
#     contiguous window   the passage sentence about the query -> ~40%
#
# The query is the STT transcript, so the window is the passage sentence the
# user actually asked about.

_WORD = re.compile(r"[a-z0-9]+")

# Words that name the *kind* of answer rather than appear next to it: a
# question about a year usually never says "year" in the answer sentence
# ("... released on December 22, 2017").  Anchoring on them lands on an
# unrelated "the year 1998" elsewhere in the passage.
ANSWER_TYPES = {"year", "years", "date", "time", "name", "number", "type",
                "kind", "sort", "much", "many", "old", "long", "tall", "high",
                "big", "large", "small", "far", "often"}


def _norm(text: str) -> str:
    return "".join(_WORD.findall(text.lower()))


def _grams(text: str, n: int = 4) -> set[str]:
    s = _norm(text)
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))} if s else set()


def _jac(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_window_ids(sp, ids, query_ids, k: int, *, head_w: float = 3.0,
                      sentence_bonus: float = 0.6,
                      word_bonus: float = 0.2) -> list[int]:
    """Return the ``k``-token contiguous passage window that best matches the query.

    Same shape as the training cards (``build_cards_win2.py``): one contiguous
    sentence window, so the block the model reads at serving looks like the
    block it was trained on.

    The query is a *question*, not the answer, so plain similarity picks the
    topic sentence and stops right before the fact::

        query   who directed the post
        jaccard The Post is a 2017 American historical political thriller
                film directed and          <- no "Steven Spielberg"

    Factoid answers sit next to the question's predicate, so the selector
    anchors on the query's rarest content word (``directed``) and puts it a
    third of the way in -- room for a name before it and the answer after it::

        anchored  film directed and produced by Steven Spielberg and written
                  by Liz Hannah and Josh Singer

    Falls back to the best 4-gram Jaccard window when no query word occurs in
    the passage, and to the passage head when there is no query at all.
    """
    ids = [int(t) for t in ids if 0 < int(t) < 32000]
    k = int(k)
    if not ids or k <= 0:
        return []
    if len(ids) <= k:
        return ids
    if not query_ids:
        return ids[:k]

    # -- anchor on the rarest question word --------------------------------
    pieces = [sp.id_to_piece(t).replace("\u2581", " ").lower() for t in ids]
    text = "".join(pieces)
    starts, c = [], 0
    for p in pieces:
        starts.append(c)
        c += len(p)
    q_words = [w for w in _WORD.findall(
        sp.decode([int(t) for t in query_ids
                   if 0 < int(t) < 32000]).lower())
        if w not in STOP and w not in ANSWER_TYPES and len(w) > 2]
    if q_words:
        found = []
        for w in dict.fromkeys(q_words):          # keep query order
            # 5-char stem: "director" finds "directed", "released" -> "release"
            stem = re.escape(w[:5])
            ms = list(re.finditer(r"\b%s[a-z0-9]*\b" % stem, text))
            if ms:
                found.append((len(ms), -len(w), w, ms[0].start()))
        if found:
            _, _, word, pos = min(found)          # rarest, then longest
            idx = 0
            for i, s in enumerate(starts):        # token holding the anchor
                if s <= pos:
                    idx = i
                else:
                    break
            base = max(0, min(idx - k // 3, len(ids) - k))
            # Around the anchor, prefer the window that actually carries the
            # fact.  Answer tokens are numbers and proper nouns, so a window
            # two tokens further in often holds "December 14, 2017" that the
            # naive one cuts off.  No query-overlap term here: the anchor
            # already guarantees the query word, and re-scoring it would pull
            # the window back over the topic instead of forward onto the fact.
            best_sc, best_st = -1e9, base
            for st in range(max(0, base - k // 2),
                            min(base + k // 2, len(ids) - k) + 1):
                wp = pieces[st:st + k]
                digits = sum(1 for p in wp if any(ch.isdigit() for ch in p))
                caps = sum(1 for p in wp if p.strip()[:1].isupper())
                sc = (0.5 * min(digits, 2) / 2
                      + 0.15 * min(caps, 4) / 4
                      - 0.03 * abs(st - base))
                if sc > best_sc:
                    best_sc, best_st = sc, st
            # never start inside a number ("1969" -> "69")
            while (best_st > 0 and best_st < len(ids) - k
                   and pieces[best_st - 1].strip()[-1:].isdigit()
                   and pieces[best_st].strip()[:1].isdigit()):
                best_st += 1
            return ids[best_st:best_st + k]

    # -- fallback: best similarity window -----------------------------------
    q = _grams(sp.decode([int(t) for t in query_ids
                          if 0 < int(t) < 32000]))
    best, best_w = -1.0, ids[:k]
    for s in range(0, len(ids) - k + 1):
        w = ids[s:s + k]
        txt = sp.decode(w)
        sc = head_w * _jac(q, _grams(txt))
        if s == 0 or sp.decode(ids[s - 1:s]).strip().endswith((".", "?", "!")):
            sc += sentence_bonus
        if sp.id_to_piece(w[0]).startswith("\u2581"):
            sc += word_bonus
        if sc > best:
            best, best_w = sc, w
    return best_w

