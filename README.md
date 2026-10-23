---
library_name: moshi
tags:
  - speech
  - audio
  - text-to-speech
  - retrieval-augmented-generation
  - moshi
---

# Moshi 12.5 Hz RAG student — S28

A 12.5 Hz pruned [Moshi](https://github.com/kyutai-labs/moshi) student that
answers **from a retrieved passage** instead of from its prior knowledge. The
passage is written as a 16-token text card into the pause the model already
produces before its answer, so retrieval adds **zero frames, zero silence and
zero latency** — the audio is the model's own pause, not inserted padding.

* base: `nvidia/personaplex-7b-v1` (gated) → distilled/pruned to 12.5 Hz,
  then trained with RAG (S20 backbone + S28 RAG phase, step 1800)
* text vocabulary gains: `<ref_start>` 32000, `<ref_end>` 32001, `<lookup>`
  32002
* reference mode: `text` (the card is the passage's own tokens, not latents),
  position `pause`
* card: contiguous 16-token window of the retrieved passage, selected by
  4-gram similarity to the question with an anchor on the question's rarest
  content word (`rag_ref_select.select_window_ids`)

## Files

| path | what |
|---|---|
| `ckpt/rag_S28/step_1800/student.pt` | the student (11 GB, fp32, loads as bf16) |
| `tokenizer.model` | SentencePiece text tokenizer (32003 ids) |
| `rag_index_dense/` | local fallback index: 4998 passages + MiniLM embeddings |
| `moshi_s28_server.zip` | server code + scripts + index + tokenizer (run anywhere) |
| `setup_env.sh` | creates the Python env (torch 2.5.1/cu121 + deps) |
| `download_assets.sh` | pulls this repo + the gated base weights |
| `serve_s28_web.sh` | starts the server (web search + local fallback) |

## Quick start (new machine)

```bash
unzip moshi_s28_server.zip && cd moshi_s28_server
bash setup_env.sh                    # python env -> /workspace/menv
HF_TOKEN=hf_... bash download_assets.sh
HF_TOKEN=hf_... bash serve_s28_web.sh
# open http://<host>:8998
```

`serve_s28_web.sh` reads `HF_TOKEN` only to resolve the gated base weights; the
built-in Tavily key covers web search (override with `TAVILY_API_KEY`).

## Hardware

One 80 GB GPU is comfortable (15.9 GB used with the local index loaded;
`RETRIEVER=web` drops the sentence-transformer + cross-encoder). The frame loop
is 80 ms, so any GPU that keeps a 12.5 Hz Moshi step under ~70 ms works.

## Limits

* The card budget is 16 tokens, so a very long sentence can push the answer
  word past the window; answers are best for single-fact questions.
* The training answers are ASR paraphrases, so exact proper-noun spelling can
  drift.
* If no passage is retrieved the model answers from general knowledge.

## Licence

Derived from `nvidia/personaplex-7b-v1`; use is subject to that model's
licence. The retrieval index is built from public encyclopedic text.
