# fix_rag_server.py — make server_student_rag.py FULL DUPLEX.
#
# Why: the "listening mode" force-feeds (silence, pad) into the model's own
# stream while the user speaks and lets an external VAD decide when the model
# may talk.  The student was trained FULL DUPLEX (it always generates and owns
# its turn timing, exactly like the working server_student.py), so the forced
# silence + VAD switch leaves it stuck emitting pads — no greeting, no answer.
#
# This script replaces the nested `run_frame` with a full-duplex version:
# the model generates every frame; STT/VAD only *triggers retrieval*.
import shutil
import sys

# default: the pod path; pass a local path as argv[1] to patch the repo copy
PATH = (sys.argv[1] if len(sys.argv) > 1
        else "/workspace/s_new_student_m_rag_v1/moshi_local/moshi_distill/server_student_rag.py")

NEW_RUN_FRAME = '''            def run_frame(chunk_tensor):
                """One 80 ms frame, FULL DUPLEX.

                The model generates every frame and keeps its own turn timing
                (exactly as in training and in server_student.py); the STT/VAD
                only *triggers retrieval* on end of speech — it never forces
                the model silent.  Returns (pcm, kind, meta).
                """
                nonlocal lookup_done, ref_vecs, ref_info
                nonlocal current_transcript, stt_token_buffer, stt_in_utterance
                nonlocal stt_silence_frame_count, stt_last_vad_end
                nonlocal barge_in_frame_count, needs_prompt_replay

                input_codes = self.mimi.encode(chunk_tensor)
                stt_codes = self.stt_mimi.encode(chunk_tensor)
                stt_tokens, vad_heads = self.stt_lm_gen.step_with_extra_heads(stt_codes)
                vad_score = (float(vad_heads[2][0, 0, 0].cpu().item())
                             if vad_heads and len(vad_heads) > 2 else 0.0)

                # ---- STT transcript -> retrieval trigger (never mutes the model)
                fired = None
                if stt_tokens is not None:
                    stt_token_buffer.append(stt_tokens[:, :1, :].cpu())
                    if stt_tokens[0, 0, 0].item() not in (0, 3):
                        stt_in_utterance = True
                        stt_last_vad_end = False
                if vad_score > self.vad_threshold:
                    stt_silence_frame_count += 1
                else:
                    stt_silence_frame_count = 0
                vad_fired = (stt_silence_frame_count >= VAD_SILENCE_FRAMES_REQUIRED
                             and not stt_last_vad_end)
                stt_last_vad_end = (stt_silence_frame_count
                                    >= VAD_SILENCE_FRAMES_REQUIRED)
                if vad_fired and stt_in_utterance and stt_token_buffer:
                    transcript = decode_stt_tokens(
                        stt_token_buffer, self.stt_tokenizer,
                        self.stt_padding_token_id)
                    stt_token_buffer = []
                    stt_in_utterance = False
                    stt_silence_frame_count = 0
                    if transcript.strip():
                        k, vecs, info = self.prepare_reference(transcript)
                        ref_vecs = vecs
                        ref_info = info
                        current_transcript = transcript
                        lookup_done = False
                        self._ref_triggered = False
                        self._ref_queue = []
                        self._text_queue = []
                        self._speech_run = 0
                        fired = {"transcript": transcript, "k": k, "hit": info}

                # ---- free generation (full duplex) -------------------------
                ref_vec = self._ref_queue.pop(0) if self._ref_queue else None
                forced_text = self._text_queue.pop(0) if self._text_queue else None
                tokens = self._time_step(input_codes, ref_vec, forced_text)
                if tokens is None:
                    return (np.zeros(self.frame_size, np.float32), "silence", None)

                text_token = int(tokens[0, 0, 0].item())
                with torch.no_grad():
                    main_pcm = self.mimi.decode(tokens[:, 1:9]).cpu()
                pcm = np.clip(main_pcm[0, 0].float().numpy(), -1.0, 1.0)

                # ---- barge-in: the user talks over the model's speech ------
                speech_this_frame = (stt_tokens is not None
                                     and stt_tokens[0, 0, 0].item() not in (0, 3)
                                     and vad_score < self.vad_threshold)
                model_speaking = not self._in_pause(pcm)
                barge_in_frame_count = (barge_in_frame_count + 1
                                        if (speech_this_frame and model_speaking) else 0)
                if barge_in_frame_count >= BARGE_IN_FRAMES_REQUIRED:
                    barge_in_frame_count = 0
                    reset_streams()
                    lookup_done = False
                    ref_vecs = ref_info = None
                    stt_token_buffer = []
                    stt_in_utterance = True
                    stt_last_vad_end = False
                    stt_silence_frame_count = 0
                    needs_prompt_replay = True
                    return (np.zeros(self.frame_size, np.float32), "barge_in", None)

                # ---- never let card tokens spill into the answer -----------
                if self._text_queue:
                    if self._in_pause(pcm):
                        self._speech_run = 0
                    else:
                        self._speech_run += 1
                        if self._speech_run >= 2:
                            logger.info("card truncated at speech: %d tokens dropped",
                                        len(self._text_queue))
                            self._text_queue = []
                else:
                    self._speech_run = 0

                # ---- <lookup> -> card injection ----------------------------
                if text_token == self.lookup_id and not lookup_done:
                    self._ref_triggered = True
                if self._ref_triggered and not lookup_done:
                    job = self._ref_job
                    if job is not None and not job["done"]:
                        if (self._ref_wait < self.web_wait_frames
                                and self._in_pause(pcm)):
                            if self._ref_wait == 0:
                                logger.info("web search still running: "
                                            "spending pause frames")
                            self._ref_wait += 1
                            return (pcm, "ref_wait", {"waited": self._ref_wait})
                        self._ref_job = None          # timed out -> local card
                    elif job is not None:
                        self._ref_job = None
                        if job.get("hits") and job["hits"][0].get("text"):
                            _, ref_vecs, ref_info = self.reference_from_hit(
                                job["query"], job["hits"][0])
                            logger.info("using web passage %r",
                                        job["hits"][0]["text"][:90])
                    if self._ref_wait > 0 and not self._in_pause(pcm):
                        logger.info("ref dropped: pause ended before injection")
                        ref_vecs = None
                    n_ref = (int(ref_vecs.shape[1])
                             if torch.is_tensor(ref_vecs)
                             else len(ref_vecs or []))
                    self._ref_triggered = False
                    lookup_done = True
                    if n_ref > 0:
                        if self.ref_mode == "add":
                            k = self.start_ref_add(ref_vecs)
                            return (np.zeros(self.frame_size, np.float32),
                                    "ref_inject",
                                    {"frames": 0, "k": k, "hit": ref_info})
                        if self.ref_mode == "text" and self.ref_pos == "pause":
                            k = self.start_ref_text_pause(
                                ref_vecs if not torch.is_tensor(ref_vecs) else [])
                            self._speech_run = 0
                            return (pcm, "ref_inject",
                                    {"frames": 0, "k": k, "hit": ref_info})
                        n_frames = (self.inject_ref_text(ref_vecs)
                                    if self.ref_mode == "text"
                                    else self.inject_ref_block(ref_vecs))
                        return (np.zeros(self.frame_size * n_frames, np.float32),
                                "ref_inject",
                                {"frames": n_frames, "k": n_ref, "hit": ref_info})
                if text_token == self.lookup_id:
                    # duplicate / no-reference lookup: never leak the token
                    pcm = np.zeros(self.frame_size, np.float32)

                if fired is not None:
                    return (pcm, "turn_start", fired)
                return (pcm, "audio",
                        {"text": text_token if forced_text is None else 0})

'''

START = "            def run_frame(chunk_tensor):"
END = "            try:\n                while True:"

src = open(PATH, encoding="utf-8").read()
assert src.count(START) == 1, "run_frame not found (already patched?)"
assert src.count(END) == 1, "end marker not found"

shutil.copy(PATH, PATH + ".bak_fullduplex")

head, rest = src.split(START, 1)
_old, tail = rest.split(END, 1)
new_src = head + NEW_RUN_FRAME + END + tail

# sanity: the patched file must still parse
compile(new_src, PATH, "exec")

open(PATH, "w", encoding="utf-8", newline="\n").write(new_src)
print("patched OK -> full-duplex run_frame")
print("backup at:", PATH + ".bak_fullduplex")
