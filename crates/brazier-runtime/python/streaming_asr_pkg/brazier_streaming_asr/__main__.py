"""Streaming ASR worker emitting NDJSON events on stdout.

Two modes, one transcription path:

* one-shot (`--audio`) transcribes a file and exits, which is what a person
  debugging the environment by hand wants;
* serve (`--serve`) loads the model once and then transcribes a file per line of
  JSON on stdin, which is what a conversation needs. Loading Nemotron costs
  around three seconds, and paying that per utterance made every spoken turn
  wait for a model that was already on disk a moment earlier.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from threading import Condition, Lock, Thread


_emit_lock = Lock()


def emit(payload: dict) -> None:
    # A live request is decoded on a worker thread while the main thread keeps
    # accepting audio. Keep their NDJSON records from interleaving.
    with _emit_lock:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def load_model(model_id: str, device: str):
    """Load the processor and model once. Raises on failure."""
    from transformers import AutoModelForRNNT, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    device_map = None if device == "cpu" else device
    model = AutoModelForRNNT.from_pretrained(model_id, device_map=device_map)
    if device == "cpu":
        model = model.to("cpu")
    return processor, model


class LivePcm:
    """A blocking, append-only PCM source consumed by the feature generator."""

    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self._pcm = bytearray()
        self._finished = False
        self._cancelled = False
        self._changed = Condition()

    def append(self, encoded: str) -> None:
        chunk = base64.b64decode(encoded, validate=True)
        if len(chunk) % 2:
            raise ValueError("pcm_s16le audio must contain complete samples")
        with self._changed:
            if self._finished:
                raise ValueError("audio arrived after finish")
            self._pcm.extend(chunk)
            self._changed.notify_all()

    def finish(self) -> None:
        with self._changed:
            self._finished = True
            self._changed.notify_all()

    def cancel(self) -> None:
        with self._changed:
            self._cancelled = True
            self._finished = True
            self._changed.notify_all()

    def window_bytes(self, start: int, end: int):
        """Wait for [start, end), returning padded little-endian PCM bytes."""
        with self._changed:
            while len(self._pcm) // 2 < end and not self._finished:
                self._changed.wait()
            if self._cancelled:
                raise RuntimeError("streaming transcription cancelled")
            available = len(self._pcm) // 2
            if start >= available:
                return None
            stop = min(end, available)
            raw = bytes(self._pcm[start * 2 : stop * 2])
            if stop < end:
                raw += bytes((end - stop) * 2)
            return raw

    def window(self, start: int, end: int):
        """Convert a blocking PCM byte window into normalized float samples."""
        import numpy as np

        raw = self.window_bytes(start, end)
        if raw is None:
            return None
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class ArrayPcm:
    """The same window interface for the one-shot/file compatibility path."""

    def __init__(self, samples, sampling_rate: int):
        self.samples = samples
        self.sampling_rate = sampling_rate

    def window(self, start: int, end: int):
        import numpy as np

        if start >= self.samples.shape[0]:
            return None
        chunk = self.samples[start:end]
        if chunk.shape[0] < end - start:
            chunk = np.pad(chunk, (0, end - start))
        return chunk


def transcribe_source(processor, model, source, lookahead: int, request_id=None) -> str:
    """Transcribe an append-only source, emitting deltas while audio arrives."""
    from transformers import TextIteratorStreamer

    if hasattr(processor, "set_num_lookahead_tokens"):
        processor.set_num_lookahead_tokens(lookahead)

    sampling_rate = processor.feature_extractor.sampling_rate
    if source.sampling_rate != sampling_rate:
        raise ValueError(
            f"PCM sample rate {source.sampling_rate} does not match model rate {sampling_rate}"
        )

    first_audio = source.window(0, processor.num_samples_first_audio_chunk)
    if first_audio is None:
        raise ValueError("streaming transcription received no audio")

    first_chunk_inputs = processor(
        first_audio,
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
        return_tensors="pt",
    )
    first_chunk_inputs = first_chunk_inputs.to(model.device, dtype=model.dtype)

    def input_features_generator():
        yield first_chunk_inputs.input_features[
            :, : processor.num_mel_frames_first_audio_chunk, :
        ]
        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        start_idx = mel_frame_idx * hop_length - n_fft // 2
        while True:
            end_idx = start_idx + processor.num_samples_per_audio_chunk
            audio = source.window(start_idx, end_idx)
            if audio is None:
                break
            inputs = processor(
                audio,
                sampling_rate=sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            yield inputs.input_features
            mel_frame_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    generate_kwargs = {
        **first_chunk_inputs,
        "input_features": input_features_generator(),
        "streamer": streamer,
    }
    generation_error = []

    def generate() -> None:
        try:
            model.generate(**generate_kwargs)
        except BaseException as error:  # noqa: BLE001 — cross the thread boundary below
            generation_error.append(error)
            # TextIteratorStreamer otherwise blocks its consumer forever when
            # generation fails before placing the normal stop marker.
            streamer.end()

    thread = Thread(target=generate)
    thread.start()

    emit(
        {
            "type": "status",
            "phase": "stream",
            "message": "Streaming partial transcripts",
            "latency_ms": getattr(processor, "streaming_latency_ms", None),
            **({"id": request_id} if request_id else {}),
        }
    )

    full = []
    for text_chunk in streamer:
        if not text_chunk:
            continue
        full.append(text_chunk)
        emit(
            {
                "type": "delta",
                "text": text_chunk,
                **({"id": request_id} if request_id else {}),
            }
        )
    thread.join()
    if generation_error:
        raise generation_error[0]
    return "".join(full).strip()


def transcribe(processor, model, audio_path: Path, lookahead: int, request_id=None) -> str:
    """Transcribe one file through the same incremental feature path."""
    from transformers.audio_utils import load_audio

    sampling_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(str(audio_path), sampling_rate=sampling_rate)
    return transcribe_source(
        processor,
        model,
        ArrayPcm(audio, sampling_rate),
        lookahead,
        request_id,
    )


def serve(processor, model, default_lookahead: int) -> int:
    """Serve file requests and one incrementally fed live request at a time.

    A failed request reports itself and the worker keeps serving: one unreadable
    file is not a reason to pay the model load again.
    """
    emit({"type": "status", "phase": "ready", "message": "Worker ready"})
    active = {"id": None, "source": None, "thread": None}
    active_lock = Lock()

    def live_runner(request_id: str, source: LivePcm, lookahead: int) -> None:
        try:
            text = transcribe_source(processor, model, source, lookahead, request_id)
            emit({"type": "done", "id": request_id, "text": text})
        except Exception as error:  # noqa: BLE001 — report without killing the worker
            emit({"type": "error", "id": request_id, "message": str(error)})
        finally:
            with active_lock:
                if active["id"] == request_id:
                    active.update(id=None, source=None, thread=None)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as error:
            emit({"type": "error", "message": f"invalid request: {error}"})
            continue
        kind = request.get("type")
        request_id = str(request.get("id", ""))
        if kind == "start":
            with active_lock:
                if active["id"] is not None:
                    emit({"type": "error", "id": request_id, "message": "worker is busy"})
                    continue
                if not request_id:
                    emit({"type": "error", "message": "stream start requires an id"})
                    continue
                source = LivePcm(int(request.get("sample_rate", 0)))
                thread = Thread(
                    target=live_runner,
                    args=(request_id, source, int(request.get("lookahead", default_lookahead))),
                    daemon=True,
                )
                active.update(id=request_id, source=source, thread=thread)
                thread.start()
            continue
        if kind in ("audio", "finish", "cancel"):
            with active_lock:
                source = active["source"] if active["id"] == request_id else None
            if source is None:
                emit({"type": "error", "id": request_id, "message": "stream is not active"})
                continue
            try:
                if kind == "audio":
                    source.append(str(request.get("data", "")))
                elif kind == "finish":
                    source.finish()
                else:
                    source.cancel()
            except Exception as error:  # noqa: BLE001
                emit({"type": "error", "id": request_id, "message": str(error)})
            continue

        audio_path = Path(request.get("audio", ""))
        if not audio_path.is_file():
            emit({"type": "error", "message": f"audio file not found: {audio_path}"})
            continue
        with active_lock:
            busy = active["id"] is not None
        if busy:
            emit({"type": "error", "message": "worker is busy"})
            continue
        try:
            text = transcribe(
                processor, model, audio_path, int(request.get("lookahead", default_lookahead))
            )
            emit({"type": "done", "text": text})
        except Exception as error:  # noqa: BLE001 — a bad request must not end the worker
            emit({"type": "error", "message": str(error)})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brazier streaming ASR worker")
    parser.add_argument("--model", required=True, help="Local HF snapshot directory or repo id")
    parser.add_argument("--audio", help="Path to audio file (one-shot mode)")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Load once, then read one JSON request per line of stdin",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=6,
        help="num_lookahead_tokens (latency trade-off; default 6 ≈ 560ms)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device map (auto|cpu|cuda|mps)",
    )
    args = parser.parse_args(argv)

    if not args.serve and not args.audio:
        emit({"type": "error", "message": "pass --audio or --serve"})
        return 2

    model_path = Path(args.model)
    model_id = str(model_path if model_path.exists() else args.model)
    emit({"type": "status", "phase": "load", "message": f"Loading {model_id}"})

    try:
        processor, model = load_model(model_id, args.device)
    except Exception as error:  # noqa: BLE001 — surface import and load failures
        emit({"type": "error", "message": f"failed to load streaming ASR model: {error}"})
        return 3

    if args.serve:
        return serve(processor, model, args.lookahead)

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        emit({"type": "error", "message": f"audio file not found: {audio_path}"})
        return 2
    try:
        emit({"type": "done", "text": transcribe(processor, model, audio_path, args.lookahead)})
        return 0
    except Exception as error:  # noqa: BLE001
        emit({"type": "error", "message": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
