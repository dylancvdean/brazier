"""Streaming ASR worker: NDJSON events on stdout.

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
import json
import sys
from pathlib import Path
from threading import Thread


def emit(payload: dict) -> None:
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


def transcribe(processor, model, audio_path: Path, lookahead: int) -> str:
    """Transcribe one file, emitting `delta` events as text arrives."""
    from transformers import TextIteratorStreamer
    from transformers.audio_utils import load_audio

    if hasattr(processor, "set_num_lookahead_tokens"):
        processor.set_num_lookahead_tokens(lookahead)

    sampling_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(str(audio_path), sampling_rate=sampling_rate)

    first_chunk_inputs = processor(
        audio[: processor.num_samples_first_audio_chunk],
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
        while (end_idx := start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            inputs = processor(
                audio[start_idx:end_idx],
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
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    emit(
        {
            "type": "status",
            "phase": "stream",
            "message": "Streaming partial transcripts",
            "latency_ms": getattr(processor, "streaming_latency_ms", None),
        }
    )

    full = []
    for text_chunk in streamer:
        if not text_chunk:
            continue
        full.append(text_chunk)
        emit({"type": "delta", "text": text_chunk})
    thread.join()
    return "".join(full).strip()


def serve(processor, model, default_lookahead: int) -> int:
    """Transcribe one file per line of JSON on stdin until it closes.

    A failed request reports itself and the worker keeps serving: one unreadable
    file is not a reason to pay the model load again.
    """
    emit({"type": "status", "phase": "ready", "message": "Worker ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as error:
            emit({"type": "error", "message": f"invalid request: {error}"})
            continue
        audio_path = Path(request.get("audio", ""))
        if not audio_path.is_file():
            emit({"type": "error", "message": f"audio file not found: {audio_path}"})
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
