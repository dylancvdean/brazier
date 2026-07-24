"""CLI entry: stream partial transcripts as NDJSON on stdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from threading import Thread


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brazier streaming ASR worker")
    parser.add_argument("--model", required=True, help="Local HF snapshot directory or repo id")
    parser.add_argument("--audio", required=True, help="Path to audio file")
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

    model_path = Path(args.model)
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        emit({"type": "error", "message": f"audio file not found: {audio_path}"})
        return 2

    try:
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer
        from transformers.audio_utils import load_audio
    except Exception as error:  # noqa: BLE001 — surface import failures to Brazier
        emit({"type": "error", "message": f"failed to import streaming ASR stack: {error}"})
        return 3

    model_id = str(model_path if model_path.exists() else args.model)
    emit({"type": "status", "phase": "load", "message": f"Loading {model_id}"})

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        device_map = None if args.device == "cpu" else args.device
        model = AutoModelForRNNT.from_pretrained(model_id, device_map=device_map)
        if args.device == "cpu":
            model = model.to("cpu")

        if hasattr(processor, "set_num_lookahead_tokens"):
            processor.set_num_lookahead_tokens(args.lookahead)

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

        latency = getattr(processor, "streaming_latency_ms", None)
        emit(
            {
                "type": "status",
                "phase": "stream",
                "message": "Streaming partial transcripts",
                "latency_ms": latency,
            }
        )

        full = []
        for text_chunk in streamer:
            if not text_chunk:
                continue
            full.append(text_chunk)
            emit({"type": "delta", "text": text_chunk})
        thread.join()

        text = "".join(full).strip()
        emit({"type": "done", "text": text})
        return 0
    except Exception as error:  # noqa: BLE001
        emit({"type": "error", "message": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
