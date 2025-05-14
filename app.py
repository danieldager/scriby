"""
Minimal local-hosted web UI for French Distil-Whisper
----------------------------------------------------
* Uses faster-whisper + CTranslate2 int8 checkpoint
* Serves a Gradio interface at http://localhost:7860

python -m venv venv && source venv/bin/activate
pip install faster-whisper gradio numpy
python app.py   # then open http://localhost:7860

"""

import datetime as _dt
import itertools
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import gradio as gr
from faster_whisper import WhisperModel

# ────────────────────────── model loading ──────────────────────────
PROJECT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "fr-distil-v3-ct2-int8",
    "ctranslate2",
)

OUTPUT_DIR = Path(PROJECT_DIR) / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

_name_counter = itertools.count(1)  # transcription1, transcription2, ...

model = WhisperModel(
    MODEL_DIR,
    device="cpu",
    compute_type="int8",  # ~2 GB RAM on CPU
    cpu_threads=os.cpu_count() or 1,
    num_workers=4,
)


# ───────────────────────── transcription fn ─────────────────────────


def _sec_to_ts(sec: float) -> str:
    td = _dt.timedelta(seconds=sec)
    total_ms = int(td.total_seconds() * 1000)
    hh, remainder = divmod(total_ms, 3600_000)
    mm, remainder = divmod(remainder, 60_000)
    ss, ms = divmod(remainder, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def _sec_to_hms(sec: float) -> str:
    td = _dt.timedelta(seconds=int(sec))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ────────────────────── helpers ──────────────────────
def _get_duration(path: str) -> float | None:
    """Return audio/video duration in seconds via ffprobe or None if unknown."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(proc.stdout.strip())
    except Exception:
        return None


def _write_srt(segments, dest: Path) -> None:
    lines = []
    for idx, seg in enumerate(segments, 1):
        lines += [
            str(idx),
            f"{_sec_to_ts(seg.start)} --> {_sec_to_ts(seg.end)}",
            seg.text.strip(),
            "",
        ]
    dest.write_text("\n".join(lines), encoding="utf-8")


# ────────────────────── preview helper ──────────────────────
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}


def _extract_audio_for_preview(path: str) -> str | None:
    """
    Return a path to an audio file suitable for Gradio playback.
    * If the input is already audio, return it unchanged.
    * If it's video (e.g. .mp4), extract audio to a temp WAV
      using ffmpeg and return that path.
    * Return None if extraction fails.
    """
    ext = Path(path).suffix.lower()
    if ext in AUDIO_EXTS:
        return path

    # Assume video; extract 128‑kbps mp3
    try:
        tmpdir = Path(tempfile.mkdtemp())
        out_path = tmpdir / "audio_preview.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                path,
                "-vn",
                "-acodec",
                "libmp3lame",
                "-ab",
                "128k",
                out_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return str(out_path)
    except Exception:
        return None


# ───────────────────────── main API ─────────────────────────
def transcribe(file_path: str, desired_name: str) -> tuple[str, float | None, str]:
    """
    Kick off transcription in a background thread.
    Returns an immediate message with estimated processing time and save path.
    """
    start_ts = _dt.datetime.now(_dt.timezone.utc).timestamp()

    if not file_path:
        return "➡️  Sélectionnez un fichier, puis cliquez sur « Transcrire ».", None, ""

    duration = _get_duration(file_path)  # may be None

    # crude runtime factor: GPU≈0.3, CPU≈1.2
    est_seconds = int(duration * 1.2) if duration else 0
    est = _sec_to_hms(est_seconds) if est_seconds else "quelques instants"

    # output filename
    if desired_name.strip():
        base = Path(desired_name).stem
    else:
        base = f"transcription{next(_name_counter)}"
    out_path = OUTPUT_DIR / f"{base}.srt"

    def _job():
        segments, _ = model.transcribe(
            file_path,
            beam_size=5,
            language="fr",
            chunk_length=None,  # fastest
        )
        _write_srt(segments, out_path)

    threading.Thread(target=_job, daemon=True).start()

    return (
        f"⏳ Transcription lancée (durée estimée : {est}). "
        f"Vous pouvez fermer la page ; le fichier sera enregistré dans "
        f"`{out_path}`.",
        start_ts,
        str(out_path),
    )


# ────────────────────────────── UI  ─────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown(
        "## Whisper FR – Distil-Large-v3 (local)\n"
        "Tout se passe sur votre ordinateur."
    )

    file_input = gr.File(
        label="Audio ou vidéo (WAV, MP3, MP4…)",
        file_types=["audio", "video"],
    )
    name_box = gr.Textbox(
        label="Nom du fichier de sortie (sans extension)",
        value=f"transcription{next(_name_counter)}",
    )
    audio_player = gr.Audio(label="Pré-écoute", interactive=False)
    trans_btn = gr.Button("Transcrire")
    transcript = gr.Textbox(
        label="Transcription (SRT)",
        lines=14,
        max_lines=40,
        autoscroll=True,
    )
    elapsed_md = gr.Markdown("")
    elapsed_state = gr.State(None)
    out_path_state = gr.State("")

    trans_btn.click(
        transcribe,
        inputs=[file_input, name_box],
        outputs=[transcript, elapsed_state, out_path_state],
    )

    def _update_preview(path):
        return _extract_audio_for_preview(path)

    file_input.change(
        _update_preview,
        inputs=file_input,
        outputs=audio_player,
    )

    def _tick(start_ts, out_path, current_txt):
        if start_ts is None:
            return "", current_txt

        elapsed = _dt.datetime.now(_dt.timezone.utc).timestamp() - start_ts
        elapsed_str = f"⏱ Temps écoulé : {_sec_to_hms(elapsed)}"

        # If transcription finished and not yet shown, load it
        if out_path and Path(out_path).exists() and current_txt.startswith("⏳"):
            srt_text = Path(out_path).read_text(encoding="utf-8")
            return elapsed_str, srt_text
        return elapsed_str, current_txt

    timer = gr.Timer(1.0)
    timer.tick(
        _tick,
        inputs=[elapsed_state, out_path_state, transcript],
        outputs=[elapsed_md, transcript],
    )

if __name__ == "__main__":
    demo.launch()
