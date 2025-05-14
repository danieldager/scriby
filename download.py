from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bofenghuang/whisper-large-v3-french-distil-dec16",
    local_dir="models/fr-distil-v3-ct2-int8",
    allow_patterns="ctranslate2/*",
)
