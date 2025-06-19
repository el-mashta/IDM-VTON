import sys
import os

# Add the parent directory to the Python path so IDM-VTON can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'IDM-VTON')))

from huggingface_hub import snapshot_download

print("Starting model download and caching...")
print(f"Downloading to HF cache: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")

try:
    # Download the main IDM-VTON models
    print("Downloading IDM-VTON models...")
    snapshot_download(
        repo_id='yisol/IDM-VTON',
        cache_dir=None,
        local_files_only=False,
        resume_download=True
    )
    print("✅ IDM-VTON models downloaded successfully")

    # Download the DressCode-specific models for bottom half try-on
    print("Downloading IDM-VTON-DC models (for DressCode categories)...")
    snapshot_download(
        repo_id='yisol/IDM-VTON-DC',
        cache_dir=None,
        local_files_only=False,
        resume_download=True
    )
    print("✅ IDM-VTON-DC models downloaded successfully")

except Exception as e:
    print(f"❌ Error downloading models: {e}")
    sys.exit(1)

print("All models successfully downloaded and cached.")
print("Models are ready for use - no GPU memory used during download.")