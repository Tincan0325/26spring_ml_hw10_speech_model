"""Model B inference entry point.

Usage:
    from model_b.run import run_model_b
    result = run_model_b("input.wav")
"""
import contextlib
import logging
import os
import warnings

# Silence framework chatter — must run BEFORE transformers / desta are imported.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
for _name in (
    "transformers", "huggingface_hub", "accelerate", "desta",
    "peft", "librosa",
):
    logging.getLogger(_name).setLevel(logging.ERROR)


@contextlib.contextmanager
def _silenced():
    """Suppress stdout/stderr at the OS level — covers C-extension prints too."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


import numpy as _np
import torch

# Silero-VAD (used by DeSTA) references np.sctypes which was removed in NumPy 2.0.
if not hasattr(_np, 'sctypes'):
    _np.sctypes = {
        'int':     [_np.int8, _np.int16, _np.int32, _np.int64],
        'uint':    [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
        'float':   [_np.float16, _np.float32, _np.float64],
        'complex': [_np.complex64, _np.complex128],
        'others':  [bool, object, bytes, str],
    }

# lulutils (DeSTA dependency) does `from transformers.utils import download_url`,
# removed in transformers >= 4.46. Patch it back before DeSTA or model_a loads transformers.
import transformers.utils as _tu
if not hasattr(_tu, 'download_url'):
    import urllib.request as _urlreq
    def _download_url(url, root, filename=None, md5=None):
        import os as _os
        _os.makedirs(root, exist_ok=True)
        if filename is None:
            filename = url.split('/')[-1]
        fpath = _os.path.join(root, filename)
        _urlreq.urlretrieve(url, fpath)
        return fpath
    _tu.download_url = _download_url

# Module-level cache — loaded once, reused for every call in the loop.
_desta_model = None


def _patch_quantized_to():
    """Allow accelerate's dispatch_model to call .to(device) on quantized models.

    No-op if the model is not quantized; kept for API symmetry with model_c.
    """
    import transformers.modeling_utils as _mu

    if getattr(_mu.PreTrainedModel.to, "_patched_for_quantized", False):
        return
    _orig_to = _mu.PreTrainedModel.to

    def _patched_to(self, *args, **kwargs):
        if getattr(self, "is_quantized", False) or getattr(self, "is_loaded_in_4bit", False) or getattr(self, "is_loaded_in_8bit", False):
            return self
        return _orig_to(self, *args, **kwargs)

    _patched_to._patched_for_quantized = True
    _mu.PreTrainedModel.to = _patched_to


def _run_inference(
    audio_path: str,
    instruction: str,
) -> dict:
    global _desta_model
    _patch_quantized_to()

    with _silenced():
        if _desta_model is None:
            import sys as _sys
            if '/content/DeSTA2.5-Audio' not in _sys.path:
                _sys.path.insert(0, '/content/DeSTA2.5-Audio')
            from desta import DeSTA25AudioModel
            _desta_model = DeSTA25AudioModel.from_pretrained(
                "wilzzzz/DeSTA3-Llama-3.2-3B-distill",
            )
            _desta_model.to("cuda")

        messages = [
            {
                "role": "system",
                "content": "You are an assistant good to classify the gender in male and female. The input audio has a question and related audio data. Focus on the audio data to answer the question.",
            },
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{instruction}",
                "audios": [{"audio": audio_path, "text": None}],
            },
        ]

        outputs = _desta_model.generate(
            messages=messages,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=512,
        )
        response_text = outputs.text

    return {"response": response_text}


def run_model_b(
    audio_path: str,
    instruction: str = "Please classify the gender in the following audio clip.",
) -> dict:
    """Run Model B on an audio file.

    Args:
        audio_path: Path to the input WAV/MP3 file.
        instruction: Text instruction passed to the model alongside the audio.

    Returns:
        Dict with keys: response.
    """
    return _run_inference(audio_path, instruction)
