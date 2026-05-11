"""Apply source-level patches LLaMA-Omni / fairseq need to run on Colab T4 (Python 3.12).

Invoked once by setup.sh. Safe to run multiple times — every patch is idempotent.

Why each patch:
  - fairseq pins old configs that fail Python 3.12's strict mutable-default check.
  - LLaMA-Omni's builder.py passes `load_in_4bit=True` AND `quantization_config`,
    which modern transformers rejects; also calls `.to(device)` which fails on
    bnb-quantized models, and `low_cpu_mem_usage=False` blows past Colab CPU RAM.
  - LLaMA-Omni builds its whisper speech encoder inside __init__, which conflicts
    with `low_cpu_mem_usage=True` (meta tensors). builder.py rebuilds it after.
  - speech_generator pins flash_attention_2 and passes a 2D padding mask; on a T4
    we have no flash_attn, and modern LLaMA layers need a 4D causal mask.
"""
import os
import re
import sys

LLAMA_OMNI = "/content/LLaMA-Omni"
FAIRSEQ = "/usr/local/lib/python3.12/dist-packages/fairseq"


def _patch_file(path, replacements):
    """Apply (old, new) tuple list to file; skip if any "new" already present."""
    with open(path) as f:
        src = f.read()
    for old, new in replacements:
        if new in src:
            continue
        src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)


def patch_fairseq():
    """Convert `: SomeConfig = SomeConfig(...)` patterns to default_factory across fairseq."""
    if not os.path.isdir(FAIRSEQ):
        return
    for root, _, files in os.walk(FAIRSEQ):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path) as f:
                src = f.read()
            original = src
            # `field(default=Config())` -> `field(default_factory=lambda: Config())`
            src = re.sub(
                r"field\s*\(\s*default\s*=\s*(\w+(?:Config|Dataclass|Settings))\s*\((.*?)\)\s*\)",
                r"field(default_factory=lambda: \1(\2))",
                src,
            )
            # `: Config = Config()` -> `field(default_factory=Config)`
            src = re.sub(
                r"(:\s*(\w+(?:Config|Dataclass|Settings)))\s*=\s*\2\s*\(\s*\)",
                r"\1 = field(default_factory=\2)",
                src,
            )
            # `: Config = Config(args)` (no nested parens) -> lambda factory
            src = re.sub(
                r"(:\s*(\w+(?:Config|Dataclass|Settings)))\s*=\s*\2\s*\(([^()]+)\)",
                lambda m: f"{m.group(1)} = field(default_factory=lambda: {m.group(2)}({m.group(3)}))",
                src,
            )
            if src != original:
                if "from dataclasses import" in src and re.search(
                    r"from dataclasses import[^\n]*\bfield\b", src
                ) is None:
                    src = re.sub(
                        r"from dataclasses import", "from dataclasses import field,", src, count=1
                    )
                elif "from dataclasses import" not in src:
                    src = "from dataclasses import field\n" + src
                with open(path, "w") as f:
                    f.write(src)

    # hydra_init reads .default which is now MISSING; switch to default_factory()
    init_path = os.path.join(FAIRSEQ, "dataclass", "initialize.py")
    if os.path.isfile(init_path):
        with open(init_path) as f:
            src = f.read()
        if "default_factory" not in src:
            src = src.replace(
                "v = FairseqConfig.__dataclass_fields__[k].default",
                "_f = FairseqConfig.__dataclass_fields__[k]\n        v = _f.default_factory() if _f.default_factory is not __import__('dataclasses').MISSING else _f.default",
            )
            with open(init_path, "w") as f:
                f.write(src)


def patch_llama_omni():
    """Patch LLaMA-Omni source to run with 4-bit quantization on transformers 4.43.4."""
    builder = os.path.join(LLAMA_OMNI, "omni_speech/model/builder.py")
    if os.path.isfile(builder):
        _patch_file(
            builder,
            [
                # Drop deprecated load_in_*bit kwargs (transformers rejects them with quantization_config)
                ("    if load_8bit:\n        kwargs['load_in_8bit'] = True\n",
                 "    if load_8bit:\n        kwargs['device_map'] = {'': 0}\n        from transformers import BitsAndBytesConfig as _BNB\n        kwargs['quantization_config'] = _BNB(load_in_8bit=True)\n"),
                ("    elif load_4bit:\n        kwargs['load_in_4bit'] = True\n        kwargs['quantization_config'] = BitsAndBytesConfig(",
                 "    elif load_4bit:\n        kwargs['device_map'] = {'': 0}\n        kwargs['quantization_config'] = BitsAndBytesConfig("),
                # Avoid CPU RAM blowup during 4-bit init
                ("low_cpu_mem_usage=False", "low_cpu_mem_usage=True"),
                # Don't .to(device) a 4-bit model after from_pretrained
                ("        )\n        model = model.to(device=device)\n\n    model.get_model().speech_encoder",
                 "        )\n        if not (load_4bit or load_8bit):\n            model = model.to(device=device)\n\n    model.get_model().speech_encoder"),
            ],
        )

    arch = os.path.join(LLAMA_OMNI, "omni_speech/model/omni_speech_arch.py")
    if os.path.isfile(arch):
        _patch_file(
            arch,
            [
                # speech_encoder build conflicts with low_cpu_mem_usage; builder.py rebuilds it after
                ('if hasattr(config, "speech_encoder"):\n            self.speech_encoder = build_speech_encoder(config)\n            self.speech_projector = build_speech_projector(config)',
                 'if hasattr(config, "speech_encoder"):\n            # speech_encoder is rebuilt after from_pretrained (see builder.py) to avoid\n            # meta-tensor conflicts with low_cpu_mem_usage=True.\n            self.speech_projector = build_speech_projector(config)'),
            ],
        )

    sg = os.path.join(LLAMA_OMNI, "omni_speech/model/speech_generator/speech_generator.py")
    if os.path.isfile(sg):
        with open(sg) as f:
            src = f.read()
        # Use sdpa instead of flash_attention_2 (T4 has no flash_attn)
        src = src.replace(
            '_config._attn_implementation = "flash_attention_2"',
            '_config._attn_implementation = "sdpa"',
        )
        # Expand 2D padding mask to 4D causal mask in predict()
        old_predict = '''    def predict(self, tgt_reps):
        hidden_states, attention_mask, position_ids = self.upsample([tgt_reps])
        hidden_states = self.input_proj(hidden_states)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )'''
        new_predict = '''    def predict(self, tgt_reps):
        hidden_states, attention_mask, position_ids = self.upsample([tgt_reps])
        hidden_states = self.input_proj(hidden_states)
        import torch as _t
        seq_len = hidden_states.shape[1]
        _causal = _t.triu(_t.full((seq_len, seq_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype), diagonal=1)
        _pad = (~attention_mask)[:, None, None, :].to(hidden_states.dtype) * float("-inf")
        _pad = _t.nan_to_num(_pad, nan=0.0)
        _attn_4d = _causal[None, None, :, :] + _pad
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=_attn_4d,
                position_ids=position_ids,
            )'''
        if new_predict not in src:
            src = src.replace(old_predict, new_predict)
        with open(sg, "w") as f:
            f.write(src)


if __name__ == "__main__":
    patch_fairseq()
    patch_llama_omni()
    print("[Model B] Patches applied.")
