# ======================================================
# 🔧 PYTHON PATH FIX (REQUIRED FOR src/ LAYOUT)
# ======================================================
import os, sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ======================================================
# 🔥 CRITICAL TORCH + ATTENTION FIXES (DO NOT TOUCH)
# ======================================================
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.set_float32_matmul_precision("high")

# ======================================================
# 🔒 FORCE DIFFUSERS → EAGER ATTENTION
# ======================================================
import diffusers
from diffusers.models.attention_processor import AttnProcessor

diffusers.models.attention_processor.DEFAULT_ATTENTION_PROCESSOR = AttnProcessor()
diffusers.models.attention_processor.AttnProcessor2_0 = AttnProcessor

# ======================================================
# NORMAL IMPORTS
# ======================================================
import random
import numpy as np
import gradio as gr
import spaces
from scipy.io.wavfile import write as write_wav

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# ======================================================
# DEVICE (Patched for Mac M1/M2/M3)
# ======================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"🚀 Running on device: {DEVICE}")


MODEL = None

# ======================================================
# OUTPUT FOLDER (ADDED)
# ======================================================
OUTPUT_DIR = "AudioOutput"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# DEMO PROMPTS
# ======================================================
LANGUAGE_CONFIG = {
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
}

# ======================================================
# HELPERS
# ======================================================
def default_audio_for_ui(lang):
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")

def default_text_for_ui(lang):
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")

def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("🔄 Loading Chatterbox model...")
        MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        if hasattr(MODEL, "to"):
            MODEL.to(torch.device(DEVICE))
        print("✅ Model loaded successfully")
    return MODEL

def set_seed(seed):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# ======================================================
# 🔥 STRICT 300 CHARACTER SPLIT (REPLACED)
# ======================================================
def chunk_text(text, chunk_size=300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ======================================================
# TTS FUNCTION
# ======================================================
@spaces.GPU
def generate_tts_audio(
    text_input,
    language_id,
    audio_prompt_path_input=None,
    exaggeration_input=0.5,
    temperature_input=0.8,
    seed_num_input=0,
    cfgw_input=0.5,
):
    model = get_or_load_model()

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }

    prompt = audio_prompt_path_input or default_audio_for_ui(language_id)
    if prompt:
        kwargs["audio_prompt_path"] = prompt

    chunks = chunk_text(text_input, 300)
    print(f"🧩 Split into {len(chunks)} chunk(s)")

    audio_out = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"🎤 Generating chunk {i}/{len(chunks)}")

        wav = model.generate(chunk, language_id=language_id, **kwargs)
        audio_np = wav.squeeze(0).cpu().numpy()

        # 💾 SAVE EACH AUDIO FILE (ADDED)
        file_path = os.path.join(OUTPUT_DIR, f"hindi_part_{i:03d}.wav")
        write_wav(file_path, model.sr, audio_np)
        print(f"✅ Saved: {file_path}")

        audio_out.append(audio_np)

    final_audio = np.concatenate(audio_out)
    return (model.sr, final_audio)

# ======================================================
# GRADIO UI
# ======================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 Chatterbox Multilingual TTS")

    lang_default = "fr"

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value=default_text_for_ui(lang_default),
                label="Text",
                max_lines=10,
            )

            language_id = gr.Dropdown(
                choices=list(SUPPORTED_LANGUAGES.keys()),
                value=lang_default,
                label="Language",
            )

            ref_audio = gr.Audio(
                type="filepath",
                label="Reference Audio (optional)",
                value=default_audio_for_ui(lang_default),
            )

            exaggeration = gr.Slider(0.25, 2.0, step=0.05, value=0.5, label="Exaggeration")
            cfg_weight = gr.Slider(0.2, 1.0, step=0.05, value=0.5, label="CFG Weight")
            temperature = gr.Slider(0.05, 5.0, step=0.05, value=0.8, label="Temperature")
            seed = gr.Number(value=0, label="Seed (0 = random)")

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_out = gr.Audio(label="Output Audio")

    run_btn.click(
        generate_tts_audio,
        inputs=[text, language_id, ref_audio, exaggeration, temperature, seed, cfg_weight],
        outputs=[audio_out],
    )

# 🚫 MCP DISABLED (prevents crash)
demo.launch(share=True)