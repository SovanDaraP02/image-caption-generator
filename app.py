"""Streamlit demo app. Run locally with:
    streamlit run app.py

Deploy on Hugging Face Spaces (see DEPLOY_SPACES.md) once you have
best_checkpoint.pth -- a live link is what you hand a reviewer so they
can try their own image, not just read the code.
"""

import base64
import io
import os

import anthropic
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.caption_model import CaptionModel
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCNN

BLIP_CHECKPOINT = "Salesforce/blip-image-captioning-large"
BLIP2_CHECKPOINT = "Salesforce/blip2-opt-2.7b"
CLAUDE_MODEL = "claude-sonnet-5"
DETAILED_DESCRIPTION_PROMPT = (
    "Describe this image in one plain, natural paragraph, like you're telling a "
    "friend what's in the photo. Mention the people, objects, and setting you "
    "actually see, roughly where they are in the frame, and any colors or "
    "materials that stand out -- but only if they'd naturally come up, not as a "
    "checklist. Write in plain, ordinary language.\n\n"
    "Avoid AI-sounding filler: don't open with 'This image shows/features/"
    "captures', don't use words like 'vibrant', 'nestled', 'boasts', 'a testament "
    "to', or end with a summarizing 'overall' sentence. Just describe what's "
    "there, the way a person would.\n\n"
    "Only state what is actually visible -- do not guess at things you can't see, "
    "and do not invent people or objects that aren't present."
)


@st.cache_resource
def load_model(checkpoint_path: str = "best_checkpoint.pth") -> CaptionModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    vocab = Vocabulary()
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = checkpoint["vocab_idx2word"]

    encoder = EncoderCNN(fine_tune=False)
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder = DecoderWithAttention(vocab_size=len(vocab))
    decoder.load_state_dict(checkpoint["decoder_state"])

    return CaptionModel(encoder, decoder, vocab, device="cpu")


@st.cache_resource
def load_blip() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    processor = BlipProcessor.from_pretrained(BLIP_CHECKPOINT)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_CHECKPOINT)
    model.eval()
    return processor, model


def caption_image_blip(processor: BlipProcessor, model: BlipForConditionalGeneration,
                        image: Image.Image) -> str:
    inputs = processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            min_new_tokens=12,
            num_beams=5,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            length_penalty=1.4,
        )
    return processor.decode(output_ids[0], skip_special_tokens=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_blip2() -> tuple[Blip2Processor, Blip2ForConditionalGeneration, torch.device]:
    device = get_device()
    # float16 on GPU/MPS: halves memory footprint (~11GB -> ~5.5GB) and is
    # standard practice for inference -- output quality is not meaningfully
    # affected. CPU-only stays float32 since many CPU kernels don't support
    # fp16 well (would be slower, not faster, there).
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    processor = Blip2Processor.from_pretrained(BLIP2_CHECKPOINT)
    model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_CHECKPOINT, torch_dtype=dtype)
    model.eval()
    model.to(device)
    return processor, model, device


def caption_image_blip2(processor: Blip2Processor, model: Blip2ForConditionalGeneration,
                         device: torch.device, image: Image.Image) -> str:
    inputs = processor(image.convert("RGB"), return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=60, num_beams=3)
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def caption_image(model: CaptionModel, image: Image.Image, decoding_mode: str) -> str:
    image_tensor = preprocess(image)
    if decoding_mode == "Greedy":
        caption, _ = model.generate_greedy(image_tensor)
        return caption
    return model.generate_beam(image_tensor, beam_width=3)


def caption_image_claude(client: anthropic.Anthropic, image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    image_b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": DETAILED_DESCRIPTION_PROMPT},
            ],
        }],
    )
    return response.content[0].text


st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️")
st.title("🖼️ Multimodal Image Caption Generator")

BACKEND_CUSTOM = "🎓 My trained model (ResNet + attention + LSTM, built from scratch)"
BACKEND_BLIP = "BLIP (external pretrained model, reference only)"
BACKEND_BLIP2 = "BLIP-2 (external pretrained model, richer but slow on CPU, reference only)"
BACKEND_CLAUDE = "Claude (external pretrained model, most detailed, reference only)"

# PUBLIC_DEMO=true (set as a Space variable on the deployed public link) hides
# BLIP-2 and Claude: BLIP-2 is unusably slow on free-tier CPU-only hosting
# (no GPU means no fp16 speedup), and Claude costs API credits per request
# from every visitor. Both stay available for local use, where neither
# limitation applies.
if os.environ.get("PUBLIC_DEMO", "false").lower() == "true":
    available_backends = [BACKEND_CUSTOM, BACKEND_BLIP]
else:
    available_backends = [BACKEND_CUSTOM, BACKEND_BLIP, BACKEND_BLIP2, BACKEND_CLAUDE]

backend = st.radio("Captioning model", available_backends)

if backend == BACKEND_CUSTOM:
    st.caption("**This is the model this project is about**: a ResNet encoder + Bahdanau attention + LSTM "
               "decoder, designed, trained, and evaluated from scratch by me (see README for architecture, "
               "training results, and honestly-documented limitations, e.g. hallucination on out-of-distribution "
               "scenes). Trained on the full 113k-image COCO split (BLEU-4 0.2215, CIDEr 0.6781 on held-out "
               "test data — see README Results). Shorter, more generic captions than the options below — that's "
               "a real, expected consequence of training on ~100k images instead of hundreds of millions.")
elif backend == BACKEND_BLIP:
    st.caption("Not trained by me — Salesforce's pretrained BLIP (~470M params), shown for comparison and "
               "practical use. Short but accurate one-line captions, fast even on CPU, free.")
elif backend == BACKEND_BLIP2:
    st.caption("Not trained by me — Salesforce's pretrained BLIP-2 (~2.7B params, language-model backbone), "
               "shown for comparison. Richer captions than BLIP-large. Runs in float16 on GPU/Apple Silicon "
               "(~6s/image measured locally) but falls back to float32 on CPU-only hosting (e.g. free-tier "
               "Hugging Face Spaces), where it's much slower — better suited to local use with a GPU or Apple "
               "Silicon than a public low-resource deployment.")
else:
    st.caption("Not trained by me — calls the Anthropic API (Claude), shown for comparison and practical use "
               "when you want the most detailed, accurate result. Requires your own API key; costs a small "
               "amount per image.")

model_loaded = True
model = None
blip_processor = blip_model = None
blip2_processor = blip2_model = blip2_device = None
claude_client = None

if backend == BACKEND_CLAUDE:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.text_input(
        "Anthropic API key", type="password",
        help="Not set in the environment. Get one at https://console.anthropic.com/settings/keys — "
             "or set ANTHROPIC_API_KEY before launching the app to skip this prompt.",
    )
    if api_key:
        claude_client = anthropic.Anthropic(api_key=api_key)
    else:
        model_loaded = False
        st.warning("Enter an Anthropic API key above to use this backend.")
elif backend == BACKEND_BLIP:
    with st.spinner("Loading BLIP model (first run downloads ~1.8GB)..."):
        blip_processor, blip_model = load_blip()
elif backend == BACKEND_BLIP2:
    with st.spinner("Loading BLIP-2 model (first run downloads ~10GB, and generation is slow on CPU)..."):
        blip2_processor, blip2_model, blip2_device = load_blip2()
else:
    try:
        model = load_model()
    except FileNotFoundError:
        model_loaded = False
        st.warning("No trained checkpoint found yet (best_checkpoint.pth). "
                   "Train the model first (see notebooks/train_colab.ipynb), then "
                   "place the checkpoint in this folder.")

uploaded_files = st.file_uploader(
    "Upload one or more images", type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and model_loaded:
    if backend == BACKEND_CUSTOM:
        decoding_mode = st.radio("Decoding strategy", ["Greedy", "Beam search (width 3)"])

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        col_image, col_caption = st.columns([1, 2])
        with col_image:
            st.image(image, caption=uploaded_file.name, width="stretch")
        with col_caption:
            with st.spinner(f"Generating caption for {uploaded_file.name}..."):
                if backend == BACKEND_CLAUDE:
                    caption = caption_image_claude(claude_client, image)
                elif backend == BACKEND_BLIP:
                    caption = caption_image_blip(blip_processor, blip_model, image)
                elif backend == BACKEND_BLIP2:
                    caption = caption_image_blip2(blip2_processor, blip2_model, blip2_device, image)
                else:
                    caption = caption_image(model, image, decoding_mode)
            st.success(f"**Caption:** {caption}")
        st.divider()
