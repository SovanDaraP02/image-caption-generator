"""Streamlit demo app. Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud (see DEPLOY_STREAMLIT.md) or
Hugging Face Spaces (see DEPLOY_SPACES.md) -- a live link is what you
hand a reviewer so they can try their own image, not just read the
code. best_checkpoint.pth is git-ignored (large, regenerable); if it's
not present locally, load_model() below downloads it automatically
from CHECKPOINT_DOWNLOAD_URL, so a fresh clone/deploy doesn't need it
copied in by hand.
"""

import base64
import io
import os

import anthropic
import requests
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from caption_generator.data.dataset import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD
from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.caption_model import CaptionModel
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCLIP, EncoderCNN

BLIP_CHECKPOINT = "Salesforce/blip-image-captioning-large"
BLIP2_CHECKPOINT = "Salesforce/blip2-opt-2.7b"
BLIP3_CHECKPOINT = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
CLAUDE_MODEL = "claude-sonnet-5"
BLIP3_PROMPT_TEMPLATE = (
    "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
    "<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
)
CHECKPOINT_DOWNLOAD_URL = (
    "https://huggingface.co/sovandara6262/image-caption-generator-checkpoint/"
    "resolve/main/best_checkpoint.pth"
)
DETAILED_DESCRIPTION_PROMPT = (
    "Describe this image in one plain, natural paragraph, like you're telling a "
    "friend what's in the photo. Write in plain, ordinary language.\n\n"
    "What to prioritize, in order, and only for what's actually present: "
    "people first (name how many, and if it reads as a group, their apparent "
    "relationship or activity) -- then animals -- then other notable objects -- "
    "then the setting/background. Skip any category that isn't in the photo "
    "rather than forcing it in. If people's faces or body language clearly show "
    "an emotion or mood (happy, tense, focused, tired, celebrating, etc.), "
    "mention it naturally, as part of the sentence, only when it's actually "
    "visible -- don't guess at feelings you can't see. Mention roughly where "
    "things are in the frame, and colors or materials that stand out, but only "
    "if they'd naturally come up, not as a checklist.\n\n"
    "Match the length to what's actually in the photo -- a simple or empty "
    "scene gets a short description, a busy scene with several people or "
    "animals gets a longer one. Don't pad a simple photo with invented detail "
    "to sound thorough.\n\n"
    "Avoid AI-sounding filler: don't open with 'This image shows/features/"
    "captures', don't use words like 'vibrant', 'nestled', 'boasts', 'a testament "
    "to', or end with a summarizing 'overall' sentence. Just describe what's "
    "there, the way a person would.\n\n"
    "Only state what is actually visible -- do not guess at things you can't see, "
    "and do not invent people, animals, or objects that aren't present."
)


def download_checkpoint_if_missing(checkpoint_path: str) -> None:
    if os.path.exists(checkpoint_path):
        return
    with st.spinner(f"Downloading trained checkpoint (~400MB, first run only)..."):
        response = requests.get(CHECKPOINT_DOWNLOAD_URL, stream=True, timeout=60)
        response.raise_for_status()
        tmp_path = checkpoint_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
        os.rename(tmp_path, checkpoint_path)  # atomic: no other process sees a half-written file


@st.cache_resource
def load_model(checkpoint_path: str = "best_checkpoint.pth") -> tuple[CaptionModel, str]:
    download_checkpoint_if_missing(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    vocab = Vocabulary()
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = checkpoint["vocab_idx2word"]

    # encoder_type is absent on checkpoints saved before EncoderCLIP existed
    # -- default to "resnet50" so those older checkpoints still load.
    encoder_type = checkpoint.get("encoder_type", "resnet50")
    if encoder_type == "clip-vit-base-patch32":
        encoder = EncoderCLIP(fine_tune=False)
        decoder = DecoderWithAttention(vocab_size=len(vocab), encoder_dim=EncoderCLIP.OUTPUT_DIM)
    else:
        encoder = EncoderCNN(fine_tune=False)
        decoder = DecoderWithAttention(vocab_size=len(vocab))
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])

    return CaptionModel(encoder, decoder, vocab, device="cpu"), encoder_type


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


class _EosListStoppingCriteria:
    """From Salesforce's own demo notebook for this model -- stops
    generation at the Phi-3 <|end|> token (id 32007)."""
    def __init__(self, eos_sequence: list[int] = [32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


@st.cache_resource
def load_blip3():
    # BLIP-3 (xgen-mm) ships custom trust_remote_code=True model code that
    # predates several transformers/torch changes -- three real bugs had to
    # be worked around to get it loading at all (verified against this
    # exact checkpoint):
    #
    # 1. Its auto_map registers under the class name "AutoModelForVision2Seq",
    #    which newer transformers renamed to AutoModelForImageTextToText and
    #    no longer resolves via the old name -- bypass Auto* entirely and
    #    load the custom class directly via get_class_from_dynamic_module.
    # 2. Its vision encoder is built with open_clip.create_model_and_transforms(),
    #    which newer transformers' fast/meta-device model init doesn't expect --
    #    open_clip then calls a plain .to(device) on a meta (data-less) tensor
    #    and crashes. Patched to use .to_empty() first when meta tensors are
    #    present, exactly as PyTorch's own error message recommends.
    # 3. The vision encoder (via open_clip, hardcoded float32) and the language
    #    model (built from a sub-config independently declaring bfloat16) end
    #    up in two different dtypes no matter what torch_dtype is passed to
    #    from_pretrained -- neither submodule's dtype is controllable from the
    #    outside at construction time. Fixed by loading, then applying one
    #    blanket .to(device, dtype) call on the fully-assembled model, which
    #    recursively overrides both submodules' independently-hardcoded dtypes.
    import open_clip.factory as _oc_factory
    _orig_set_device = _oc_factory._set_model_device_and_precision

    def _patched_set_device(model, device, precision, is_timm_model=False):
        if any(p.is_meta for p in model.parameters()):
            model.to_empty(device=device)
        return _orig_set_device(model, device, precision, is_timm_model)

    _oc_factory._set_model_device_and_precision = _patched_set_device

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    device = get_device()
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    model_cls = get_class_from_dynamic_module(
        "modeling_xgenmm.XGenMMModelForConditionalGeneration", BLIP3_CHECKPOINT, trust_remote_code=True
    )
    model = model_cls.from_pretrained(BLIP3_CHECKPOINT, trust_remote_code=True, low_cpu_mem_usage=False)
    model = model.to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(BLIP3_CHECKPOINT, trust_remote_code=True, use_fast=False, legacy=False)
    image_processor = AutoImageProcessor.from_pretrained(BLIP3_CHECKPOINT, trust_remote_code=True)
    tokenizer = model.update_special_tokens(tokenizer)
    model.eval()
    return model, tokenizer, image_processor, device


def caption_image_blip3(model, tokenizer, image_processor, device: torch.device, image: Image.Image) -> str:
    inputs = image_processor([image.convert("RGB")], return_tensors="pt", image_aspect_ratio="anyres")
    prompt = BLIP3_PROMPT_TEMPLATE.format(prompt=DETAILED_DESCRIPTION_PROMPT)
    language_inputs = tokenizer([prompt], return_tensors="pt")
    inputs.update(language_inputs)

    def to_device_dtype(v):
        if not hasattr(v, "to"):
            return v
        return v.to(device=device, dtype=model.dtype) if v.is_floating_point() else v.to(device=device)
    inputs = {k: to_device_dtype(v) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs, image_size=[image.size],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False, max_new_tokens=200, top_p=None, num_beams=1,
            stopping_criteria=[_EosListStoppingCriteria()],
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True).split("<|end|>")[0].strip()


def preprocess(image: Image.Image, encoder_type: str = "resnet50") -> torch.Tensor:
    mean, std = (CLIP_MEAN, CLIP_STD) if encoder_type == "clip-vit-base-patch32" else (IMAGENET_MEAN, IMAGENET_STD)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def caption_image(model: CaptionModel, image: Image.Image, decoding_mode: str,
                   encoder_type: str = "resnet50") -> str:
    image_tensor = preprocess(image, encoder_type)
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
BACKEND_BLIP3 = "BLIP-3 (external pretrained model, experimental, reference only)"
BACKEND_CLAUDE = "Claude (external pretrained model, most detailed, reference only)"

# PUBLIC_DEMO=true (set as a Space variable on the deployed public link) hides
# BLIP-2 and BLIP-3: both are heavy local models that need real
# RAM/GPU/Apple Silicon to run well, and free-tier CPU-only hosting doesn't
# have it -- BLIP-3 in particular needs ~18GB just for weights, which
# reliably crashes the app on Streamlit Community Cloud's free tier
# (confirmed in practice, not just a theoretical concern). Claude stays
# public: it's a lightweight API call, no heavy local model to load, and
# costs the deployer nothing extra since visitors must enter their own
# API key (no ANTHROPIC_API_KEY secret is set for the public deploy) --
# see caption_image_claude's api_key handling below.
if os.environ.get("PUBLIC_DEMO", "false").lower() == "true":
    available_backends = [BACKEND_CUSTOM, BACKEND_BLIP, BACKEND_CLAUDE]
else:
    available_backends = [BACKEND_CUSTOM, BACKEND_BLIP, BACKEND_BLIP2, BACKEND_BLIP3, BACKEND_CLAUDE]

backend = st.radio("Captioning model", available_backends)

if backend == BACKEND_CUSTOM:
    st.caption("**This is the model this project is about**: a CLIP ViT-B/32 encoder (swapped in from an "
               "ImageNet-pretrained ResNet -- see README for why) + Bahdanau attention + LSTM decoder, designed, "
               "trained, and evaluated from scratch by me (see README for architecture, training results, and "
               "honestly-documented limitations, e.g. hallucination on out-of-distribution scenes). Trained on "
               "the full 113k-image COCO split (BLEU-4 0.2490, CIDEr 0.7729 on held-out test data, +12% BLEU-4 "
               "over the same data with the original ResNet encoder — see README Results). Shorter, more generic "
               "captions than the options below — that's a real, expected consequence of training on ~100k "
               "images instead of hundreds of millions.")
elif backend == BACKEND_BLIP:
    st.caption("Not trained by me — Salesforce's pretrained BLIP (~470M params), shown for comparison and "
               "practical use. Short but accurate one-line captions, fast even on CPU, free.")
elif backend == BACKEND_BLIP2:
    st.caption("Not trained by me — Salesforce's pretrained BLIP-2 (~2.7B params, language-model backbone), "
               "shown for comparison. Richer captions than BLIP-large. Runs in float16 on GPU/Apple Silicon "
               "(~6s/image measured locally) but falls back to float32 on CPU-only hosting (e.g. free-tier "
               "Hugging Face Spaces), where it's much slower — better suited to local use with a GPU or Apple "
               "Silicon than a public low-resource deployment.")
elif backend == BACKEND_BLIP3:
    st.caption("Not trained by me — Salesforce's pretrained BLIP-3/xGen-MM (~4.6B params, Phi-3 backbone), "
               "shown for comparison. Instruction-tuned (unlike plain BLIP/BLIP-2), so it can attempt the same "
               "prioritized/emotion-aware description prompt as Claude below -- realistically weaker at "
               "following it than Claude, since it's a much smaller model. ~18GB download on first use; the "
               "official model code needed several compatibility fixes to run on current library versions "
               "(see load_blip3() for details) -- experimental, not production-grade.")
else:
    st.caption("Not trained by me — calls the Anthropic API (Claude), shown for comparison and practical use "
               "when you want the most detailed, accurate result. Requires your own API key; costs a small "
               "amount per image.")

model_loaded = True
model = None
model_encoder_type = "resnet50"
blip_processor = blip_model = None
blip2_processor = blip2_model = blip2_device = None
blip3_model = blip3_tokenizer = blip3_image_processor = blip3_device = None
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
elif backend == BACKEND_BLIP3:
    with st.spinner("Loading BLIP-3 model (first run downloads ~18GB, may take a while)..."):
        blip3_model, blip3_tokenizer, blip3_image_processor, blip3_device = load_blip3()
elif backend == BACKEND_CUSTOM:
    try:
        model, model_encoder_type = load_model()
    except requests.exceptions.RequestException as e:
        model_loaded = False
        st.warning(f"Couldn't download the trained checkpoint ({e}). "
                   "Check your internet connection and reload the page.")
    except FileNotFoundError:
        model_loaded = False
        st.warning("No trained checkpoint found yet (best_checkpoint.pth), and the automatic "
                   "download failed. Train the model first (see notebooks/train_colab.ipynb), "
                   "then place the checkpoint in this folder.")

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
                elif backend == BACKEND_BLIP3:
                    caption = caption_image_blip3(
                        blip3_model, blip3_tokenizer, blip3_image_processor, blip3_device, image
                    )
                else:
                    caption = caption_image(model, image, decoding_mode, model_encoder_type)
            st.success(f"**Caption:** {caption}")
        st.divider()
