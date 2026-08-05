"""Streamlit demo app. Run locally with:
    streamlit run app.py

Deploy on Hugging Face Spaces (see DEPLOY_SPACES.md) once you have
best_checkpoint.pth -- a live link is what you hand a reviewer so they
can try their own image, not just read the code.
"""

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.caption_model import CaptionModel
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCNN


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


st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️")
st.title("🖼️ Multimodal Image Caption Generator")
st.caption("ResNet encoder + attention mechanism + LSTM decoder, trained on Flickr8k")

try:
    model = load_model()
    model_loaded = True
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
    decoding_mode = st.radio("Decoding strategy", ["Greedy", "Beam search (width 3)"])

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        col_image, col_caption = st.columns([1, 2])
        with col_image:
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        with col_caption:
            with st.spinner(f"Generating caption for {uploaded_file.name}..."):
                caption = caption_image(model, image, decoding_mode)
            st.success(f"**Caption:** {caption}")
        st.divider()
