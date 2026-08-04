"""
Streamlit demo app (Day 13). Run locally with:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud or Hugging Face Spaces once
you have best_checkpoint.pth — that live link is what you send to a
reviewer so they can try their own image, not just watch a video.
"""

import sys
import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, "data")
sys.path.insert(0, "models")
from vocabulary import Vocabulary          # noqa: E402
from encoder import EncoderCNN              # noqa: E402
from decoder import DecoderWithAttention    # noqa: E402
from caption_model import CaptionModel      # noqa: E402


@st.cache_resource
def load_model(checkpoint_path: str = "best_checkpoint.pth"):
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


st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️")
st.title("🖼️ Multimodal Image Caption Generator")
st.caption("ResNet-50 encoder + attention mechanism + LSTM decoder, trained on Flickr8k")

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning("No trained checkpoint found yet (best_checkpoint.pth). "
               "Train the model with train.py on Colab first, then place "
               "the checkpoint in this folder.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and model_loaded:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your image", use_column_width=True)

    decoding_mode = st.radio("Decoding strategy", ["Greedy", "Beam search (width 3)"])

    with st.spinner("Generating caption..."):
        image_tensor = preprocess(image)
        if decoding_mode == "Greedy":
            caption, _ = model.generate_greedy(image_tensor)
        else:
            caption = model.generate_beam(image_tensor, beam_width=3)

    st.success(f"**Generated caption:** {caption}")
