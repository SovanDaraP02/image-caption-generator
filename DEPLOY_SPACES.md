# Deploying to Hugging Face Spaces (optional — not the current live demo)

This project's live demo runs on Streamlit Community Cloud (see
`DEPLOY_STREAMLIT.md`), which is free with no account verification but
can't run BLIP-2 publicly (~1GB RAM vs. BLIP-2's ~14.6GB load peak — see
`MODEL_CARD.md`). This file documents an alternative that *could* run
BLIP-2 publicly, for anyone who wants to extend this project that way —
it isn't set up as of this writing, and getting there is more friction
than the Streamlit Cloud path, for reasons below.

## What's actually true here as of this writing

- **Streamlit is no longer a first-class Space SDK.** Hugging Face's
  "Create a new Space" page now only offers Gradio, Docker, or Static
  directly — a Streamlit app is deployed via the **Docker** template
  (pick Docker, then choose the Streamlit template from the list inside
  it), not a dedicated "Streamlit" button like it used to be.
- **Docker and Gradio Spaces may require a paid plan or account
  verification.** On at least one tested account, both were shown as
  "Paid" with a prompt to subscribe to PRO, even though Hugging Face's
  own docs describe a free CPU-basic tier (2 vCPU / 16GB RAM,
  unmetered). It's unconfirmed whether adding a payment method (without
  subscribing) unlocks this for free, as it used to — Settings → Billing
  → add a card, then check back on the new-Space page, is worth trying
  before assuming a subscription is required. If a subscription really
  is required now, that's a real ~$9/mo cost decision, not a setup step.
- **16GB RAM is the actual reason this tier would fit BLIP-2** where
  Streamlit Cloud's ~1GB can't — `load_blip2()`'s CPU path peaks at
  ~14.6GB while loading the full fp32 model, before quantization
  shrinks it to ~3.3GB resident. Quantizing after loading doesn't lower
  that peak (see the `PUBLIC_DEMO` comment in `app.py`).

## If you get a working compute Space

1. [huggingface.co/new-space](https://huggingface.co/new-space) → Docker
   → Streamlit template → Hardware: CPU basic → Visibility: Public
2. `git remote add space https://huggingface.co/spaces/<your-username>/image-caption-generator && git push space master:main`
3. `app.py`'s `load_model()` downloads `best_checkpoint.pth` automatically
   on first run — nothing to upload by hand.
4. **BLIP-2 still won't show publicly by default** — `app.py`'s
   `PUBLIC_DEMO=true` currently hides both BLIP-2 and BLIP-3
   unconditionally (see `app.py`), because the project's default
   assumption is Streamlit Cloud-sized hosting. To actually expose
   BLIP-2 on a host that can take it, edit the `available_backends` list
   in `app.py` (search `PUBLIC_DEMO`) to include `BACKEND_BLIP2`, or set
   a different variable and branch on it — this needs a small code
   change, it isn't a flip-a-flag config option today.
5. Once live: `https://huggingface.co/spaces/<your-username>/image-caption-generator`
