# Deploying to Hugging Face Spaces (optional — not the current live demo)

This project's live demo runs on Streamlit Community Cloud instead (see `DEPLOY_STREAMLIT.md`), which is free with no account verification but can't run BLIP-2 publicly — its free tier has about 1GB of RAM, and BLIP-2 peaks at around 14.6GB while loading (see `MODEL_CARD.md`). This file covers an alternative that could run BLIP-2 publicly, for anyone who wants to take the project that direction. It isn't set up as of this writing, and getting there takes more effort than the Streamlit Cloud path, for the reasons below.

## What's actually true here as of this writing

- **Streamlit isn't a first-class Space SDK anymore.** Hugging Face's "Create a new Space" page only offers Gradio, Docker, or Static directly. A Streamlit app now gets deployed through the Docker template — pick Docker, then choose Streamlit from the templates inside it — instead of a dedicated Streamlit button like it used to have.
- **Docker and Gradio Spaces may require a paid plan or account verification.** On at least one account I tested, both showed as "Paid" with a prompt to subscribe to PRO, even though Hugging Face's own docs describe a free CPU-basic tier (2 vCPU, 16GB RAM, unmetered). It's not confirmed whether adding a payment method without subscribing unlocks it for free, the way it used to. Worth trying — Settings → Billing → add a card, then check the new-Space page again — before assuming a subscription is actually required. If it turns out one is, that's a real ~$9/month decision, not just a setup step.
- **16GB of RAM is the actual reason this tier would fit BLIP-2** where Streamlit Cloud's 1GB can't. `load_blip2()`'s CPU path peaks at about 14.6GB while loading the full fp32 model, before quantization shrinks it to about 3.3GB resident. Quantizing after loading doesn't lower that peak — see the `PUBLIC_DEMO` comment in `app.py`.

## If you get a working compute Space

1. [huggingface.co/new-space](https://huggingface.co/new-space) → Docker → Streamlit template → Hardware: CPU basic → Visibility: Public.
2. `git remote add space https://huggingface.co/spaces/<your-username>/image-caption-generator && git push space master:main`
3. `app.py`'s `load_model()` downloads `best_checkpoint.pth` automatically on first run — nothing to upload by hand.
4. BLIP-2 still won't show up publicly by default. `app.py`'s `PUBLIC_DEMO=true` currently hides both BLIP-2 and BLIP-3 no matter what, since the project's default assumption is Streamlit Cloud-sized hosting. To actually expose BLIP-2 on a host that can handle it, edit the `available_backends` list in `app.py` (search for `PUBLIC_DEMO`) to include `BACKEND_BLIP2`, or add a separate variable to branch on. That's a small code change, not a config flag you can just flip today.
5. Once it's live: `https://huggingface.co/spaces/<your-username>/image-caption-generator`
