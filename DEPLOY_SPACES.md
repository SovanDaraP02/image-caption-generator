# Deploying the live demo to Hugging Face Spaces

`app.py` is already a working Streamlit app — this puts it on a public URL
you can hand to anyone (a reviewer, a recruiter) without them installing
anything.

**Why Spaces over Streamlit Community Cloud** (see `DEPLOY_STREAMLIT.md`
for that alternative): Spaces' free `cpu-basic` tier gives ~16GB RAM.
Streamlit Community Cloud's free tier gives roughly ~1GB. That gap matters
specifically for the BLIP-2 backend — even after int8 quantization,
`load_blip2()`'s CPU path has to materialize the *full fp32* model
(~14.6GB peak) before it can shrink it down, and no amount of quantization
changes that peak. ~1GB hosts crash there; ~16GB hosts have room. If
you don't care about BLIP-2 working on the public link, Streamlit
Community Cloud is simpler to set up (see `DEPLOY_STREAMLIT.md`) and this
distinction doesn't matter.

**Note:** Hugging Face requires a payment method on file (even for the
free `cpu-basic` tier) before a *new* account can create a live-compute
Space (Streamlit/Gradio/Docker) — confirmed via `whoami()`'s `canPay`/
`isPro` fields returning a 402 on `create_repo` otherwise. This doesn't
cost anything on the free tier; it's Hugging Face's fraud-prevention
check, not a charge.

**Prerequisite:** none — `app.py`'s `load_model()` downloads
`best_checkpoint.pth` automatically on first run from a public Hugging
Face model repo (`CHECKPOINT_DOWNLOAD_URL` in `app.py`) if it isn't
already present. Nothing needs to be uploaded by hand.

## 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space) (sign
   up if you don't have an account — free, card required per the note above)
2. Space name: `image-caption-generator` (or anything you like)
3. SDK: **Streamlit**
4. Hardware: **CPU basic (free)** — 16GB RAM, enough for BLIP-2's CPU path;
   don't downgrade this if you want BLIP-2 to work on the public link
5. Visibility: Public (so the link actually works for other people)
6. Click "Create Space" — it gives you a git URL like
   `https://huggingface.co/spaces/<your-username>/image-caption-generator`

## 2. Push this project to it

```bash
# from this project's root
git remote add space https://huggingface.co/spaces/<your-username>/image-caption-generator
git push space master:main
```

If it asks for credentials: use your HF username and an access token
(not your password) — generate one at huggingface.co → Settings → Access Tokens.

## 3. Set PUBLIC_DEMO

On the Space page → **Settings → Variables and secrets → New variable**:
- Name: `PUBLIC_DEMO`
- Value: `true`

This hides only the BLIP-3 backend (~18GB, too large for any free tier).
Your trained model, BLIP, BLIP-2, and Claude all stay available to public
visitors — see the `PUBLIC_DEMO` comment in `app.py` for the exact memory
reasoning per backend.

## 4. Check it built

Open the Space URL — it takes 1-2 minutes to build the first time, plus
however long the checkpoint/model downloads take on first page load
(BLIP-2 alone is ~10GB). If it fails, click "Logs" on the Space page; the
most common cause is a missing package in `requirements.txt` (Spaces
installs from that file automatically).

## 5. Put the link in your README

Once it's live, replace the placeholder in `README.md`:

```markdown
## Live demo
https://huggingface.co/spaces/<your-username>/image-caption-generator
```

That link is what you actually send someone — it lets them upload their own
photo and see a real caption come back, which reads very differently from
"here's my GitHub repo."
