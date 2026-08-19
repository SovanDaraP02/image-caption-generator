# Deploying the live demo to Streamlit Community Cloud

`app.py` is already a working Streamlit app — this puts it on a public URL you can hand to anyone (a reviewer, a recruiter, your professor) without them installing anything. It's free, and unlike Hugging Face Spaces (see `DEPLOY_SPACES.md`), no payment method is needed even for a brand-new account. Streamlit Community Cloud is built specifically for hosting apps like this one, for free, straight from a GitHub repo.

**Prerequisite:** none, really. `best_checkpoint.pth` doesn't need to be uploaded anywhere — `app.py`'s `load_model()` downloads it automatically on first run from a public Hugging Face model repo (`CHECKPOINT_DOWNLOAD_URL` in `app.py`) if it isn't already present. Same behavior locally and on a fresh cloud deploy.

## 1. Sign in and connect your GitHub account

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. If this repo is private (it is, by default here), authorize the Streamlit GitHub App for it during sign-in or first deploy — either grant access to all repositories, or select this one specifically (`SovanDaraP02/image-caption-generator`).

## 2. Deploy the app

1. Click **"New app"**.
2. Repository: `SovanDaraP02/image-caption-generator`.
3. Branch: `master`.
4. Main file path: `app.py`.
5. Click **"Deploy"**. Takes a few minutes the first time — it installs `requirements.txt`, then downloads the checkpoint the first time someone loads the page.

## 3. Restrict the public demo to what actually fits this tier

Streamlit Community Cloud's free tier has about 1GB of RAM, CPU only. BLIP-2 (2.7B params) needs about 14.6GB at its peak just to load, even though quantization shrinks it down to about 3.3GB after loading — that peak crashes this tier outright, and it has. BLIP-3 needs about 18GB just for its weights, which is worse. Your own trained model and BLIP are both small and fast even on CPU, and Claude is a lightweight API call with no local model to load. So for the public link, hide the two that don't fit:

App page → **⋮ (menu) → Settings → Secrets**, add:

```toml
PUBLIC_DEMO = "true"
```

This restricts the "Captioning model" picker to your trained model, BLIP, and Claude. Locally, with `PUBLIC_DEMO` unset, you still get all five backends, including BLIP-2 and BLIP-3 (fast on GPU or Apple Silicon in fp16, slower but working on CPU thanks to quantization — the memory problem above is specific to hosts with very little RAM, not CPU inference in general). See `DEPLOY_SPACES.md` for a path to a BLIP-2-capable public deploy on a host with more RAM.

## 4. Check it built

The app page shows build logs while it starts up. If it fails, the usual cause is a missing package in `requirements.txt` (Streamlit Cloud installs from that file automatically), or the checkpoint download timing out on a slow connection — reloading the page retries it.

## 5. Put the link in your README

Once it's live (the URL looks like `https://<something>.streamlit.app`), put it in `README.md`:

```markdown
## Live demo
https://<your-app>.streamlit.app
```

That link is what you actually send someone. It lets them upload their own photo and see a caption come back, which lands very differently than "here's my GitHub repo."
