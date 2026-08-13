# Deploying the live demo to Streamlit Community Cloud

`app.py` is already a working Streamlit app — this puts it on a public URL
you can hand to anyone (a reviewer, a recruiter, your professor) without
them installing anything. Free, and unlike Hugging Face Spaces (see
`DEPLOY_SPACES.md`), no payment method is required even for a brand-new
account — Streamlit Community Cloud is built specifically for hosting
Streamlit apps like this one, for free, directly from a GitHub repo.

**Prerequisite:** `best_checkpoint.pth` doesn't need to be uploaded
anywhere by hand — `app.py`'s `load_model()` downloads it automatically
on first run from a public Hugging Face model repo
(`CHECKPOINT_DOWNLOAD_URL` in `app.py`) if it isn't already present. This
works the same way locally and on a fresh cloud deploy.

## 1. Sign in and connect your GitHub account

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. If this repo is private (it is, by default here), authorize the
   Streamlit GitHub App for it: during sign-in/first deploy you'll be
   prompted to grant repo access — either "All repositories" or select
   this one specifically (`SovanDaraP02/image-caption-generator`)

## 2. Deploy the app

1. Click **"New app"**
2. Repository: `SovanDaraP02/image-caption-generator`
3. Branch: `master`
4. Main file path: `app.py`
5. Click **"Deploy"** — it takes a few minutes the first time (installing
   `requirements.txt`, then downloading the checkpoint on first page load)

## 3. Restrict the public demo to the fast backends

Streamlit Community Cloud's free tier is CPU-only, no GPU. BLIP-2 (2.7B
params) falls back to float32 there and takes minutes per image — a bad
first impression for someone testing the link live. Claude works but
costs API credits on every visitor's request. Your own trained model and
BLIP are both small and fast even on CPU, so for the public link, hide
the other two:

App page → **⋮ (menu) → Settings → Secrets**, add:

```toml
PUBLIC_DEMO = "true"
```

This restricts the "Captioning model" picker to just your trained model
and BLIP. Locally (where `PUBLIC_DEMO` is unset), you still have all four
backends, including BLIP-2 (fast there, via GPU/Apple Silicon fp16) and
Claude.

## 4. Check it built

The app page shows build logs while it's starting. If it fails, the most
common cause is a missing package in `requirements.txt` (Streamlit Cloud
installs from that file automatically) or the checkpoint download timing
out on a slow connection — reloading the page retries it.

## 5. Put the link in your README

Once it's live (URL looks like
`https://<something>.streamlit.app`), put it in `README.md`:

```markdown
## Live demo
https://<your-app>.streamlit.app
```

That link is what you actually send someone — it lets them upload their
own photo and see a real caption come back, which reads very differently
from "here's my GitHub repo."
