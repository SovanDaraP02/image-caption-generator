# Deploying the live demo to Hugging Face Spaces

`app.py` is already a working Streamlit app — this just puts it on a public
URL you can hand to anyone (a reviewer, a recruiter, Mr. Khim) without them
installing anything. Free tier, no credit card.

**Prerequisite:** you need a trained `best_checkpoint.pth` first (run
`notebooks/train_colab.ipynb` on Colab — see the main README).

## 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space) (sign
   up if you don't have an account — free)
2. Space name: `image-caption-generator` (or anything you like)
3. SDK: **Streamlit**
4. Hardware: CPU basic (free) — inference on one image at a time is fine on CPU
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

## 3. Add the checkpoint

`best_checkpoint.pth` is git-ignored in this repo (it's large and
regenerable), so push it separately straight into the Space repo:

```bash
git clone https://huggingface.co/spaces/<your-username>/image-caption-generator space-repo
cp best_checkpoint.pth space-repo/
cd space-repo
git add best_checkpoint.pth
git commit -m "Add trained checkpoint"
git push
```

(Hugging Face repos use Git LFS automatically for large files over 10MB —
no extra setup needed.)

## 4. Check it built

Open the Space URL — it takes 1-2 minutes to build the first time. If it
fails, click "Logs" on the Space page; the most common cause is a missing
package in `requirements.txt` (Spaces installs from that file automatically).

## 5. Put the link in your README

Once it's live, replace the placeholder in `README.md`:

```markdown
## Live demo
https://huggingface.co/spaces/<your-username>/image-caption-generator
```

That link is what you actually send someone — it lets them upload their own
photo and see a real caption come back, which reads very differently from
"here's my GitHub repo."
