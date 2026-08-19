# What I learned building an image captioning model from scratch

I built a model that looks at a photo and writes a caption for it — encoder, attention, decoder, all hand-implemented in PyTorch, no pretrained captioning model involved. Getting it running wasn't the hard part. Figuring out why each attempt to improve it did or didn't work was.

First improvement: more data. I started with Flickr8k, 6,000 images, then scaled up to a 50,000-image COCO subset. BLEU-4 went up 38%, CIDEr up 48%. Same code, more examples, a meaningfully better model.

So I tried it again — more than doubled the training set to 113,000 images. This time BLEU-4 barely moved, less than 1%. Validation loss flattened out over the last few epochs.

It would've been easy to just call that a ceiling and move on. Instead I wanted to know why: was the problem data volume, or something the extra data couldn't fix?

The encoder was a ResNet-50, pretrained on ImageNet classification — 1,000 object labels with no linguistic structure to them at all. "Golden retriever" and "beagle" are just two unrelated classes to it. The decoder was trying to build sentences out of that. Maybe the features themselves were the ceiling, not the amount of data.

So I kept the same 113k images and swapped the encoder for CLIP, which is pretrained to match images with their actual captions instead of classifying them. Same data, same decoder, one thing changed. BLEU-4 jumped 12.4% — a bigger gain than the entire second data-scaling attempt had given me.

Then I fine-tuned that CLIP encoder's last layer, at a learning rate ten times lower than the decoder's, starting from the converged checkpoint. Another real gain, +4.5% BLEU-4, and validation loss started flattening again by the final epoch — a sign this phase was hitting its own limit too, not that anything had broken.

Final numbers: BLEU-4 0.26, CIDEr 0.81, on a 5,000-image held-out COCO test split, up from 0.16 BLEU-4 at the Flickr8k baseline. What mattered more than the final number was realizing "add more data" and "fix the architecture" are two different fixes for two different problems, and the only way to tell which one you're facing is to isolate the variable and actually measure it.

Full writeup, code, and all five checkpoints compared: see the [README](./README.md).
