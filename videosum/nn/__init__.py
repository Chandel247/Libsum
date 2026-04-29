# Neural network models for videosum.
#
# Planned modules (not yet implemented):
#   frame_caption_net.py  — CNN (ResNet-18) + LSTM decoder for frame captioning
#   video_summary_net.py  — Bidirectional LSTM encoder + Attention decoder for summarization
#   tokenizer.py          — Word-level tokenizer shared by both models
#   train_caption.py      — Training script for FrameCaptionNet (MS COCO dataset)
#   train_summary.py      — Training script for VideoSummaryNet (CNN/DailyMail dataset)
#   weights/              — Saved .pt files (gitignored)
#
# To use these models, install the neural extra:
#   pip install -e ".[neural]"
#
# Then wire them up via NeuralProvider:
#   from videosum.providers import NeuralProvider
#   provider = NeuralProvider(
#       vision_weights="videosum/nn/weights/caption.pt",
#       text_weights="videosum/nn/weights/summary.pt",
#   )
