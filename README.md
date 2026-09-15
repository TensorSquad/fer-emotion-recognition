# Facial Emotion Recognition (FER2013)

Deep learning pipeline for facial emotion recognition, classifying faces into 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral) using an EfficientNet-B3 backbone with a CBAM attention module.

## Project structure

```
.
├── app.py              # Streamlit inference app
├── model.py            # EmotionNet architecture (EfficientNet-B3 + CBAM)
├── requirements.txt
├── models/
│   └── best_emotion_model.pth   # trained checkpoint (not included, add your own)
└── README.md
```

## Architecture

- Backbone: EfficientNet-B3 (ImageNet pretrained during training)
- Attention: CBAM (Channel Attention + Spatial Attention)
- Classifier head: Linear(1536,512) -> BN -> ReLU -> Dropout -> Linear(512,256) -> BN -> ReLU -> Dropout -> Linear(256,7)
- Trained with Focal Loss + Label Smoothing, MixUp, AdamW, CosineAnnealingLR, AMP

## Setup

```bash
pip install -r requirements.txt
```

Place your trained checkpoint at `models/best_emotion_model.pth`.

## Run

```bash
streamlit run app.py
```

## Dataset

[FER2013](https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset) via Kaggle, ~49,779 grayscale facial images across 7 emotion classes.
