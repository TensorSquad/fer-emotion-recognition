import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model import EmotionNet, CLASS_NAMES

MODEL_PATH = "models/best_emotion_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


@st.cache_resource
def load_model():
    model = EmotionNet(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict(image, model):
    tensor = val_transform(image.convert("L")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    return probs


st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("Facial Emotion Recognition")
st.caption("EfficientNet-B3 + CBAM | FER2013 | 7 classes")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input image", width=300)

    try:
        model = load_model()
    except FileNotFoundError:
        st.error(f"Model checkpoint not found at {MODEL_PATH}. Add best_emotion_model.pth to the models/ folder.")
        st.stop()

    probs = predict(image, model)
    pred_idx = probs.argmax()

    st.subheader(f"Predicted: {CLASS_NAMES[pred_idx].upper()}")
    st.write(f"Confidence: {probs[pred_idx] * 100:.1f}%")

    df = pd.DataFrame({"emotion": CLASS_NAMES, "probability": probs})
    st.bar_chart(df.set_index("emotion"))
