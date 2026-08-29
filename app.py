import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEVICE = torch.device("cpu")

CLASSES = ["NORMAL", "PNEUMONIA"]


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = models.densenet121(weights=None)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        2
    )

    model_path = hf_hub_download(
        repo_id="Divyaanshvats/pneumovision-densenet121",
        filename="densenet121_pneumonia.pth"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config(
    page_title="PneumoVision",
    page_icon="🫁"
)

st.title("🫁 PneumoVision")

st.subheader(
    "Chest X-Ray Pneumonia Classification"
)

st.write(
    "DenseNet121 Transfer Learning Model"
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

        confidence = probabilities[
            predicted_class
        ].item()

    st.divider()

    st.subheader("Prediction")

    st.write(
        f"### {CLASSES[predicted_class]}"
    )

    st.write(
        f"Confidence: **{confidence:.2%}**"
    )

    st.progress(confidence)

    st.divider()

    st.caption(
        "For educational and research purposes only. "
        "This application is not a medical diagnostic tool."
    )
