import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import pickle
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Digit Classification",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Digit Classification App")
st.markdown("Upload a handwritten digit image and classify it.")

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("cnn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --------------------------------------------------
# Class Labels
# --------------------------------------------------
CLASS_LABELS = [str(i) for i in range(10)]

# --------------------------------------------------
# Layout
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    invert_colors = st.checkbox(
        "Invert Colors",
        value=False
    )

    image = None

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

with col2:
    st.subheader("Prediction Result")

    if image is not None:

        if st.button("🚀 Predict"):

            try:
                # Convert to grayscale
                img = image.convert("L")

                if invert_colors:
                    img = ImageOps.invert(img)

                # Resize
                img = img.resize((32, 32))

                # Normalize
                img_array = np.array(img).astype("float32") / 255.0

                # Shape adjustment
                img_array = img_array.reshape(1, 32, 32, 1)

                # Predict
                predictions = model.predict(img_array)

                predicted_class = int(np.argmax(predictions))
                confidence = float(np.max(predictions))

                st.success("Prediction Complete!")

                st.markdown(
                    f"<h1 style='text-align:center'>{predicted_class}</h1>",
                    unsafe_allow_html=True
                )

                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

                # Probabilities
                probs = predictions[0]

                df = pd.DataFrame({
                    "Digit": CLASS_LABELS,
                    "Probability (%)": probs * 100
                })

                st.dataframe(df)

                # Chart
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(CLASS_LABELS, probs * 100)
                ax.set_ylabel("Probability (%)")
                ax.set_xlabel("Digit")
                ax.set_title("Prediction Probabilities")

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Prediction Error: {e}")

    else:
        st.info("Upload an image first.")
