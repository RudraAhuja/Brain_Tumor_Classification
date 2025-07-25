import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

@st.cache_resource
def load_mobilenet_model():
    return load_model('best_mobilenetv2.h5')

model = load_mobilenet_model()

class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

st.title("🧠 Brain Tumor MRI Classification")

st.markdown("""
Upload an MRI brain scan image, and this app will predict whether it indicates:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**
""")

uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    processed = np.expand_dims(img_array, axis=0)

    prediction = model.predict(processed)[0]
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("🧾 Prediction Result:")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
