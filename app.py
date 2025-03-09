import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Charger le modèle
model = tf.keras.models.load_model("c:\Users\PC\Downloads\PROJET DEEPL\cats_vs_dogs_model.keras")


# Fonction pour faire une prédiction
def predict_image(img):
    img = img.resize((150, 150))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  

    prediction = model.predict(img_array)
    return "Chien 🐶" if prediction[0][0] > 0.5 else "Chat 🐱"

# Interface Streamlit
st.title(" Détection de chien 🐶 ou chat 🐱")
st.write("Chargez une image pour obtenir la prédiction.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)  # Ouvrir l'image
    st.image(image_pil, caption="Image chargée", use_column_width=True)

    if st.button("Prédire"):
        result = predict_image(image_pil)
        st.success(f"Résultat : {result}")
