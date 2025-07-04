import streamlit as st
import json
import tempfile
import os
import matplotlib.pyplot as plt

from main import speech_to_text, text_analysis, generate_image_from_text


st.set_page_config(page_title="Analyseur d'Émotions", layout="centered")

st.title("🧠 Analyse Émotionnelle d'un Texte ou d'un Audio")
st.markdown("Ce programme détecte le **niveau de 6 émotions** dans un texte ou un fichier audio : joie, anxiété, tristesse, colère, fatigue et peur.")

# Mode de saisie
mode = st.radio("📥 Choisis un mode d’entrée :", ["Texte", "Audio (.mp3)"])

text_to_analyze = ""

if mode == "Texte":
    text_input = st.text_area("✍️ Entre ton texte ici :", height=200)
    if st.button("Analyser le texte"):
        if text_input.strip() == "":
            st.warning("⚠️ Merci d’entrer un texte.")
        else:
            with st.spinner("Analyse en cours..."):
                result = text_analysis(text_input)
                st.success("Analyse terminée !")
                st.json(result)

                # Graphique en barres
                fig, ax = plt.subplots()
                ax.bar(result.keys(), result.values())
                ax.set_title("Intensité des émotions")
                ax.set_ylim(0, 1)
                st.pyplot(fig)

                # Image générée à partir du rêve
                st.markdown("## 🎨 Image générée à partir de ton rêve")

                dream_prompt = f"Un rêve surréaliste : {text_input}. Illustration onirique, style artistique doux et lumineux."

                try:
                    image_data = generate_image_from_text(dream_prompt)
                    st.image(image_data, caption="Image du rêve", use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la génération de l’image : {e}")
                    st.text(f"Prompt utilisé : {dream_prompt}")

elif mode == "Audio (.mp3)":
    uploaded_file = st.file_uploader("🎵 Charge un fichier audio (.mp3)", type=["mp3"])
    if uploaded_file and st.button("Analyser l'audio"):
        with st.spinner("Transcription de l'audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_audio_path = tmp_file.name

            try:
                text_from_audio = speech_to_text(temp_audio_path)
                st.markdown("**Texte extrait :**")
                st.write(text_from_audio)

                result = text_analysis(text_from_audio)
                st.success("Analyse terminée !")
                st.json(result)

                # Graphique
                fig, ax = plt.subplots()
                ax.bar(result.keys(), result.values())
                ax.set_title("Intensité des émotions")
                ax.set_ylim(0, 1)
                st.pyplot(fig)

                # Image générée à partir du rêve
                st.markdown("## 🎨 Image générée à partir de ton rêve")

                try:
                    prompt = f"Crée une image à partir de ce rêve : {text_from_audio}"
                    image_data = generate_image_from_text(prompt)
                    st.image(image_data, caption="Image du rêve", use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la génération de l’image : {e}")
                    st.text(f"Prompt utilisé : {text_from_audio}")

            finally:
                os.remove(temp_audio_path)