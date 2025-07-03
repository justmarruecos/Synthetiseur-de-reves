from groq import Groq
from mistralai import Mistral
from dotenv import load_dotenv
import os
import json
import math
import requests
load_dotenv()

def read_file(text_file_path):
    with open(text_file_path, "r") as file:
        return file.read()
    
def softmax(predictions):
    output = {}
    for sentiment, predicted_value in predictions.items():
        output[sentiment] = math.exp(predicted_value*10) / sum(math.exp(value*10) for value in predictions.values())
    return output

def speech_to_text(audio_path, language="fr"):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            prompt="Extrait le texte de l'audio de la manière la plus factuelle possible", 
            response_format="verbose_json",
            timestamp_granularities = ["word", "segment"],
            language=language,  
            temperature=0.0 
        )
        return transcription.text
    
def text_analysis(text: str) -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    system_prompt = read_file("context_analysis.txt")

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyse le texte suivant (réponds en JSON) : {text}"}
        ]
    )

    raw_json = response.choices[0].message.content
    predictions = json.loads(raw_json)
    return softmax(predictions)

def generate_image_from_text(prompt: str) -> bytes:
    api_key = os.getenv("CLIPDROP_API_KEY")
    url = "https://clipdrop-api.co/text-to-image/v1"
    headers = {
        "x-api-key": api_key
    }

    files = {
        "prompt": (None, prompt, "text/plain")
    }

    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        print("✅ ClipDrop a bien répondu :", response.status_code)
        return response.content  # image binaire PNG
    else:
        raise Exception(f"Erreur API ClipDrop : {response.status_code} - {response.text}")


"""    if __name__ == "__main__":
        audio_path = "../reve.mp3"
        print("extraction de texte:")
        text = speech_to_text(audio_path, language="fr")
        print(f"text extrait: {text}\n")
        analysis = text_analysis(text)
        print(analysis)
        
        """
