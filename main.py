from groq import Groq
from mistralai import Mistral
from dotenv import load_dotenv
import os
import json
import math
import requests
import base64
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
    
def text_analysis(text):
    client = Mistral(api_key = os.environ["MISTRAL_API_KEY"])
    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "system",
                "content": read_file(text_file_path="context_analysis.txt"),
            },
            {
                "role": "user",
                "content": f"Analyse le texte ci-dessous (ta réponse doit etre dans le format JSON): {text}",
            },
        ],
        response_format = {"type": "json_object",}
    )

    predictions = json.loads(chat_response.choices[0].message.content)
    return softmax(predictions) 

def generate_image_from_text(prompt: str, width: int = 512, height: int = 512, seed: int = 123) -> str:
    api_key = os.getenv("CLIPDROP_API_KEY")
    url = "https://clipdrop-api.co/text-to-image/v1"

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["image"]
    else:
        raise Exception(f"Erreur API ClipDrop : {response.status_code}, {response.text}")

"""    if __name__ == "__main__":
        audio_path = "../reve.mp3"
        print("extraction de texte:")
        text = speech_to_text(audio_path, language="fr")
        print(f"text extrait: {text}\n")
        analysis = text_analysis(text)
        print(analysis)
        
        """
