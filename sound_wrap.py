from IPython.display import Audio, display
from gtts import gTTS
import whisper

import importlib
import sys
sys.path.append("/content")

import dragontravel_chatbot_vclaude
importlib.reload(dragontravel_chatbot_vclaude)

from dragontravel_chatbot_vclaude import DragonTravelBot, bookings_db

model_whisper = whisper.load_model("medium")
chatbot = DragonTravelBot()

def text_to_speech(flow, text, lang="en"):
    """Convert text to speech and play it"""
    # Create a gTTS object with the text and language
    if flow == "query":
        if lang == "es":
            tts = gTTS(text=text, lang=lang, tld='com.mx')
        else:
            tts = gTTS(text=text, lang=lang, tld='us')
    else:
        if lang == "es":
            tts = gTTS(text=text, lang=lang, tld='us')
        else:
            tts = gTTS(text=text, lang=lang, tld='co.uk')
    
    # Save to a temporary file
    tts.save(f"{flow}.mp3")
    
    # Display an audio widget with the response
    return Audio(f"{flow}.mp3", autoplay=True)

# Function to interact with the chatbot via audio
def audio_chat_interaction():
    # Welcome message
    welcome = "Welcome to DragonTravel / Bienvenido a DragonTravel."
    # display(text_to_speech(welcome))
    display(welcome)
    
    while True:
        # print("\nSpeak now to interact with DragonTravel or type 'exit' to quit")
        
        # Option to use text input 
        text_input = input("Type your message here / Ingrese su mensaje: ")
        
        if text_input.lower() == 'exit':
            break
        
        # Detect language
        if not chatbot.language_set:
            chatbot.detect_language(text_input)
            lang = chatbot.detected_language
            chatbot.language_set = True

        display(text_to_speech("query", text_input, lang))

        user_message = model_whisper.transcribe("query.mp3", )["text"]

        print(f"\nUser message: {user_message}")
        
        response = chatbot.process_message(user_message)
        lang = chatbot.detected_language
        display(text_to_speech("response", response, lang))
        print(f"Response message: {response}")