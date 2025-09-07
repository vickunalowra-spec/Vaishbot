import streamlit as st
import sounddevice as sd
import numpy as np
import queue
from scipy.io.wavfile import write
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from bs4 import BeautifulSoup
from datetime import datetime
from gtts import gTTS
import pygame
import os
import tempfile

# --- UI Setup ---
st.set_page_config(page_title="VaishBot", page_icon="🎓")

# ✅ Use absolute path for avatar
IMAGE_PATH = r"D:\Vaishbot\avatar.jpg"
if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, width=120)
else:
    st.warning("⚠️ avatar.jpg not found at the specified path.")

st.markdown("<h1 style='text-align:center;color:maroon;'>Welcome to VaishBot</h1>", unsafe_allow_html=True)

# --- Audio Playback with pygame ---
pygame.mixer.init()

def play_audio(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    # Delete the temp file after playback
    os.remove(file_path)

# --- Hybrid TTS with mkstemp ---
def speak(text, lang_code="en"):
    lang_map = {
        "ta": "ta", "hi": "hi", "en": "en", "te": "te", "ml": "ml",
        "kn": "kn", "gu": "gu", "or": "or", "ks": "ks", "raj": "raj"
    }
    # Create temp MP3 file
    fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)  # Close so pygame can access
    tts = gTTS(text=text, lang=lang_map.get(lang_code, "en"))
    tts.save(tmp_path)
    play_audio(tmp_path)

# --- Silence-based Recording ---
def record_until_silence(fs=16000, silence_thresh=500, silence_duration=1.0):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback)
    audio_data = []
    silent_chunks = 0
    chunk_size = 1024
    chunks_per_sec = int(fs / chunk_size)

    with stream:
        st.info("🎙 Speak now... stop when silent")
        while True:
            chunk = q.get()
            audio_data.append(chunk)
            volume = np.abs(chunk).mean()
            if volume < silence_thresh:
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks > silence_duration * chunks_per_sec:
                break

    audio = np.concatenate(audio_data, axis=0)
    return audio.flatten(), fs

def save_wav(filename, data, fs):
    write(filename, fs, data)

# --- ASR via Whisper API ---
def transcribe_whisper(audio_path, lang_code="en"):
    url = "http://localhost:8000/transcribe"
    with open(audio_path, "rb") as f:
        response = requests.post(url, files={"file": f})
    return response.json().get("text", "")

# --- Scrape Website ---
def scrape_sdnbvc():
    html = requests.get("https://sdnbvc.edu.in").text
    soup = BeautifulSoup(html, "html.parser")
    return " ".join([p.text for p in soup.find_all("p")])

# --- Match Answer ---
def find_answer(query, scraped_text):
    return query if query.lower() in scraped_text.lower() else "Sorry, I couldn’t find an answer."

# --- Log Callback ---
def log_callback_request(user_query, phone_number, lang_code):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("VaishBot Callback Log").sheet1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, lang_code, user_query, phone_number])

# --- Confirm Callback ---
def confirm_callback(lang_code):
    confirmation_text = {
        "ta": "உங்கள் அழைப்பு கோரிக்கை பதிவு செய்யப்பட்டது. எங்கள் குழு விரைவில் தொடர்பு கொள்கிறது.",
        "hi": "आपकी कॉल बैक अनुरोध दर्ज कर ली गई है। हमारी टीम जल्द ही संपर्क करेगी।",
        "en": "Your callback request has been logged. Our team will contact you soon.",
        "te": "మీ కాల్‌బ్యాక్ అభ్యర్థన నమోదు చేయబడింది. మా బృందం త్వరలో సంప్రదిస్తుంది.",
        "ml": "നിങ്ങളുടെ കോൾബാക്ക് അഭ്യർത്ഥന രജിസ്റ്റർ ചെയ്തിരിക്കുന്നു. ഞങ്ങളുടെ ടീം ഉടൻ ബന്ധപ്പെടും.",
        "kn": "ನಿಮ್ಮ ಕಾಲ್‌ಬ್ಯಾಕ್ ವಿನಂತಿಯನ್ನು ದಾಖಲಿಸಲಾಗಿದೆ. ನಮ್ಮ ತಂಡ ಶೀಘ್ರದಲ್ಲೇ ಸಂಪರ್ಕಿಸುತ್ತದೆ.",
        "gu": "તમારું કોલબેક વિનંતી નોંધાઈ ગઈ છે. અમારી ટીમ ટૂંક સમયમાં સંપર્ક કરશે.",
        "or": "ଆପଣଙ୍କର କଲ୍ ବ୍ୟାକ୍ ଅନୁରୋଧ ରେକର୍ଡ ହୋଇଛି। ଆମ ଟିମ୍ ଶୀଘ୍ର ଯୋଗାଯୋଗ କରିବେ।",
        "ks": "تُهند کال بیک درخواست درج کی گئی ہے۔ ہمارا عملہ جلد رابطہ کرے گا۔",
        "raj": "थारो कॉलबैक अनुरोध दर्ज करियो गयो है। हमारी टीम जल्द ही संपर्क करेगी।"
    }
    speak(confirmation_text.get(lang_code, confirmation_text["en"]), lang_code)

# --- Language Selection ---
welcome_text = "Welcome to SDNB Vaishnav College. Please say your preferred language: Tamil, Hindi, Telugu, Malayalam, Kannada, Gujarati, Odia, Kashmiri, or Rajasthani."
speak(welcome_text, "en")

audio, fs = record_until_silence()
fd, tmp_path = tempfile.mkstemp(suffix=".wav")
os.close(fd)
save_wav(tmp_path, audio, fs)
lang_text = transcribe_whisper(tmp_path, "en").strip().lower()
os.remove(tmp_path)

language_map = {
    "tamil": "ta", "hindi": "hi", "telugu": "te", "malayalam": "ml",
    "kannada": "kn", "gujarati": "gu", "odia": "or", "kashmiri": "ks", "rajasthani": "raj"
}
lang_code = language_map.get(lang_text, "en")

confirm_text = {
    "ta": "நீங்கள் தமிழ் தேர்ந்தெடுத்துள்ளீர்கள்.",
    "hi": "आपने हिंदी चुनी है।",
    "en": "You selected English.",
    "te": "మీరు తెలుగు ఎంచుకున్నారు.",
    "ml": "നിങ്ങൾ മലയാളം തിരഞ്ഞെടുക്കുക.",
    "kn": "ನೀವು ಕನ್ನಡ ಆಯ್ಕೆ ಮಾಡಿಕೊಂಡಿದ್ದೀರಿ.",
    "gu": "તમે ગુજરાતી પસંદ કરી છે.",
    "or": "ଆପଣ ଓଡ଼ିଆ ଚୟନ କରିଛନ୍ତି।",
    "ks": "تُهند کشمیری منتخب کی گئی ہے۔",
    "raj": "थारे राजस्थानी चुनी है।"
}
speak(confirm_text.get(lang_code, confirm_text["en"]), lang_code)

# --- Main App ---
scraped_text = scrape_sdnbvc()

if st.button("🎙 Ask a Question"):
    with st.spinner("Recording..."):
        audio, fs = record_until_silence()
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        save_wav(tmp_path, audio, fs)
        query = transcribe_whisper(tmp_path, lang_code)
        os.remove(tmp_path)

        answer = find_answer(query, scraped_text)
        speak(answer, lang_code)

        # Ask for callback
        speak("Would you like a callback?", lang_code)
        audio, fs = record_until_silence()
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        save_wav(tmp_path, audio, fs)
        reply = transcribe_whisper(tmp_path, lang_code)
        os.remove(tmp_path)

        if "yes" in reply.lower():
            log_callback_request(query, "9876543210", lang_code)
            confirm_callback(lang_code)
