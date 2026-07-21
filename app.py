"""
AI Voice Assistant - Flask Backend
-----------------------------------
Pipeline: Speech (mic) -> Whisper (speech-to-text) -> Gemini (generate answer)
          -> gTTS (text-to-speech) -> Speech (reply)

BEGINNER NOTE: You do NOT need to edit the pipeline code below.
The ONLY thing you must do before running this is set your Gemini API key
as an environment variable (see instructions below) — never paste it
directly into this file, especially if this project goes on GitHub.
"""

from flask import Flask, render_template, request, jsonify, send_file
import whisper
from google import genai
from google.genai import types
from gtts import gTTS
from datetime import datetime
import os
import re
import uuid


def strip_markdown(text):
    """Removes common markdown symbols so TTS doesn't read them aloud."""
    text = re.sub(r"[*_#`]", "", text)          # bold/italic/header/code symbols
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)  # bullet points
    return text.strip()

app = Flask(__name__)

# ---------------------------------------------------------------
# API KEY SETUP (manual step — do this once per terminal session)
# ---------------------------------------------------------------
# Windows Command Prompt, BEFORE running "python app.py":
#     set GEMINI_API_KEY=your_key_here
#
# To make it permanent instead of per-session:
#     setx GEMINI_API_KEY "your_key_here"
#     (then close and reopen Command Prompt)
#
# Get a key at: https://aistudio.google.com/apikey
# ---------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Run: set GEMINI_API_KEY=your_key_here  "
        "then run python app.py again."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------
# CHAT SESSION (this is what gives the assistant memory)
# ---------------------------------------------------------------
# Instead of calling generate_content() fresh each time (which forgets
# everything instantly), we create ONE chat session when the server
# starts and reuse it for every request. Gemini automatically keeps
# track of the full conversation inside this object.
#
# NOTE: since this is stored in memory, it resets if you restart
# app.py — that's expected and fine for a portfolio demo. It does NOT
# reset if you just refresh the browser page, since the memory lives
# on the server, not in the browser.
# ---------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a voice assistant. Your reply will be converted to speech and "
    "read aloud, so respond in plain, natural spoken sentences only — no "
    "markdown, asterisks, headings, bullet points, or numbered lists. "
    "Match your reply length to what's being asked: "
    "for simple factual questions (like a name, a date, a yes/no, a single "
    "fact), answer in 1-2 short sentences and nothing else. "
    "for requests to explain, describe, or teach a concept (e.g. containing "
    "words like 'explain', 'what is', 'how does X work', 'describe', "
    "'tell me about'), give a clear, complete spoken explanation — a few "
    "sentences covering the key idea and a simple example if it helps, "
    "still without headings or lists, just natural spoken paragraphs. "
    "You will be given the current date and time before each message — "
    "use it if the question needs it, otherwise ignore it. Remember "
    "details the user tells you (like their name) and use them naturally "
    "in later replies."
)

chat = client.chats.create(
    model="gemini-flash-latest",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        max_output_tokens=500,   # generous ceiling; the instruction above controls actual length
    )
)

# ---------------------------------------------------------------
# LOAD MODELS (this happens once, when the server starts)
# ---------------------------------------------------------------
print("Loading Whisper model (speech-to-text)... this may take a minute the first time.")
stt_model = whisper.load_model("tiny")  # options: tiny, base, small (bigger = slower but more accurate)

print("Models loaded. Server is ready!")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_audio():
    """
    Receives the recorded audio from the browser, runs it through:
    1. Whisper -> converts speech to text
    2. Gemini  -> generates a text answer
    3. gTTS    -> converts the answer back to speech
    Returns the answer text + a link to the generated audio reply.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400

    audio_file = request.files["audio"]

    input_filename = f"{uuid.uuid4()}.wav"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    audio_file.save(input_path)

    # STEP 1: Speech -> Text
    result = stt_model.transcribe(input_path)
    user_text = result["text"].strip()

    if user_text == "":
        return jsonify({"error": "Could not understand audio. Please try again."}), 400

    # STEP 2: Generate an answer with Gemini (using the ongoing chat session,
    # so it remembers earlier turns — e.g. your name)
    now_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    message_with_time = f"[Current date and time: {now_str}] {user_text}"

    response = chat.send_message(message_with_time)
    reply_text = strip_markdown(response.text)

    # STEP 3: Text -> Speech
    output_filename = f"{uuid.uuid4()}.mp3"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    tts = gTTS(text=reply_text, lang="en")
    tts.save(output_path)

    return jsonify({
        "user_text": user_text,
        "reply_text": reply_text,
        "audio_url": f"/audio/{output_filename}"
    })


@app.route("/audio/<filename>")
def get_audio(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(path):
        return jsonify({"error": "Audio file not found"}), 404

    return send_file(path, mimetype="audio/mpeg")


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
