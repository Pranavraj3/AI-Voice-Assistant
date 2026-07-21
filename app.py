"""
AI Voice Assistant - Flask Backend
-----------------------------------
Pipeline: Speech (mic) -> Gemini (speech-to-text) -> Gemini (generate answer)
          -> gTTS (text-to-speech) -> Speech (reply)

BEGINNER NOTE: You do NOT need to edit the pipeline code below.
The ONLY thing you must do before running this is set your Gemini API key
as an environment variable (see instructions below) — never paste it
directly into this file, especially if this project goes on GitHub.

DEPLOYMENT NOTE: This version transcribes audio using the Gemini API
instead of a locally-loaded Whisper model. That's a deliberate choice —
Whisper + torch need 500MB-900MB+ of RAM just to load, which doesn't fit
in Render's free 512MB tier. Sending audio straight to Gemini keeps this
app's memory footprint small enough to run there for free.
"""

from flask import Flask, render_template, request, jsonify, send_file
from google import genai
from google.genai import types
from gtts import gTTS
from datetime import datetime
import os
import re
import subprocess
import uuid

def strip_markdown(text):
    """Removes common markdown symbols so TTS doesn't read them aloud."""
    text = re.sub(r"[*_#`]", "", text)          # bold/italic/header/code symbols
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)  # bullet points
    return text.strip()

app = Flask(__name__)

# ---------------------------------------------------------------
# API KEY SETUP (manual step — do this once per terminal session
# when running locally; on Render, set it as an Environment
# Variable in the service dashboard instead)
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
        "GEMINI_API_KEY is not set. Locally: run "
        "'set GEMINI_API_KEY=your_key_here' then run python app.py again. "
        "On Render: add GEMINI_API_KEY under your service's Environment tab."
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

print("Models ready. Server is starting!")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def convert_to_wav(input_path, output_path):
    """
    Converts whatever audio format the browser recorded (webm/ogg/etc.)
    into a 16kHz mono WAV file, which Gemini's audio input reliably
    accepts. Uses the ffmpeg binary directly (no extra Python package
    needed) — Render's native Python environment ships with ffmpeg
    preinstalled, so this works there with no extra setup.
    """
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore"))


def transcribe_audio(wav_path):
    """
    STEP 1: Speech -> Text, done by sending the audio straight to Gemini
    instead of running a local Whisper model. This is the main change
    that makes this app small enough to fit Render's free 512MB tier.
    """
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=[
            "Transcribe the following audio exactly as spoken. Return ONLY "
            "the transcript text — no quotation marks, labels, or extra "
            "commentary. If the audio is silent, unintelligible, or has no "
            "speech, return an empty string.",
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        ],
    )
    return (response.text or "").strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_audio():
    """
    Receives the recorded audio from the browser, runs it through:
    1. Gemini  -> transcribes speech to text
    2. Gemini  -> generates a text answer (same chat session, so it
                  remembers earlier turns)
    3. gTTS    -> converts the answer back to speech
    Returns the answer text + a link to the generated audio reply.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400

    audio_file = request.files["audio"]

    file_id = uuid.uuid4().hex
    raw_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_raw")
    wav_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.wav")

    audio_file.save(raw_path)

    try:
        # STEP 0: normalize whatever the browser recorded into WAV
        convert_to_wav(raw_path, wav_path)

        # STEP 1: Speech -> Text (via Gemini)
        user_text = transcribe_audio(wav_path)

        if user_text == "":
            return jsonify({"error": "Could not understand audio. Please try again."}), 400

        # STEP 2: Generate an answer with Gemini (using the ongoing chat
        # session, so it remembers earlier turns — e.g. your name)
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
    except RuntimeError as e:
        return jsonify({"error": f"Could not process audio: {e}"}), 400
    finally:
        # Clean up the temporary input files (keep the mp3 reply — it's
        # still needed by the /audio/<filename> route below)
        for path in (raw_path, wav_path):
            if os.path.exists(path):
                os.remove(path)


@app.route("/audio/<filename>")
def get_audio(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, mimetype="audio/mpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
