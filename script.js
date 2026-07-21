// This file handles:
// 1. Recording your voice using the browser microphone
// 2. Animating the mic button + pipeline stages
// 3. Sending the recording to Flask (/process)
// 4. Rendering the transcript into the session log and playing the reply

const recordBtn = document.getElementById("recordBtn");
const statusText = document.getElementById("status");
const logBody = document.getElementById("logBody");
const audioPlayer = document.getElementById("audioPlayer");

const stageListen = document.getElementById("stage-listen");
const stageThink = document.getElementById("stage-think");
const stageSpeak = document.getElementById("stage-speak");

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

function resetStages() {
  [stageListen, stageThink, stageSpeak].forEach(s => {
    s.classList.remove("active", "done");
  });
}

function setStage(stage, state) {
  // state: "active" or "done"
  stage.classList.remove("active", "done");
  stage.classList.add(state);
}

recordBtn.addEventListener("click", async () => {
  if (!isRecording) {
    // Start recording
    resetStages();
    setStage(stageListen, "active");

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = sendAudioToServer;

    mediaRecorder.start();
    isRecording = true;
    recordBtn.classList.add("recording");
    statusText.textContent = "Listening — press again to stop";
  } else {
    // Stop recording
    mediaRecorder.stop();
    isRecording = false;
    recordBtn.classList.remove("recording");
    setStage(stageListen, "done");
    setStage(stageThink, "active");
    statusText.textContent = "Thinking...";
  }
});

async function sendAudioToServer() {
  const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");

  try {
    const response = await fetch("/process", {
      method: "POST",
      body: formData
    });
    const data = await response.json();

    if (data.error) {
      statusText.textContent = data.error;
      resetStages();
      return;
    }

    setStage(stageThink, "done");
    setStage(stageSpeak, "active");

    appendLog("You", data.user_text, "user");
    appendLog("Assistant", data.reply_text, "bot");

    audioPlayer.src = data.audio_url;
    audioPlayer.classList.add("visible");
    audioPlayer.play();

    audioPlayer.onended = () => setStage(stageSpeak, "done");

    statusText.textContent = "Press to speak";
  } catch (err) {
    statusText.textContent = "Something went wrong. Check the server console.";
    resetStages();
    console.error(err);
  }
}

function appendLog(who, text, cls) {
  const emptyMsg = logBody.querySelector(".log__empty");
  if (emptyMsg) emptyMsg.remove();

  const line = document.createElement("p");
  line.className = `log__line ${cls}`;
  line.innerHTML = `<span class="tag">${who}</span>${text}`;
  logBody.appendChild(line);
  logBody.scrollTop = logBody.scrollHeight;
}
