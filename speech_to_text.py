import whisper
import json

model = whisper.load_model("small").to("cpu")

result = model.transcribe(
    audio="audios/tutorial-1.mkv.mp3",
    word_timestamps=False
)

formatted = {
    "chunks": []
}

for segment in result["segments"]:
    formatted["chunks"].append({
        "audio": "tutorial-1.mkv.mp3",
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })

print(formatted)

with open("jsons/tutorial-1.json", "w") as f:
    json.dump(formatted, f, indent=4)
