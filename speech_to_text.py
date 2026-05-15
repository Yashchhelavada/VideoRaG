import whisper
import json

model = whisper.load_model("small").to("cuda")
result = model.transcribe(audio = "audios/sample.mp3",
                          word_timestamps = False )
print(result["segments"])
chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"],"end": segment["end"], "text": segment["text"]})

print(chunks)
with open("output.json", "w") as f:
    json.dump(chunks, f, indent=4)             

