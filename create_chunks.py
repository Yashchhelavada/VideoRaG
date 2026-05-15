import whisper
import json
import os
from tqdm import tqdm

model = whisper.load_model("small").to("cuda")
audios = os.listdir("audios")

for audio in tqdm(audios, desc="Transcribing files"):
    

    result = model.transcribe(audio = f"audios/{audio}",
    # result = model.transcribe(audio = f"audios/sample.mp3",                          
                        word_timestamps = False )
    


    chunks = []
    for segment in result["segments"]:
        chunks.append({"audio": audio,"start": segment["start"],"end": segment["end"], "text": segment["text"]})

    chunks_with_metedata = {"chunks": chunks, "text": result["text"]}

    with open(f"json/{audio}.json", "w") as f:
        json.dump(chunks_with_metedata, f, indent=4)