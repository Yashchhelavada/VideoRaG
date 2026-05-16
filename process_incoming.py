import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity  
import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
API_KEY = os.getenv("API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY)

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-text",
        "input": text_list
    })


    embedding = r.json()["embeddings"] 
    return embedding


def inference_openai(prompt):
    print("Thinking...")
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
            {"role": "system", "content": "You are a helpful teaching assistant for a Data Science course."},
            {"role": "user", "content": prompt}
        ]
    )
    input = prompt
    

    return response.choices[0].message.content


df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a question:")
question_embedding = create_embedding([incoming_query])[0]

# Find similarities of question_embedding with other embeddings

similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten() # type: ignore
print(similarities)
top_results = 5
max_idx = similarities.argsort()[::-1][0:top_results]
print(max_idx)
new_df = df.iloc[max_idx]
# print(new_df[['audio', 'text','start','end']])

prompt = f''' I am teaching a class on the following topics: Machine learning, Data Science, and Artificial Intelligence.
Here are video chunks containing video, start time in seconds, end time in seconds,
the text at that time:
{new_df[['audio','start','end','text']].to_json()}
-------------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer where and how much
content is taught and in which video amd aslo give the brief summary of the topic and aslo guide the user to go
to that particular video. If user ask any irrelevant question, reply with "I don't know" and ask them to ask question 
only related to the course. 
'''
with open('prompt.txt','w') as f:
    f.write(prompt)

response = inference_openai(prompt)

with open("response.txt", "w") as f:
    f.write(response)



# for index, item in new_df.iterrows():
#     print(f"Audio: {item['audio']}, Start: {item['start']}, End: {item['end']}")
#     print(f"Text: {item['text']}")
#     print("-----")
