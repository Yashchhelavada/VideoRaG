# 🎥 Video-Based RAG (AI Teaching Assistant)

A Retrieval-Augmented Generation (RAG) based AI chatbot that transforms your video lectures and tutorials into an interactive, searchable knowledge base. Instead of scrubbing through hours of video to find a specific topic, simply ask a question, and the assistant will pinpoint the exact video and timestamp while providing a concise summary.

## 🌟 Problem Statement

Finding specific information in a large collection of video tutorials or recorded lectures is traditionally a frustrating and time-consuming process. Imagine reviewing for an exam and needing to clarify just one topic—manually searching through hours of video is highly inefficient. 

This AI Assistant solves that by allowing you to feed in your videos and retrieve exact timestamps, source video references, and summaries simply by asking natural language questions.

## ✨ Features
- **Video Transcription:** Automatically extracts audio and converts it to text with precise timestamps.
- **Semantic Search:** Uses vector embeddings to find the most relevant video segments based on meaning, not just keyword matching.
- **Context-Aware Answers:** Leverages a powerful LLM to synthesize the retrieved information and directly answer your query.
- **Local & Cloud Hybrid Processing:** Uses local models (Whisper, Ollama) to maintain privacy and manage costs, while leveraging high-performance LLMs via API for reasoning.

## 🛠 Tech Stack
- **Audio/Video Processing:** `ffmpeg`
- **Speech-to-Text:** [OpenAI Whisper](https://github.com/openai/whisper) (Runs locally)
- **Embeddings:** [Ollama](https://ollama.com/) with the `nomic-embed-text` model
- **Vector Storage & Similarity Search:** `pandas`, `joblib`, `scikit-learn` (`cosine_similarity`)
- **LLM Engine:** [Groq API](https://groq.com/) using `llama-3.3-70b-versatile`

## ⚙️ Prerequisites
Before running the project, ensure you have the following installed:
1. **Python 3.8+**
2. **FFmpeg:** Required for extracting audio from videos.
3. **Ollama:** Installed and running locally. Pull the required embedding model before starting:
   ```bash
   ollama run nomic-embed-text
   ```
4. **Groq API Key:** Get your API key from Groq and set it up as an environment variable.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Video-Based-RAG
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   API_KEY=your_groq_api_key_here
   ```

## 📖 How to Use

Follow these steps to process your videos and start querying:

### Step 1: Collect your videos
Move all your target video files into the designated directory for the scripts.

### Step 2: Extract Audio
Run the script to convert videos to `mp3` format:
```bash
python video_to_mp3.py
```

### Step 3: Transcribe & Chunk
Convert the `mp3` audio files into JSON chunks containing text and timestamps using Whisper:
```bash
python mp3_to_json.py
```
*(Note: You can modify the Whisper model size inside the script based on your PC's specifications, e.g., "tiny", "base", "small", "medium", or "large".)*

### Step 4: Generate Embeddings
Convert the JSON transcript chunks into vector embeddings and save them locally:
```bash
python preprocess_json.py
```
*(Note: The embeddings are stored locally as a Joblib pickle file. For larger datasets, consider adapting the script to use a Vector Database like ChromaDB or FAISS.)*

### Step 5: Query the Assistant
Load the embeddings and interact with the chatbot:
```bash
python process_incoming.py
```
You will be prompted to ask a question. The system will retrieve the most relevant transcript segments and generate a helpful response, guiding you to the exact video and timestamp!
