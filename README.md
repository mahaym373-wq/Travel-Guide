# 🇮🇳 Smart India Travel Guide

![Project Banner](https://img.shields.io/badge/Status-Active-success)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393)

**Smart India Travel Guide** is a deterministic, semantic-network-powered chatbot designed to provide accurate, explainable, and enthusiastic travel advice for Indian destinations. 

Unlike traditional chatbots that rely on external Large Language Models (LLMs) which can hallucinate or require API keys, this project uses **pure symbolic reasoning** over a custom-built knowledge base. The result is a robust, lightweight, and completely self-contained conversational agent.

---

## ✨ Features

- **Deterministic Reasoning Engine**: No LLMs involved. All intelligence is powered by graph traversal over a structured semantic network.
- **Bi-Directional Inference**: Can answer questions via both forward reasoning (e.g., "What is the best time to visit Goa?") and reverse reasoning (e.g., "Which places are good for beaches?").
- **Explainable AI (XAI)**: Every response includes a transparent reasoning path (the exact nodes and edges traversed in the knowledge graph) so you can see *how* the bot arrived at its answer.
- **Enthusiastic Persona**: Implements rule-based Natural Language Processing (NLP) to provide warm, engaging, and child-friendly conversational responses.
- **Modern Web Interface**: Clean, responsive frontend with a chat UI to interact with the guide seamlessly.

---

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, Pydantic
- **Frontend**: HTML5, Vanilla JS, CSS
- **Core Logic**: Custom implementation of Semantic Networks (`network.py`) and Rule-Based NLP (`nlp.py`).

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/your-username/smart-india-travel-guide.git
   cd "smart-india-travel-guide"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required dependencies**:
   ```bash
   pip install fastapi uvicorn pydantic
   ```

### Running the Application

**Option 1: Using the provided script (Windows only)**
Double-click `start.bat` in the root folder, or run it via command prompt:
```cmd
start.bat
```

**Option 2: Manually via Uvicorn**
Start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

Once the server is running, open your web browser and navigate to:
**http://localhost:8000/**

---

## 📂 Project Structure

```text
├── main.py        # FastAPI server, endpoints, and application lifecycle
├── data.py        # The Knowledge Base (destinations, facts, relationships)
├── network.py     # Graph-traversal engine for inference
├── nlp.py         # Rule-based chat processor and intent classification
├── index.html     # Frontend UI served at the root endpoint
├── start.bat      # Automation script to start the server
└── README.md      # Project documentation
```

---

## 🧠 How it Works

1. **Input Parsing**: The user's query is cleaned and parsed to extract known entities present in `data.py`.
2. **Intent Classification**: The query is mapped to a specific intent (e.g., seeking attributes, finding related places, or general small talk).
3. **Graph Traversal**: `network.py` navigates the knowledge graph using forward or reverse chaining based on the intent.
4. **Natural Language Generation**: The raw data output is fed into conversational templates in `nlp.py` to produce a human-friendly answer.
5. **Response Delivery**: The response, along with the reasoning path, is returned to the frontend via the `/api/chat` REST endpoint.

---

## 🤝 Contributing
Contributions are welcome! If you'd like to add new destinations to the knowledge base, enhance the NLP parsing, or improve the UI, feel free to fork this repository and submit a Pull Request.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
