# 🌌 BioSpace NASA — AI Research Assistant 🚀

An intelligent research assistant built for NASA’s **space biology research**, featuring **RAG-powered document analysis**, **AI chat**, and **voice interaction** capabilities.

---

## 🛠️ Setup Guide

### Prerequisites

Before getting started, make sure you have:

* **Python 3.8+** 🐍 — [Download here](https://www.python.org/downloads/)
* **Node.js 18+** 📦 — [Download here](https://nodejs.org/)
* **Gemini API Key** 🔑 — [Get one from Google AI Studio](https://aistudio.google.com/)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/biospace-nasa.git
cd biospace-nasa
```

---

### 2. Backend Setup (Python) 🧠

#### Step 1: Create and activate virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

#### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Set up environment variables

Create a `.env` file in the **root or backend** directory and add your Gemini API key:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

#### Step 4: Run the Python server

```bash
python main.py
```

✅ The backend will start at: **[http://localhost:5000](http://localhost:5000)**

---

### 3. Frontend Setup (Node.js) ⚛️

#### Step 1: Install dependencies

Open a new terminal, navigate to your frontend folder, and run:

```bash
npm install
```

#### Step 2: Start the development server

```bash
npm run dev
```

✅ The frontend will run at: **[http://localhost:5173](http://localhost:5173)**

---

### 4. Add Research Documents 📄

Place your **NASA space biology PDFs** in the `pmc_docs/` folder:

```bash
# Example:
pmc_docs/
├── plant_growth_microgravity.pdf
├── microbial_response_spaceflight.pdf
```

> 🧠 The system automatically processes new documents and builds embeddings for them.

---

## 🎮 How to Use

### 💬 Chat Interface

1. Type your question in the input box.
2. Press **Enter** or click **Send ➡️**.
3. View **citations** by clicking the numbered sources beside responses.

### 🎤 Voice Features

* **🎙️ Mic button** – Speak your questions directly.
* **🤖 Robot mode** – Engage in full voice conversations.
* **🔊 Read button** – Listen to the AI’s response aloud.

---

## 🧩 API Reference

| Endpoint  | Method | Description          |
| --------- | ------ | -------------------- |
| `/query`  | POST   | Chat with RAG model  |
| `/health` | GET    | Check backend status |
| `/reset`  | POST   | Rebuild embeddings   |

**Example:**

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Effects of microgravity on plant growth"}'
```

---

## 🐛 Troubleshooting

| Problem                  | Possible Fix                                           |
| ------------------------ | ------------------------------------------------------ |
| ❌ Backend not connecting | Ensure `python main.py` is running                     |
| ❌ No documents found     | Add PDFs to `pmc_docs/` and restart backend            |
| ❌ Gemini API error       | Check `.env` and ensure key is valid                   |
| ❌ Voice not working      | Use Chrome/Edge and allow microphone access            |
| ❌ Module not found       | Run `pip install -r requirements.txt` or `npm install` |

### 🔍 Debugging Tips

* Check **backend logs** for RAG and embedding info
* Open **browser console** for frontend errors
* Make sure your **virtual environment is activated** before running backend

---

## 💡 Pro Tips

* 🧠 First run may take time — embeddings are being generated.
* 📘 Large PDFs (1000+ pages) are supported efficiently.
* 🔁 Restart backend after adding new PDFs.
* 🎧 Voice features work best in **Chrome**.

---

## 🆘 Need Help?

1. Check backend logs from `python main.py`
2. Verify `.env` contains a valid Gemini key
3. Make sure all dependencies are installed
4. Check browser console for frontend issues

---

*Start exploring space biology research — from microbial experiments to plant growth in microgravity!* 🌱🚀

---
