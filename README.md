# SnapStudy AI 📚
### Multimodal AI Visual Study Assistant & Tutoring Dashboard

SnapStudy AI is an educational web application designed to help students master complex academic materials. Built with **Streamlit** and powered by **Google Gemini Vision** via the official `google-genai` SDK, SnapStudy AI transcribes, analyzes, and explains textbook pages, handwritten notes, mathematical equations, technical diagrams, and code snippets from uploaded images. It supports interactive follow-up tutoring and can generate and email high-yield revision summaries directly to the student.

---

## 📌 Problem Statement

Students frequently encounter complex homework questions, handwritten derivations, architectural diagrams, and algorithmic code snippets while preparing for exams. Traditional search engines and standard text-only LLMs present significant friction:
1. Students cannot easily copy/paste equations, circuit diagrams, or handwritten lecture notes.
2. Default AI chatbots often give immediate raw answers without teaching the foundational principles or step-by-step derivation.
3. Students lack an automated mechanism to synthesize their interactive learning sessions into consolidated, high-yield revision sheets for quick exam prep.

---

## 💡 Solution

**SnapStudy AI** solves this by providing a unified, multimodal learning dashboard:
- **Instant Visual Ingestion**: Upload photos or screenshots directly in PNG, JPG, or WEBP format.
- **Pedagogical Structure**: Deconstructs every uploaded problem into structured educational components: *Material Overview*, *Transcription*, *Concept Explanation*, *Step-by-Step Solution*, and *Practice Questions*.
- **Interactive Tutoring**: Allows students to ask follow-up questions anchored strictly to their study material.
- **Automated Revision Summaries**: Compiles the entire session's lessons into an exam revision sheet and delivers it via SMTP to the student's email.
- **Enterprise Resilience**: Features an automatic multi-model fallback pipeline that shields students from API rate limits and temporary traffic spikes.

---

## ✨ Main Features

- 📷 **Multimodal Image Analysis**: Accurately interprets handwritten classroom notes, textbook problems, mathematical formulas, and scientific diagrams.
- 🧮 **Step-by-Step Problem Solving**: Generates detailed, step-by-step mathematical solutions with LaTeX formula rendering via KaTeX.
- 💻 **Code & Diagram Explanation**: Explains programming logic, syntax, edge cases, time/space complexity, and flowchart workflows.
- 💬 **Interactive Chat Composer**: Full-width study composer with quick-prompt chips (*Explain this simply*, *Give me an example*, *Create practice questions*, *Explain step by step*).
- 🎯 **Targeted Practice Generator**: Proposes subject-specific practice questions to test and reinforce conceptual mastery.
- 📧 **Automated Revision Notes to Email**: Generates a formatted study sheet and emails it directly to the student via Gmail SMTP.
- 🛡️ **Zero-Downtime Model Fallback**: Automatically cascades through fallback models (`gemini-3.8-flash` → `gemini-3.6-flash` → `gemini-3.5-flash-lite`) if a temporary 503 or 429 is encountered.
- 🔒 **Safe In-Memory Processing**: Uploaded images are buffered in RAM and never written to disk, preventing persistent file exposure.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language & backend logic |
| **Streamlit** | Modern reactive web frontend and session state management |
| **Google GenAI SDK (`google-genai`)** | Official Python client for Gemini Vision and Multimodal models |
| **Pillow (`PIL`)** | In-memory image validation and format handling |
| **SMTP (`smtplib`, `email.mime`)** | Python standard library for dispatching revision summaries via TLS |
| **TOML** | Configuration and secret isolation (`secrets.toml`) |

---

## 🤖 AI & Gemini Vision Integration

SnapStudy AI connects to the Google Gemini API using the modern `google-genai` SDK:
- **Multimodal Pipeline**: Uploaded binary image data is wrapped into `types.Part.from_bytes` and transmitted alongside custom pedagogical prompts.
- **Curated Model Pipeline**:
  1. `gemini-3.8-flash` (Primary high-speed multimodal reasoning model)
  2. `gemini-3.6-flash` (Secondary fallback model)
  3. `gemini-3.5-flash-lite` (Tertiary high-availability fallback model)
- **Pedagogical Guardrails**: System instructions in `prompts.py` keep Gemini strictly focused on educational topics and prevent off-topic distractions.
- **Error Classification**: Identifies retryable errors (HTTP 503, 429) vs. permanent errors (HTTP 401, 400), executing immediate model failover with zero manual intervention required by the student.

---

## 🖼️ Supported Image Formats

SnapStudy AI accepts all standard image formats commonly captured by student smartphones, webcams, and screenshot tools:
- **PNG** (`image/png`)
- **JPG / JPEG** (`image/jpeg`)
- **WEBP** (`image/webp`)

*Maximum recommended file size: 200MB per file. Processing is performed completely in-memory.*

---

## 🔄 Application Workflow

```text
[ Student Profile Setup ]
          │
          ▼
[ Upload Study Material (PNG / JPG / WEBP) ]
          │
          ▼
[ In-Memory Binary Validation ]
          │
          ▼
[ Gemini Multimodal Vision Pipeline ] ──(If 503/429)──► [ Automatic Fallback Model ]
          │
          ▼
[ 5-Tier Structured Analysis Display ]
  ├── 🔍 What I Found
  ├── 🧠 Concept Explanation
  ├── 📝 Step-by-Step Solution
  ├── ✅ Final Answer
  └── 🎯 Practice Questions
          │
          ▼
[ Interactive Tutoring Chat & Quick Prompts ]
          │
          ▼
[ Generate Study Revision Summary ]
          │
          ▼
[ SMTP Dispatch to Student's Email ]
```

---

## 📁 Project Structure

```text
SnapStudy-AI/
├── app.py                      # Main Streamlit application, UI layout, & workflow
├── prompts.py                  # Educational prompts & structured system instructions
├── requirements.txt            # Application dependencies
├── README.md                   # Project documentation & setup instructions
├── .gitignore                  # Git ignore rules (secrets, virtualenv, caches)
├── .streamlit/
│   ├── secrets.toml            # Private local credentials (ignored by Git)
│   └── secrets.toml.example    # Configuration template for deployment
└── tests/
    ├── test_suite.py           # Automated 12-test comprehensive verification suite
    └── sample_materials/       # Test images (math_problem.png, physics_notes.jpg, code_snippet.webp)
```

---

## 💻 Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- A Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))
- (Optional) Gmail Account with an **App Password** for sending email summaries

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/SnapStudy-AI.git
cd SnapStudy-AI
```

### 3. Create and Activate a Virtual Environment

- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔐 Environment & Secrets Setup

1. Copy the template secrets file:
   - **Windows**:
     ```powershell
     Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
     ```
   - **macOS / Linux**:
     ```bash
     cp .streamlit/secrets.toml.example .streamlit/secrets.toml
     ```

2. Open `.streamlit/secrets.toml` and configure your credentials:
   ```toml
   # Google Gemini API Key (Required)
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"

   # Preferred primary model
   GEMINI_MODEL = "gemini-3.8-flash"

   # Gmail SMTP Settings for Email Summaries (Optional)
   SMTP_EMAIL = "your_email@gmail.com"
   SMTP_APP_PASSWORD = "your_16_char_app_password"
   SMTP_SERVER = "smtp.gmail.com"
   SMTP_PORT = 587
   ```

> 🔒 **Security Notice**: `.streamlit/secrets.toml` is strictly ignored by Git in `.gitignore`. Never commit or share your API keys or email passwords publicly.

---

## 🚀 How to Run Locally

Start the Streamlit application:
```bash
python -m streamlit run app.py
```

The web dashboard will automatically launch in your browser at:
```text
http://localhost:8501
```

---

## 🧪 Testing & Verification

SnapStudy AI includes an automated 12-test verification suite that tests all aspects of configuration, security, failure resilience, and multimodal calls:

```bash
# Run the automated test suite
python tests/test_suite.py
```

### Test Coverage Summary:
- **Test 1**: Application startup and model pipeline hierarchy
- **Test 2**: Student email format validation
- **Test 3**: Image format ingestion (PNG, JPG, WEBP)
- **Test 4, 6, 12**: Live Gemini Vision analysis, interactive chat, and latency benchmarks
- **Test 5**: HTTP 503, 429, 401, and 400 error classification
- **Test 7**: Text-only chat handling without image context
- **Test 8**: Session state clearing and workspace reset
- **Test 9**: SMTP email error handling and credential validation
- **Test 10**: Security isolation audit (verifying no hardcoded keys)
- **Test 11**: End-to-end failover resilience simulation

---

## 📸 Screenshots & UI Showcase

| Screen | Description |
|---|---|
| **Study Dashboard** | Clean, light-theme educational interface with profile controls and format badges. |
| **Visual Ingestion** | Full-width dashed dropzone supporting drag-and-drop for PNG, JPG, and WEBP. |
| **Structured Breakdown** | 5 distinct visual cards: *What I Found*, *Concept Explanation*, *Step-by-Step*, *Final Answer*, and *Practice*. |
| **Interactive Composer** | Full-width document-flow chat input with suggestion chips and responsive send button. |

---

## 🔮 Future Enhancements

- 📄 **Multi-Page PDF Support**: Ability to upload and parse multi-page syllabus PDFs or complete exam papers.
- 🎙️ **Voice Interaction**: Integration of speech-to-text and audio feedback for hands-free audio study sessions.
- 🗂️ **Flashcard Export**: One-click export of generated practice questions into Anki and Quizlet-compatible CSV formats.
- 📊 **Progress Analytics**: Student learning history dashboard tracking completed topics and mastered concepts over time.

---

## 👨‍💻 Author & Project Credits

- **Project**: SnapStudy AI – Visual Study Assistant
- **Curriculum / Category**: BCA Final-Year Project / AI & Cloud Computing Portfolio
- **Developer**: Rizwan
- **AI Platform**: Google Gemini API via `google-genai` SDK
- **UI Framework**: Streamlit Open Source
