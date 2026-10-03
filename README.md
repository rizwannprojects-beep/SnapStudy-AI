# SnapStudy AI 📚

A Visual Study Assistant powered by Gemini and Streamlit.

SnapStudy AI helps students learn difficult concepts, solve challenging questions step-by-step, decode diagrams, and understand code snippets from textbook photos or notes. When a study session finishes, SnapStudy AI generates a concise study revision summary and emails it directly to the student via Gmail SMTP.

---

## ✨ Features

- **Study material image upload**: Supports JPG, JPEG, PNG, and WEBP formats.
- **Gemini Vision analysis**: Accurately interprets handwritten notes, textbook pages, math formulas, code, and diagrams.
- **Text/transcription extraction**: Transcribes text, equations, and code snippets directly from the uploaded material.
- **Step-by-step explanations**: Solves quantitative and theoretical problems clearly showing each logical step.
- **Mathematical problem solving**: Handles algebra, calculus, arithmetic, and quantitative reasoning with LaTeX formatting.
- **Programming/code understanding**: Decodes algorithms, syntax, edge cases, and optimizations.
- **Follow-up questions**: Interactive study chat anchored strictly to the educational context of the uploaded content.
- **Practice question generation**: Generates related practice problems to reinforce student learning.
- **Study summary**: High-yield revision sheet summarizing key formulas, definitions, and takeaways.
- **Email summary**: Dispatches personalized study notes to the student's email using Python's standard `smtplib`.
- **Gemini retry/fallback handling**: Seamless automatic model fallback pipeline (`gemini-3.8-flash` → `gemini-3.6-flash` → `gemini-3.5-flash-lite`) for temporary demand spikes (503 / 429) without user downtime.

---

## 🛠️ Technology Stack

- **Python** (Core application logic)
- **Streamlit** (Interactive web application interface)
- **Google Gemini API** (via official `google-genai` Python SDK)
- **Pillow** (Image format handling & validation)
- **SMTP** (Python standard library `smtplib` for email delivery)

---

## 📁 Project Structure

```text
SnapStudy-AI/
├── app.py                      # Main Streamlit application with chat & vision workflow
├── prompts.py                  # Educational system instructions & prompt templates
├── requirements.txt            # Minimal dependencies (streamlit, google-genai, pillow)
├── README.md                   # Complete documentation and setup guide
├── .gitignore                  # Ignores secrets.toml, virtual environments, and caches
└── .streamlit/
    └── secrets.toml.example    # Configuration template for Gemini and Gmail SMTP
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)
- A Gmail account with an **App Password** (for sending study summaries)

### 2. Navigate to the Project Directory
```bash
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

### 5. Configure Secrets
1. Copy the example secrets file:
   - **Windows**:
     ```powershell
     Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
     ```
   - **macOS / Linux**:
     ```bash
     cp .streamlit/secrets.toml.example .streamlit/secrets.toml
     ```
2. Open `.streamlit/secrets.toml` in your editor and enter your values:
   ```toml
   # Google Gemini API Key
   GEMINI_API_KEY = "your_actual_gemini_api_key"

   # Preferred model (gemini-3.8-flash with automatic fallback)
   GEMINI_MODEL = "gemini-3.8-flash"

   # Gmail SMTP Settings
   SMTP_EMAIL = "your_sender_gmail@gmail.com"
   SMTP_APP_PASSWORD = "your_16_character_app_password"
   SMTP_SERVER = "smtp.gmail.com"
   SMTP_PORT = 587
   ```

> ⚠️ **Important Security Note**: The `.streamlit/secrets.toml` file contains your private credentials and is already added to `.gitignore`. Never commit or share this file publicly.

---

## 🔑 How to Obtain Credentials

### 1. Google Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **Get API key** > **Create API key**.
4. Copy the generated key into `.streamlit/secrets.toml` under `GEMINI_API_KEY`.

### 2. Gmail App Password (for Email Summaries)
Gmail does not allow standard account passwords for SMTP. You must generate an App Password:
1. Go to your [Google Account Security Settings](https://myaccount.google.com/security).
2. Enable **2-Step Verification** (if not already enabled).
3. Under *How you sign in to Google*, click on **2-Step Verification**, scroll down and select **App passwords**.
4. Enter an app name (e.g. `SnapStudy AI`) and click **Create**.
5. Copy the generated 16-character password into `.streamlit/secrets.toml` under `SMTP_APP_PASSWORD`.

---

## 🏃 Running the Application

Launch the Streamlit web application:
```bash
python -m streamlit run app.py
```

Streamlit will start a local web server (usually at `http://localhost:8501`).

---

## 💡 How to Use the App

1. **Enter Your Details**: Fill in your Name and Email in the sidebar onboarding form.
2. **Upload Study Material**: Drop an image of textbook exercises, homework questions, handwritten notes, equations, diagrams, or code into the sidebar uploader.
3. **Analyze**: Click **Analyze Material** to let Gemini Vision interpret the image and present an initial breakdown.
4. **Ask Follow-Up Questions**: Use the chat bar to ask for clarification, simpler analogies, or more practice problems.
5. **Get Your Revision Notes**: Click **Send Summary to Email** in the sidebar to have Gemini compile a study summary and dispatch it straight to your inbox!
