"""
SnapStudy AI - AI-Powered Visual Study Assistant
Built with Streamlit and Google Gemini Vision via the official google-genai SDK.
"""

import os
import random
import re
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Callable, Optional, Tuple

import streamlit as st
from google import genai
from google.genai import types

from prompts import (
    INITIAL_IMAGE_ANALYSIS_PROMPT,
    STUDY_ASSISTANT_SYSTEM_PROMPT,
    SUMMARY_GENERATION_PROMPT,
)

# ==============================================================================
# 1. Configuration & Secret Helpers
# ==============================================================================

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely retrieves a configuration key from st.secrets or os.environ.
    Prevents runtime KeyError exceptions if secrets.toml is not yet populated.
    """
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        # st.secrets might be unavailable or unconfigured
        pass
    env_val = os.environ.get(key)
    if env_val:
        return env_val.strip()
    return default


def is_valid_email(email: str) -> bool:
    """Basic email regex validator."""
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return bool(re.match(pattern, email.strip()))


# ==============================================================================
# Global Model Configuration & Fallback Order
# ==============================================================================
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


def get_model_pipeline() -> list:
    """
    Returns the configured Gemini models fallback pipeline.
    Preserves gemini-3.8-flash as primary (or any override from secrets).
    """
    configured = get_secret("GEMINI_MODEL")
    if configured:
        return [configured] + [m for m in GEMINI_MODELS if m != configured]
    return list(GEMINI_MODELS)


# ==============================================================================
# 2. Gemini Client & Immediate Model Fallback Logic (google-genai SDK)
# ==============================================================================

class GeminiServiceError(Exception):
    """User-friendly custom exception for Gemini API failures."""
    def __init__(
        self,
        user_message: str,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code
        self.original_error = original_error


def classify_gemini_error(e: Exception, model_name: str) -> Tuple[bool, str, Optional[int]]:
    """
    Analyzes an exception from google-genai or network layers.
    Returns:
        (is_retryable: bool, user_friendly_message: str, status_code: Optional[int])
    """
    err_str = str(e).lower()
    code = getattr(e, "code", None)

    # Permanent Client Errors (DO NOT FALLBACK / RETRY)
    # 1. 401 Unauthorized / Invalid API Key
    if code == 401 or "api_key_invalid" in err_str or "api key not valid" in err_str or "unauthenticated" in err_str:
        return (
            False,
            "🔐 AI configuration needs attention. Your Gemini API key appears invalid or expired. Please check GEMINI_API_KEY in .streamlit/secrets.toml.",
            401,
        )

    # 2. 403 Forbidden / Permission Denied
    if code == 403 or "permission_denied" in err_str:
        return (
            False,
            "🔐 AI configuration needs attention. Ensure the Gemini API is enabled for your project.",
            403,
        )

    # 3. 404 Model Not Found
    if code == 404 or "not_found" in err_str or ("model" in err_str and "not found" in err_str):
        return (
            False,
            f"🔄 The selected AI model is unavailable. Switching to a supported model. ('{model_name}')",
            404,
        )

    # 4. 400 Invalid Argument / Malformed Request / Unsupported format
    if code == 400 or "invalid_argument" in err_str or "unsupported" in err_str:
        return (
            False,
            "The study material or prompt could not be processed by Gemini (Invalid Request). Please try a clearer image (PNG, JPG, WEBP).",
            400,
        )

    # Temporary / Transient Errors (ELIGIBLE FOR IMMEDIATE MODEL FALLBACK)
    # 5. 503 Service Unavailable / High Demand
    if code == 503 or "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
        return (
            True,
            "⏳ Gemini is temporarily busy. SnapStudy is retrying automatically.",
            503,
        )

    # 6. 429 Rate Limit / Resource Exhausted
    if code == 429 or "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str or "quota" in err_str:
        return (
            True,
            "⚡ AI usage limit reached temporarily. Please try again shortly.",
            429,
        )

    # 7. 500, 502, 504 Gateway / Server Issues
    if code in {500, 502, 504} or any(k in err_str for k in ["500", "502", "504", "bad gateway", "gateway timeout"]):
        return (
            True,
            "Gemini service temporarily encountered an internal issue. Please try again shortly.",
            code or 500,
        )

    # 8. Network / Connection Timeouts
    if isinstance(e, (TimeoutError, ConnectionError)) or "timeout" in err_str or "connection" in err_str:
        return (
            True,
            "Connection to Gemini timed out. Please check your network connection and try again.",
            None,
        )

    # Default fallback for unknown exceptions
    return (
        False,
        "Something went wrong while analyzing the material. Please try again.",
        code,
    )


def call_gemini_with_fallback(
    client: genai.Client,
    contents: list,
    config: Optional[types.GenerateContentConfig] = None,
    models: Optional[list] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Executes a Gemini generate_content call with immediate model fallback on temporary 503/429/5xx errors.
    Fallback order:
      1. gemini-3.8-flash
      2. gemini-3.6-flash
      3. gemini-3.5-flash-lite
    Permanent errors (401, 403, 404, 400) immediately raise without fallback.
    Logs successful model to terminal.
    """
    models_to_try = models if models else GEMINI_MODELS
    last_error: Optional[Exception] = None
    last_status_code: Optional[int] = None

    for idx, model_name in enumerate(models_to_try):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            print(f"[SnapStudy AI] Successfully generated content using model: {model_name}")
            return response.text or "No response received from Gemini."
        except Exception as e:
            last_error = e
            is_retryable, friendly_msg, code = classify_gemini_error(e, model_name)
            last_status_code = code

            # Permanent client errors must NOT fallback
            if not is_retryable:
                print(f"[SnapStudy AI] Permanent error on model '{model_name}': {friendly_msg}")
                raise GeminiServiceError(friendly_msg, status_code=code, original_error=e)

            # Temporary error (503 / 429 / 5xx): try next fallback model immediately
            has_next = (idx + 1) < len(models_to_try)
            next_model = models_to_try[idx + 1] if has_next else None

            print(
                f"[SnapStudy AI] Model '{model_name}' returned temporary error ({code or 'UNAVAILABLE'}). "
                f"{'Switching immediately to ' + next_model if has_next else 'No more fallback models available.'}"
            )

            if has_next:
                if status_callback:
                    status_callback("Gemini is temporarily busy. Trying another available model...")
                # Immediate fallback without long sleeps (brief 0.2s pause for socket release)
                time.sleep(0.2)
            else:
                break

    # If all models in the fallback pipeline failed with temporary errors
    print("[SnapStudy AI] All configured Gemini models returned temporary busy/unavailable errors.")
    raise GeminiServiceError(
        "Gemini is temporarily busy. Please try again in a few moments.",
        status_code=last_status_code or 503,
        original_error=last_error,
    )


# Backward-compatible alias for any callers expecting call_gemini_with_retry
def call_gemini_with_retry(
    client: genai.Client,
    model_name: Optional[str] = None,
    contents: list = None,
    config: Optional[types.GenerateContentConfig] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    fallback_model: Optional[str] = None,
    models: Optional[list] = None,
    **kwargs,
) -> str:
    models_to_use = models
    if not models_to_use:
        if model_name:
            models_to_use = [model_name] + [m for m in GEMINI_MODELS if m != model_name]
        else:
            models_to_use = GEMINI_MODELS
    return call_gemini_with_fallback(
        client=client,
        contents=contents,
        config=config,
        models=models_to_use,
        status_callback=status_callback,
    )


def get_gemini_client(api_key: str) -> genai.Client:
    """Initializes the official google-genai Client."""
    return genai.Client(api_key=api_key)


def analyze_uploaded_material(
    client: genai.Client,
    image_bytes: bytes,
    mime_type: str,
    custom_question: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    models: Optional[list] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Sends the uploaded study image to Gemini Vision with structured prompts and immediate model fallback.
    """
    prompt = custom_question if custom_question else INITIAL_IMAGE_ANALYSIS_PROMPT

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    text_part = types.Part.from_text(text=prompt)

    config = types.GenerateContentConfig(
        system_instruction=STUDY_ASSISTANT_SYSTEM_PROMPT,
        temperature=0.4,
    )

    models_to_use = models
    if not models_to_use:
        if model_name:
            models_to_use = [model_name] + [m for m in GEMINI_MODELS if m != model_name]
        else:
            models_to_use = GEMINI_MODELS

    return call_gemini_with_fallback(
        client=client,
        contents=[image_part, text_part],
        config=config,
        models=models_to_use,
        status_callback=status_callback,
    )


def generate_chat_reply(
    client: genai.Client,
    messages: list,
    image_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    models: Optional[list] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Generates a conversational reply maintaining full study context and image reference with model fallback.
    """
    contents = []

    # First turn: attach the image if available so Gemini maintains vision context
    first_user_turn = True
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        parts = []

        if role == "user" and first_user_turn and image_bytes and mime_type:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            first_user_turn = False

        parts.append(types.Part.from_text(text=msg["content"]))
        contents.append(types.Content(role=role, parts=parts))

    config = types.GenerateContentConfig(
        system_instruction=STUDY_ASSISTANT_SYSTEM_PROMPT,
        temperature=0.5,
    )

    models_to_use = models
    if not models_to_use:
        if model_name:
            models_to_use = [model_name] + [m for m in GEMINI_MODELS if m != model_name]
        else:
            models_to_use = GEMINI_MODELS

    return call_gemini_with_fallback(
        client=client,
        contents=contents,
        config=config,
        models=models_to_use,
        status_callback=status_callback,
    )


def generate_session_summary(
    client: genai.Client,
    student_name: str,
    student_email: str,
    messages: list,
    status_callback: Optional[Callable[[str], None]] = None,
    models: Optional[list] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Synthesizes the entire tutoring session into high-yield revision notes with model fallback.
    """
    conversation_transcript = "\n\n".join(
        [f"**{m['role'].capitalize()}**: {m['content']}" for m in messages]
    )

    summary_instruction = SUMMARY_GENERATION_PROMPT.format(
        student_name=student_name,
        student_email=student_email,
    )

    prompt = (
        f"{summary_instruction}\n\n"
        f"--- TUTORING CONVERSATION TRANSCRIPT ---\n"
        f"{conversation_transcript}\n"
        f"--- END OF TRANSCRIPT ---"
    )

    config = types.GenerateContentConfig(
        system_instruction="You are an expert academic summarizer creating concise, high-impact revision notes.",
        temperature=0.3,
    )

    models_to_use = models
    if not models_to_use:
        if model_name:
            models_to_use = [model_name] + [m for m in GEMINI_MODELS if m != model_name]
        else:
            models_to_use = GEMINI_MODELS

    return call_gemini_with_fallback(
        client=client,
        contents=[types.Part.from_text(text=prompt)],
        config=config,
        models=models_to_use,
        status_callback=status_callback,
    )


# ==============================================================================
# 3. Email Dispatch via Gmail SMTP
# ==============================================================================

def send_study_summary_email(
    to_email: str,
    student_name: str,
    summary_markdown: str,
    smtp_email: str,
    smtp_password: str,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> Tuple[bool, str]:
    """
    Sends the generated study revision summary to the student's email using Gmail SMTP.
    """
    if not smtp_email or not smtp_password:
        return False, "SMTP credentials are missing. Please configure them in secrets.toml."

    # Convert basic markdown formatting to clean HTML paragraphs
    html_body_content = summary_markdown.replace("\n", "<br>")

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: #f8fafc;
                color: #1e293b;
                margin: 0;
                padding: 24px;
            }}
            .card {{
                max-width: 680px;
                margin: 0 auto;
                background: #ffffff;
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            }}
            .header {{
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: #ffffff;
                padding: 28px 32px;
            }}
            .header h1 {{
                margin: 0 0 6px 0;
                font-size: 24px;
                font-weight: 700;
            }}
            .header p {{
                margin: 0;
                font-size: 14px;
                opacity: 0.9;
            }}
            .content {{
                padding: 32px;
                line-height: 1.65;
                font-size: 15px;
            }}
            .footer {{
                background-color: #f1f5f9;
                padding: 16px 32px;
                text-align: center;
                font-size: 12px;
                color: #64748b;
                border-top: 1px solid #e2e8f0;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <h1>SnapStudy AI 📚 Study Summary</h1>
                <p>Personalized Revision Notes for <strong>{student_name}</strong></p>
            </div>
            <div class="content">
                <p>Hello <strong>{student_name}</strong>,</p>
                <p>Here is your concise study revision summary from your recent SnapStudy session:</p>
                <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
                <div>{html_body_content}</div>
            </div>
            <div class="footer">
                Generated automatically by SnapStudy AI &bull; Keep learning and aiming high!
            </div>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"SnapStudy AI 📚 Your Study Revision Notes ({student_name})"
    msg["From"] = f"SnapStudy AI <{smtp_email}>"
    msg["To"] = to_email

    # Plain-text and HTML versions
    part_text = MIMEText(summary_markdown, "plain")
    part_html = MIMEText(html_template, "html")

    msg.attach(part_text)
    msg.attach(part_html)

    try:
        server = smtplib.SMTP(smtp_server, int(smtp_port), timeout=20)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_email, smtp_password)
        server.send_message(msg)
        server.quit()
        return True, f"Study notes successfully sent to {to_email}!"
    except smtplib.SMTPAuthenticationError:
        return (
            False,
            "Authentication failed with Gmail SMTP. If using Gmail, make sure you use an 'App Password' "
            "rather than your primary Google account password (requires 2-Step Verification).",
        )
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"


# ==============================================================================
# 4. Presentation Helpers & UI Formatting
# ==============================================================================

def format_friendly_error(e: Exception, model_name: str) -> str:
    """Formats an exception into a user-friendly message without raw JSON or trace leaks."""
    if isinstance(e, GeminiServiceError):
        return e.user_message
    _, msg, _ = classify_gemini_error(e, model_name)
    return msg


def render_image(image_bytes: bytes, caption: Optional[str] = None):
    """Safely renders image across Streamlit versions with responsive width."""
    try:
        st.image(image_bytes, caption=caption, width="stretch")
    except TypeError:
        st.image(image_bytes, caption=caption, use_container_width=True)


def render_structured_analysis(content: str):
    """
    Renders the Gemini analysis with a distinct, readable visual hierarchy:
    1. 🔍 What I Found (Detected material / extracted text)
    2. 🧠 Concept Explanation (Core theory & principles)
    3. 📝 Step-by-Step Solution (Clearly separated steps)
    4. ✅ Final Answer (Highlight in an attractive Emerald success card)
    5. 🎯 Practice (Practice questions in distinct individual cards)
    """
    if not content or not content.strip():
        return

    # Parse sections based on markdown headings
    parts = re.split(r'\n(?=#{2,3}\s+)', content.strip())

    sections = {
        'found': [],
        'concept': [],
        'solution': [],
        'final_answer': [],
        'practice': []
    }

    # Check if content has structured headers
    has_standard_headers = any(
        any(k in part[:80].lower() for k in ['material', 'overview', 'transcription', 'extraction', 'concept', 'step-by-step', 'solution', 'practice', 'next step'])
        for part in parts
    )

    if not has_standard_headers or len(parts) <= 1:
        st.markdown(f'<div class="analysis-content">\n\n{content}\n\n</div>', unsafe_allow_html=True)
        return

    for part in parts:
        lines = part.strip().split('\n', 1)
        header = lines[0].lower() if lines else ''
        body = lines[1].strip() if len(lines) > 1 else ''

        if any(k in header for k in ['material overview', 'transcription', 'extraction', 'what i found', 'overview']):
            sections['found'].append(body or part)
        elif any(k in header for k in ['core concept', 'concept', 'explanation', 'theory']):
            sections['concept'].append(body or part)
        elif any(k in header for k in ['step-by-step', 'solution', 'detailed breakdown', 'solve']):
            # Check if final answer is embedded
            fa_match = re.search(r'(\*\*Final Answer:?\*\*|###\s*Final Answer:?|Final Answer:)([\s\S]+)$', body or part, re.IGNORECASE)
            if fa_match:
                sol_body = (body or part)[:fa_match.start()].strip()
                fa_body = fa_match.group(2).strip()
                sections['solution'].append(sol_body)
                sections['final_answer'].append(fa_body)
            else:
                sections['solution'].append(body or part)
        elif any(k in header for k in ['final answer', 'result', 'conclusion']):
            sections['final_answer'].append(body or part)
        elif any(k in header for k in ['next step', 'question', 'practice']):
            sections['practice'].append(body or part)
        else:
            if not sections['found']:
                sections['found'].append(part)
            else:
                sections['solution'].append(body or part)

    st.markdown('<div class="analysis-content">', unsafe_allow_html=True)

    # 1. 🔍 What I Found
    if sections['found']:
        st.markdown(
            """
            <div class="analysis-section-header">
                <span class="analysis-section-icon" style="color: #4F46E5;">🔍</span>
                <span class="analysis-section-title">What I Found</span>
                <span class="analysis-section-pill pill-indigo">Detected Material</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in sections['found']:
            st.markdown(item)
        st.markdown('<div class="analysis-divider"></div>', unsafe_allow_html=True)

    # 2. 🧠 Concept Explanation
    if sections['concept']:
        st.markdown(
            """
            <div class="analysis-section-header">
                <span class="analysis-section-icon" style="color: #7C3AED;">🧠</span>
                <span class="analysis-section-title">Concept Explanation</span>
                <span class="analysis-section-pill pill-violet">Core Theory</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in sections['concept']:
            st.markdown(item)
        st.markdown('<div class="analysis-divider"></div>', unsafe_allow_html=True)

    # 3. 📝 Step-by-Step Solution
    if sections['solution']:
        st.markdown(
            """
            <div class="analysis-section-header">
                <span class="analysis-section-icon" style="color: #0891B2;">📝</span>
                <span class="analysis-section-title">Step-by-Step Solution</span>
                <span class="analysis-section-pill pill-cyan">Walkthrough</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in sections['solution']:
            st.markdown(item)

    # 4. ✅ Final Answer (Highlight in an attractive Emerald success card)
    if sections['final_answer']:
        fa_text = "\n\n".join(sections['final_answer'])
        st.markdown(
            """
            <div class="final-answer-card">
                <div class="final-answer-header">
                    <span class="final-answer-icon">✅</span>
                    <span class="final-answer-title">Final Answer</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="final-answer-box">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(fa_text)
        st.markdown("</div>", unsafe_allow_html=True)

    # 5. 🎯 Practice (Practice questions in distinct individual cards)
    if sections['practice']:
        st.markdown(
            """
            <div class="analysis-section-header">
                <span class="analysis-section-icon" style="color: #EA580C;">🎯</span>
                <span class="analysis-section-title">Practice</span>
                <span class="analysis-section-pill pill-coral">Practice Questions</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        practice_combined = "\n\n".join(sections['practice'])
        q_items = re.split(r'\n(?=[0-9]+\.\s+)', practice_combined.strip())
        valid_questions = [q.strip() for q in q_items if q.strip()]

        if len(valid_questions) > 1:
            for idx, q in enumerate(valid_questions, 1):
                clean_q = re.sub(r'^[0-9]+\.\s*', '', q)
                st.markdown(
                    f"""
                    <div class="practice-card-box">
                        <span class="practice-q-badge">Question {idx}</span>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(clean_q)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="practice-card-box">
                """,
                unsafe_allow_html=True,
            )
            st.markdown(practice_combined)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# 5. Streamlit Application Interface
# ==============================================================================

def main():
    st.set_page_config(
        page_title="SnapStudy AI - Visual Study Assistant",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --------------------------------------------------------------------------
    # Distinctive Educational AI SaaS Color System:
    # Primary: #4F46E5 (Indigo) | Secondary: #7C3AED (Violet) | Accent: #06B6D4 (Cyan)
    # Action Highlight: #F97316 (Coral) | Success: #10B981 (Emerald)
    # Background: #F5F7FF | Sidebar: #EEF2FF | Card: #FFFFFF | Text: #172033 | Border: #DDE3F0
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            /* Global Font & Theme Baseline */
            html, body, [class*="css"], .stApp {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                color: #172033 !important;
                background-color: #F5F7FF !important;
                overflow-x: hidden !important;
            }

            /* Remove Streamlit top colored decoration line & transparent top bar */
            div[data-testid="stDecoration"] {
                display: none !important;
                height: 0 !important;
            }

            header[data-testid="stHeader"] {
                background: transparent !important;
                border-bottom: none !important;
                box-shadow: none !important;
                height: 2.2rem !important;
            }

            header[data-testid="stHeader"] [data-testid="stToolbar"] {
                right: 1.5rem !important;
                top: 0.5rem !important;
            }

            /* Container padding adjustments - Clean 1240px grid */
            .block-container {
                padding-top: 1.2rem !important;
                padding-bottom: 1.6rem !important;
                padding-left: 32px !important;
                padding-right: 32px !important;
                max-width: 1240px !important;
                width: 100% !important;
                margin: 0 auto !important;
                background-color: transparent !important;
                overflow-x: hidden !important;
                box-sizing: border-box !important;
            }

            /* Expander Polish */
            [data-testid="stExpander"] {
                background: #FFFFFF !important;
                border: 1px solid #DDE3F0 !important;
                border-radius: 12px !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
                margin-top: 12px !important;
            }

            [data-testid="stExpander"] summary {
                color: #172033 !important;
                font-weight: 600 !important;
            }

            [data-testid="stExpander"] summary:hover {
                color: #4F46E5 !important;
            }

            /* Alert Polish */
            div[data-testid="stAlert"] {
                border-radius: 10px !important;
                border: 1px solid #DDE3F0 !important;
            }

            /* Modern Compact Header */
            .snapstudy-header {
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
                background: #FFFFFF !important;
                border: 1px solid #DDE3F0 !important;
                border-radius: 14px !important;
                padding: 16px 24px !important;
                margin-bottom: 8px !important;
                box-shadow: 0 1px 4px rgba(79, 70, 229, 0.04) !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            .header-brand {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .header-brand-icon {
                font-size: 22px;
                background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
                border: 1px solid #C7D2FE;
                border-radius: 10px;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .header-brand-title {
                font-size: 17px;
                font-weight: 700;
                color: #4F46E5;
                line-height: 1.2;
            }

            .header-brand-subtitle {
                font-size: 12px;
                font-weight: 600;
                color: #7C3AED;
            }

            .header-status {
                display: flex;
                align-items: center;
                gap: 8px;
                background: #ECFDF5;
                border: 1px solid #A7F3D0;
                color: #065F46;
                padding: 5px 14px;
                border-radius: 20px;
                font-size: 12.5px;
                font-weight: 600;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #10B981;
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
                animation: status-pulse 2s infinite;
            }

            @keyframes status-pulse {
                0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
                100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }

            .header-accent-line {
                height: 3px;
                background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 45%, #06B6D4 100%);
                border-radius: 2px;
                margin-bottom: 22px;
            }

            /* Sidebar Styling: #EEF2FF background, white cards, indigo headings */
            [data-testid="stSidebar"] {
                background-color: #EEF2FF !important;
                border-right: 1px solid #DDE3F0 !important;
                min-width: 320px !important;
                max-width: 340px !important;
                width: 330px !important;
            }

            [data-testid="stSidebarUserContent"] {
                padding-top: 1.5rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-bottom: 2rem !important;
            }

            [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #FFFFFF !important;
                border: 1px solid #DDE3F0 !important;
                border-radius: 12px !important;
                box-shadow: 0 1px 3px rgba(79, 70, 229, 0.03) !important;
                margin-bottom: 16px !important;
            }

            [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 16px !important;
            }

            .sidebar-section-title {
                font-size: 13px !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                color: #4F46E5 !important;
                margin-bottom: 12px !important;
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
            }

            /* Remove 'Press Enter to submit form' text from inside the input field */
            [data-testid="InputInstructions"],
            .stInputInstructions,
            div[data-testid="InputInstructions"],
            [data-testid="stTextInput"] [data-testid="InputInstructions"] {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                height: 0 !important;
                width: 0 !important;
                overflow: hidden !important;
                position: absolute !important;
                pointer-events: none !important;
            }

            /* Clean, vertically centered styling for text inputs */
            div[data-testid="stTextInput"] label {
                font-size: 13.5px !important;
                font-weight: 600 !important;
                color: #172033 !important;
                margin-bottom: 5px !important;
            }

            div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
                background-color: #FFFFFF !important;
                border: 1px solid #DDE3F0 !important;
                border-radius: 8px !important;
                min-height: 46px !important;
                height: 46px !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
            }

            div[data-testid="stTextInput"] div[data-baseweb="input"] {
                background-color: #FFFFFF !important;
                border: 1px solid #DDE3F0 !important;
                border-radius: 8px !important;
                min-height: 46px !important;
                height: 46px !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
            }

            div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
            div[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {
                border-color: #4F46E5 !important;
                box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15) !important;
            }

            div[data-testid="stTextInput"] input {
                color: #172033 !important;
                font-size: 14px !important;
                font-family: inherit !important;
                padding-left: 14px !important;
                padding-right: 14px !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                min-height: 46px !important;
                height: 46px !important;
                line-height: normal !important;
                box-sizing: border-box !important;
                vertical-align: middle !important;
                background-color: transparent !important;
            }

            div[data-testid="stTextInput"] input::placeholder {
                color: #94A3B8 !important;
                font-size: 14px !important;
                line-height: normal !important;
                vertical-align: middle !important;
                opacity: 1 !important;
            }

            /* Native Streamlit border container card styling in Main Area */
            .block-container div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #FFFFFF !important;
                border-radius: 14px !important;
                border: 1px solid #DDE3F0 !important;
                box-shadow: 0 1px 4px rgba(79, 70, 229, 0.03) !important;
                margin-bottom: 24px !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            .block-container div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 24px 28px !important;
            }

            /* Headings */
            h1, h2, h3 {
                color: #172033 !important;
                letter-spacing: -0.015em !important;
            }
            h4, h5 {
                color: #4F46E5 !important;
                letter-spacing: -0.01em !important;
            }

            /* Hero Section */
            .hero-wrapper {
                background: #FFFFFF !important;
                border: 1px solid #DDE3F0 !important;
                border-radius: 14px !important;
                padding: 32px 36px !important;
                margin-bottom: 24px !important;
                box-shadow: 0 1px 4px rgba(79, 70, 229, 0.03) !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            .hero-heading {
                font-size: 26px !important;
                font-weight: 700 !important;
                color: #172033 !important;
                margin-top: 0 !important;
                margin-bottom: 8px !important;
                line-height: 1.25 !important;
                text-align: left !important;
            }

            .hero-subheading {
                font-size: 14.5px !important;
                color: #64748B !important;
                line-height: 1.5 !important;
                margin: 0 !important;
                text-align: left !important;
            }

            /* Upload Area Styling */
            [data-testid="stFileUploader"] {
                width: 100% !important;
            }

            [data-testid="stFileUploader"] section {
                background-color: #F8FAFF !important;
                border: 1.5px dashed #818CF8 !important;
                border-radius: 12px !important;
                padding: 22px 20px !important;
                min-height: 110px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                transition: all 0.2s ease !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            [data-testid="stFileUploader"] section:hover {
                border-color: #4F46E5 !important;
                background-color: #EEF2FF !important;
            }

            /* Study Workspace Cards */
            .workspace-card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 14px;
                padding-bottom: 8px;
                border-bottom: 1px solid #DDE3F0;
            }

            .workspace-card-title {
                font-size: 15.5px;
                font-weight: 700;
                color: #4F46E5;
                display: flex;
                align-items: center;
                gap: 8px;
                margin: 0;
            }

            /* Stepper Card for Multi-Step Loading */
            .stepper-card {
                background: #FFFFFF !important;
                border: 1px solid #C7D2FE !important;
                border-radius: 14px !important;
                padding: 20px 24px !important;
                margin-bottom: 20px !important;
                box-shadow: 0 4px 16px rgba(79, 70, 229, 0.08) !important;
            }

            .stepper-header {
                font-size: 16px !important;
                font-weight: 700 !important;
                color: #4F46E5 !important;
                margin-bottom: 14px !important;
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
            }

            .stepper-row {
                display: flex !important;
                align-items: center !important;
                gap: 10px !important;
                font-size: 14px !important;
                margin-bottom: 8px !important;
                color: #172033 !important;
            }

            .step-done {
                color: #10B981 !important;
                font-weight: 600 !important;
            }

            .step-active {
                color: #F97316 !important;
                font-weight: 700 !important;
            }

            .step-wait {
                color: #64748B !important;
            }

            /* Primary Action Button: #4F46E5 Indigo, Hover: #7C3AED Violet */
            button[kind="primary"] {
                background: #4F46E5 !important;
                background-color: #4F46E5 !important;
                border: none !important;
                color: #FFFFFF !important;
                font-weight: 600 !important;
                border-radius: 10px !important;
                padding: 8px 18px !important;
                box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25) !important;
                transition: all 0.15s ease !important;
            }

            button[kind="primary"]:hover {
                background: #7C3AED !important;
                background-color: #7C3AED !important;
                box-shadow: 0 4px 12px rgba(124, 58, 237, 0.35) !important;
                transform: translateY(-1px) !important;
            }

            /* Dedicated Action Highlight: Coral #F97316 for Analyze Material */
            .analyze-coral-btn button {
                background: #F97316 !important;
                background-color: #F97316 !important;
                border: none !important;
                color: #FFFFFF !important;
                font-weight: 700 !important;
                font-size: 15px !important;
                border-radius: 10px !important;
                padding: 10px 22px !important;
                box-shadow: 0 3px 10px rgba(249, 115, 22, 0.35) !important;
                transition: all 0.15s ease !important;
            }

            .analyze-coral-btn button:hover {
                background: #EA580C !important;
                background-color: #EA580C !important;
                box-shadow: 0 6px 16px rgba(249, 115, 22, 0.45) !important;
                transform: translateY(-1px) !important;
            }

            /* Secondary Button Polish */
            button[kind="secondary"] {
                border: 1px solid #DDE3F0 !important;
                border-radius: 10px !important;
                font-weight: 500 !important;
                background: #FFFFFF !important;
                color: #172033 !important;
                transition: all 0.15s ease !important;
            }

            button[kind="secondary"]:hover {
                border-color: #C7D2FE !important;
                background: #EEF2FF !important;
                color: #4F46E5 !important;
            }

            /* Chat Messages: User in light indigo, AI in white with violet left border */
            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
                background-color: #EEF2FF !important;
                border: 1px solid #C7D2FE !important;
                border-radius: 14px !important;
                color: #172033 !important;
                padding: 12px 16px !important;
            }

            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
                background-color: #FFFFFF !important;
                border: 1px solid #DDE3F0 !important;
                border-left: 4px solid #7C3AED !important;
                border-radius: 14px !important;
                color: #172033 !important;
                box-shadow: 0 2px 6px rgba(124, 58, 237, 0.05) !important;
                padding: 14px 18px !important;
            }

            /* ==============================================================
               PREMIUM AI CHAT COMPOSER (NORMAL FLOW & FULL WIDTH)
               ============================================================== */
            /* Put stBottom / stBottomBlockContainer into normal document flow to eliminate overlap */
            div[data-testid="stBottom"],
            div[data-testid="stBottomBlockContainer"] {
                position: relative !important;
                bottom: auto !important;
                left: auto !important;
                right: auto !important;
                width: 100% !important;
                max-width: 1240px !important;
                margin: 16px auto 36px auto !important;
                padding: 0 32px !important;
                background: transparent !important;
                box-shadow: none !important;
                z-index: 10 !important;
                box-sizing: border-box !important;
            }

            div[data-testid="stBottomBlockContainer"] > div {
                max-width: 100% !important;
                width: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            /* Chat Composer Container */
            div[data-testid="stChatInput"] {
                width: 100% !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                background: #FFFFFF !important;
                border: 1px solid #DDE3F0 !important;
                border-radius: 16px !important;
                padding: 6px 10px 6px 18px !important;
                box-shadow: 0 2px 12px rgba(79, 70, 229, 0.06), 0 1px 3px rgba(0, 0, 0, 0.02) !important;
                transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
                min-height: 58px !important;
            }

            div[data-testid="stChatInput"]:focus-within {
                border-color: #4F46E5 !important;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15), 0 4px 16px rgba(79, 70, 229, 0.1) !important;
            }

            /* Make inner wrappers stretch full width */
            div[data-testid="stChatInput"] > div,
            div[data-testid="stChatInput"] [data-baseweb="base-input"],
            div[data-testid="stChatInput"] [data-baseweb="textarea"] {
                flex: 1 1 auto !important;
                width: 100% !important;
                min-width: 0 !important;
                background: transparent !important;
                border: none !important;
                padding: 0 !important;
                margin: 0 !important;
                box-shadow: none !important;
            }

            /* Full-width, comfortably sized textarea */
            div[data-testid="stChatInput"] textarea {
                flex: 1 1 auto !important;
                width: 100% !important;
                min-width: 0 !important;
                background: transparent !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
                font-family: inherit !important;
                font-size: 15px !important;
                line-height: 1.45 !important;
                color: #172033 !important;
                padding: 8px 4px !important;
                min-height: 44px !important;
                height: 44px !important;
                max-height: 120px !important;
                resize: none !important;
                box-sizing: border-box !important;
            }

            div[data-testid="stChatInput"] textarea:focus {
                outline: none !important;
                box-shadow: none !important;
            }

            div[data-testid="stChatInput"] textarea::placeholder {
                color: #64748B !important;
                font-size: 15px !important;
                opacity: 1 !important;
            }

            /* Ensure no helper / instructions text ever overlaps inside composer */
            div[data-testid="stChatInput"] [data-testid="InputInstructions"] {
                display: none !important;
            }

            /* Send Button */
            div[data-testid="stChatInput"] button {
                flex: 0 0 46px !important;
                width: 46px !important;
                height: 46px !important;
                min-width: 46px !important;
                min-height: 46px !important;
                border-radius: 12px !important;
                background: #4F46E5 !important;
                background-color: #4F46E5 !important;
                border: none !important;
                margin-left: 12px !important;
                margin-right: 0 !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                padding: 0 !important;
                box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25) !important;
                cursor: pointer !important;
                transition: all 0.15s ease !important;
            }

            div[data-testid="stChatInput"] button:hover {
                background: #7C3AED !important;
                background-color: #7C3AED !important;
                box-shadow: 0 4px 12px rgba(124, 58, 237, 0.35) !important;
                transform: translateY(-1px) !important;
            }

            div[data-testid="stChatInput"] button:disabled {
                background: #CBD5E1 !important;
                background-color: #CBD5E1 !important;
                box-shadow: none !important;
                cursor: not-allowed !important;
                transform: none !important;
            }

            /* Send Icon: White Arrow */
            div[data-testid="stChatInput"] button svg {
                fill: #FFFFFF !important;
                color: #FFFFFF !important;
                stroke: #FFFFFF !important;
                width: 18px !important;
                height: 18px !important;
            }

            /* Composer Header / Context Section */
            .composer-meta-wrapper {
                width: 100% !important;
                max-width: 100% !important;
                margin: 24px 0 10px 0 !important;
                padding: 0 2px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                flex-wrap: wrap !important;
                gap: 10px !important;
                box-sizing: border-box !important;
            }

            .composer-title-group {
                display: flex !important;
                flex-direction: column !important;
            }

            .composer-title {
                font-size: 15.5px !important;
                font-weight: 700 !important;
                color: #172033 !important;
                letter-spacing: -0.01em !important;
            }

            .composer-subtitle {
                font-size: 12.5px !important;
                color: #64748B !important;
                margin-top: 2px !important;
            }

            .composer-attached-pill {
                display: inline-flex !important;
                align-items: center !important;
                gap: 6px !important;
                background: #ECFEFF !important;
                border: 1px solid #A5F3FC !important;
                color: #0891B2 !important;
                padding: 4px 12px !important;
                border-radius: 16px !important;
                font-size: 12px !important;
                font-weight: 600 !important;
            }

            .composer-empty-pill {
                display: inline-flex !important;
                align-items: center !important;
                gap: 6px !important;
                background: #EEF2FF !important;
                border: 1px solid #C7D2FE !important;
                color: #4F46E5 !important;
                padding: 4px 12px !important;
                border-radius: 16px !important;
                font-size: 12px !important;
                font-weight: 500 !important;
            }

            /* Suggestion Chips Wrapper */
            .composer-chips-row {
                width: 100% !important;
                max-width: 100% !important;
                margin: 0 0 14px 0 !important;
            }

            .composer-chip-btn button {
                background: #EEF2FF !important;
                background-color: #EEF2FF !important;
                border: 1px solid #C7D2FE !important;
                color: #4F46E5 !important;
                font-size: 12.5px !important;
                font-weight: 600 !important;
                border-radius: 20px !important;
                padding: 6px 14px !important;
                box-shadow: none !important;
                transition: all 0.15s ease !important;
                white-space: nowrap !important;
            }

            .composer-chip-btn button:hover {
                background: #E0E7FF !important;
                background-color: #E0E7FF !important;
                border-color: #818CF8 !important;
                color: #3730A3 !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 2px 6px rgba(79, 70, 229, 0.12) !important;
            }

            /* Follow-up Prompt Pills: Violet Accent */
            button[key^="pill_"] {
                background-color: #F5F3FF !important;
                border: 1px solid #DDD6FE !important;
                color: #7C3AED !important;
                font-size: 13px !important;
                font-weight: 600 !important;
                border-radius: 20px !important;
                transition: all 0.15s ease !important;
            }

            button[key^="pill_"]:hover {
                background-color: #EDE9FE !important;
                border-color: #7C3AED !important;
                color: #6D28D9 !important;
                transform: translateY(-1px) !important;
            }

            /* ==============================================================
               ANALYSIS RESULT SECTION - MODERN EDUCATIONAL HIERARCHY
               ============================================================== */
            .analysis-section-header {
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
                margin-top: 16px !important;
                margin-bottom: 8px !important;
            }

            .analysis-section-icon {
                font-size: 18px !important;
                line-height: 1 !important;
            }

            .analysis-section-title {
                font-size: 15.5px !important;
                font-weight: 700 !important;
                color: #172033 !important;
                letter-spacing: -0.01em !important;
            }

            .analysis-section-pill {
                font-size: 11px !important;
                font-weight: 700 !important;
                padding: 2px 8px !important;
                border-radius: 6px !important;
                letter-spacing: 0.04em !important;
                text-transform: uppercase !important;
            }

            .pill-indigo {
                background: #EEF2FF !important;
                color: #4F46E5 !important;
                border: 1px solid #C7D2FE !important;
            }

            .pill-violet {
                background: #F5F3FF !important;
                color: #7C3AED !important;
                border: 1px solid #DDD6FE !important;
            }

            .pill-cyan {
                background: #ECFEFF !important;
                color: #0891B2 !important;
                border: 1px solid #A5F3FC !important;
            }

            .pill-coral {
                background: #FFF7ED !important;
                color: #EA580C !important;
                border: 1px solid #FED7AA !important;
            }

            .analysis-divider {
                border-top: 1px solid #EEF2FF !important;
                margin: 16px 0 !important;
            }

            /* Success Card for Final Answer */
            .final-answer-card {
                background: #ECFDF5 !important;
                border: 1.5px solid #10B981 !important;
                border-radius: 12px !important;
                padding: 14px 18px !important;
                margin: 16px 0 10px 0 !important;
                box-shadow: 0 2px 8px rgba(16, 185, 129, 0.08) !important;
            }

            .final-answer-header {
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
                margin-bottom: 6px !important;
            }

            .final-answer-icon {
                font-size: 17px !important;
            }

            .final-answer-title {
                font-size: 13.5px !important;
                font-weight: 700 !important;
                color: #065F46 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.04em !important;
            }

            .final-answer-box {
                background: #ECFDF5 !important;
                border-left: 3px solid #10B981 !important;
                border-radius: 0 8px 8px 0 !important;
                padding: 10px 16px !important;
                margin-bottom: 14px !important;
            }

            /* Practice Questions in Individual Cards */
            .practice-card-box {
                background: #FFFFFF !important;
                border: 1px solid #DDE3F0 !important;
                border-left: 4px solid #F97316 !important;
                border-radius: 10px !important;
                padding: 12px 16px !important;
                margin-bottom: 10px !important;
                box-shadow: 0 1px 3px rgba(249, 115, 22, 0.04) !important;
                transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
            }

            .practice-card-box:hover {
                border-color: #F97316 !important;
                box-shadow: 0 3px 8px rgba(249, 115, 22, 0.1) !important;
            }

            .practice-q-badge {
                display: inline-block !important;
                background: #FFF7ED !important;
                border: 1px solid #FED7AA !important;
                color: #EA580C !important;
                font-size: 11px !important;
                font-weight: 700 !important;
                padding: 2px 8px !important;
                border-radius: 6px !important;
                margin-bottom: 6px !important;
                letter-spacing: 0.03em !important;
            }

            /* Enhanced Typography for Explanations, Math & Code */
            .analysis-content {
                line-height: 1.68 !important;
                color: #172033 !important;
                font-size: 14.5px !important;
            }

            .analysis-content p {
                color: #172033 !important;
                margin-bottom: 10px !important;
            }

            .analysis-content ul, .analysis-content ol {
                padding-left: 20px !important;
                margin-bottom: 12px !important;
            }

            .analysis-content li {
                margin-bottom: 6px !important;
                line-height: 1.6 !important;
                color: #172033 !important;
            }

            .analysis-content strong {
                color: #172033 !important;
                font-weight: 700 !important;
            }

            /* LaTeX KaTeX displays */
            .katex-display {
                background: #F8FAFF !important;
                padding: 10px 14px !important;
                border-radius: 8px !important;
                border: 1px solid #EEF2FF !important;
                margin: 12px 0 !important;
                overflow-x: auto !important;
            }

            .katex {
                font-size: 1.05em !important;
                color: #172033 !important;
            }

            /* Code Blocks */
            .analysis-content pre {
                background: #F8FAFF !important;
                border: 1px solid #DDE3F0 !important;
                border-left: 4px solid #06B6D4 !important;
                border-radius: 8px !important;
                padding: 12px 16px !important;
                margin: 12px 0 !important;
                overflow-x: auto !important;
            }

            .analysis-content code:not(pre code) {
                background: #EEF2FF !important;
                color: #4F46E5 !important;
                padding: 2px 6px !important;
                border-radius: 6px !important;
                font-size: 0.9em !important;
                border: 1px solid #C7D2FE !important;
            }

            .analysis-content blockquote {
                border-left: 4px solid #06B6D4 !important;
                background: #ECFEFF !important;
                padding: 10px 16px !important;
                border-radius: 0 8px 8px 0 !important;
                color: #172033 !important;
                margin: 12px 0 !important;
            }

            /* ==============================================================
               RESPONSIVE ADJUSTMENTS (TABLET & MOBILE)
               ============================================================== */
            @media (min-width: 769px) and (max-width: 1024px) {
                .block-container {
                    padding-left: 20px !important;
                    padding-right: 20px !important;
                }
                div[data-testid="stBottom"],
                div[data-testid="stBottomBlockContainer"] {
                    padding: 0 20px !important;
                }
                [data-testid="stSidebar"] {
                    min-width: 290px !important;
                    max-width: 300px !important;
                    width: 295px !important;
                }
            }

            @media (max-width: 768px) {
                [data-testid="stSidebar"] {
                    min-width: 100vw !important;
                    max-width: 100vw !important;
                    width: 100vw !important;
                }
                .block-container {
                    padding-left: 16px !important;
                    padding-right: 16px !important;
                    padding-top: 1rem !important;
                }
                div[data-testid="stBottom"],
                div[data-testid="stBottomBlockContainer"] {
                    padding: 0 16px !important;
                    margin: 16px auto 24px auto !important;
                }
                .snapstudy-header {
                    flex-direction: column !important;
                    align-items: flex-start !important;
                    gap: 12px !important;
                    padding: 16px !important;
                }
                .hero-wrapper {
                    padding: 20px 18px !important;
                }
                .hero-heading {
                    font-size: 20px !important;
                }
                .block-container div[data-testid="stVerticalBlockBorderWrapper"] > div {
                    padding: 18px 16px !important;
                }
                div[data-testid="stChatInput"] {
                    border-radius: 14px !important;
                    padding: 4px 6px 4px 12px !important;
                    min-height: 52px !important;
                }
                div[data-testid="stChatInput"] button {
                    flex: 0 0 40px !important;
                    width: 40px !important;
                    height: 40px !important;
                    min-width: 40px !important;
                    min-height: 40px !important;
                    margin-left: 8px !important;
                }
                .composer-meta-wrapper {
                    flex-direction: column !important;
                    align-items: flex-start !important;
                    gap: 6px !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # Configuration & Credentials
    # --------------------------------------------------------------------------
    secret_gemini_key = get_secret("GEMINI_API_KEY", "")
    active_models = get_model_pipeline()
    primary_model = active_models[0]
    smtp_email = get_secret("SMTP_EMAIL", "")
    smtp_password = get_secret("SMTP_APP_PASSWORD", "")
    smtp_server = get_secret("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(get_secret("SMTP_PORT", "587"))

    # --------------------------------------------------------------------------
    # Session State Initialization
    # --------------------------------------------------------------------------
    if "student_name" not in st.session_state:
        st.session_state.student_name = ""
    if "student_email" not in st.session_state:
        st.session_state.student_email = ""
    if "onboarding_complete" not in st.session_state:
        st.session_state.onboarding_complete = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None
    if "uploaded_image_mime" not in st.session_state:
        st.session_state.uploaded_image_mime = None
    if "uploaded_image_name" not in st.session_state:
        st.session_state.uploaded_image_name = None
    if "last_summary" not in st.session_state:
        st.session_state.last_summary = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # --------------------------------------------------------------------------
    # Sidebar: Cards & Controls
    # --------------------------------------------------------------------------
    with st.sidebar:
        # Sidebar Brand
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px; padding: 2px 0;">
                <div style="font-size: 22px; background: #FFFFFF; border: 1px solid #C7D2FE; border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center;">💡</div>
                <div>
                    <div style="font-size: 16px; font-weight: 700; color: #4F46E5; line-height: 1.2;">SnapStudy AI</div>
                    <div style="font-size: 11px; color: #7C3AED; font-weight: 600;">Study Dashboard</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Temporary API Key override if not configured in secrets
        api_key = secret_gemini_key
        if not api_key:
            with st.container(border=True):
                st.markdown('<div class="sidebar-section-title">🔑 API Key</div>', unsafe_allow_html=True)
                st.caption("Provide key or set GEMINI_API_KEY in secrets.toml")
                api_key = st.text_input(
                    "Gemini API Key:",
                    type="password",
                    label_visibility="collapsed",
                    placeholder="Enter Gemini API key...",
                    key="manual_api_key_input",
                )

        # 1. PROFILE CARD
        with st.container(border=True):
            st.markdown('<div class="sidebar-section-title">👤 Student Profile</div>', unsafe_allow_html=True)
            if st.session_state.onboarding_complete:
                st.markdown(f"**Name:** <span style='color: #172033; font-weight: 600;'>{st.session_state.student_name}</span>", unsafe_allow_html=True)
                st.markdown(f"**Email:** <code style='color: #4F46E5; background: #EEF2FF; border: 1px solid #C7D2FE;'>{st.session_state.student_email}</code>", unsafe_allow_html=True)
                if st.button("✏️ Edit Profile", use_container_width=True, key="btn_edit_profile"):
                    st.session_state.onboarding_complete = False
                    st.rerun()
            else:
                with st.form("sidebar_onboarding_form"):
                    name_val = st.text_input("Full Name", value=st.session_state.student_name, placeholder="e.g. Alex Johnson")
                    email_val = st.text_input("Email Address", value=st.session_state.student_email, placeholder="e.g. alex@university.edu")
                    if st.form_submit_button("Save Profile", use_container_width=True):
                        if not name_val.strip():
                            st.error("Please provide your name.")
                        elif not is_valid_email(email_val):
                            st.error("Please provide a valid email address.")
                        else:
                            st.session_state.student_name = name_val.strip()
                            st.session_state.student_email = email_val.strip()
                            st.session_state.onboarding_complete = True
                            st.success("Profile saved!")
                            st.rerun()

        # 2. STUDY MATERIAL CARD
        with st.container(border=True):
            st.markdown('<div class="sidebar-section-title">📚 Study Material</div>', unsafe_allow_html=True)
            sidebar_file = st.file_uploader(
                "Upload study material",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                key="sidebar_file_uploader",
                help="Supported: PNG, JPG, JPEG, WEBP",
            )
            if sidebar_file is not None:
                if st.session_state.uploaded_image_name != sidebar_file.name:
                    st.session_state.uploaded_image_bytes = sidebar_file.getvalue()
                    st.session_state.uploaded_image_mime = sidebar_file.type
                    st.session_state.uploaded_image_name = sidebar_file.name
                    st.session_state.messages = []
                    st.session_state.last_summary = None
                    st.rerun()

            if st.session_state.uploaded_image_name:
                st.caption(f"📄 **{st.session_state.uploaded_image_name}**")
                render_image(st.session_state.uploaded_image_bytes)
            else:
                st.caption("No material uploaded yet.")

        # 3. ACTIONS CARD
        with st.container(border=True):
            st.markdown('<div class="sidebar-section-title">⚙️ Actions</div>', unsafe_allow_html=True)

            can_send_summary = (
                st.session_state.onboarding_complete
                and len(st.session_state.messages) > 0
                and bool(api_key)
            )

            if st.button("📧 Send Summary to Email", disabled=not can_send_summary, use_container_width=True, key="btn_send_summary"):
                if not smtp_email or not smtp_password:
                    st.error("SMTP credentials missing. Please configure SMTP_EMAIL and SMTP_APP_PASSWORD in secrets.toml.")
                else:
                    summary_status = st.empty()
                    summary = None
                    with st.spinner("Gemini is creating your concise revision summary..."):
                        try:
                            client = get_gemini_client(api_key=api_key)
                            summary = generate_session_summary(
                                client=client,
                                student_name=st.session_state.student_name,
                                student_email=st.session_state.student_email,
                                messages=st.session_state.messages,
                                models=active_models,
                                status_callback=lambda msg: summary_status.info(f"⏳ {msg}"),
                            )
                            summary_status.empty()
                            st.session_state.last_summary = summary
                        except Exception as e:
                            summary_status.empty()
                            st.error(format_friendly_error(e, primary_model))

                    if summary:
                        with st.spinner(f"Sending study summary to {st.session_state.student_email}..."):
                            success, msg = send_study_summary_email(
                                to_email=st.session_state.student_email,
                                student_name=st.session_state.student_name,
                                summary_markdown=summary,
                                smtp_email=smtp_email,
                                smtp_password=smtp_password,
                                smtp_server=smtp_server,
                                smtp_port=smtp_port,
                            )
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)

            if st.button("🔄 Clear Chat & Reset", use_container_width=True, key="btn_clear_reset"):
                st.session_state.messages = []
                st.session_state.uploaded_image_bytes = None
                st.session_state.uploaded_image_mime = None
                st.session_state.uploaded_image_name = None
                st.session_state.last_summary = None
                st.rerun()

        # 4. MODEL CARD
        with st.container(border=True):
            st.markdown('<div class="sidebar-section-title">🤖 AI Model</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="font-size: 13px; color: #172033; margin-bottom: 4px;">
                    <strong>Active Model:</strong> <code style="color: #4F46E5; background: #EEF2FF; border: 1px solid #C7D2FE;">{primary_model}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if len(active_models) > 1:
                st.markdown(
                    f"""
                    <div style="font-size: 11.5px; color: #64748B;">
                        <strong>Fallbacks:</strong> {', '.join(f'<code style=\"color: #7C3AED; background: #F5F3FF;\">{m}</code>' for m in active_models[1:])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------------------------
    # Main Top Header
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="snapstudy-header">
            <div class="header-brand">
                <div class="header-brand-icon">💡</div>
                <div>
                    <div class="header-brand-title">SnapStudy AI</div>
                    <div class="header-brand-subtitle">Visual Study Assistant</div>
                </div>
            </div>
            <div class="header-status">
                <span class="status-dot"></span>
                <span><strong>Gemini Vision</strong> Online</span>
            </div>
        </div>
        <div class="header-accent-line"></div>
        """,
        unsafe_allow_html=True,
    )

    # API Key check banner
    if not api_key:
        st.error(
            "🔐 **AI Configuration Required**: Please configure your `GEMINI_API_KEY` in `.streamlit/secrets.toml` "
            "or enter it in the sidebar to begin."
        )

    # --------------------------------------------------------------------------
    # Main Content Branching
    # --------------------------------------------------------------------------
    has_image = bool(st.session_state.uploaded_image_bytes)
    has_messages = len(st.session_state.messages) > 0
    student_display = st.session_state.student_name if st.session_state.student_name else "Student"

    # ==========================================================================
    # BRANCH 1: LANDING SCREEN (No image uploaded and no active chat)
    # ==========================================================================
    if not has_image and not has_messages:
        # Hero Section
        st.markdown(
            f"""
            <div class="hero-wrapper">
                <h1 class="hero-heading">Welcome back, {student_display} 👋</h1>
                <p class="hero-subheading">Turn your study material into clear explanations, step-by-step solutions, and practice questions with AI.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Large Upload Card
        with st.container(border=True):
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 12px;">
                    <div style="font-size: 28px; background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 12px; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center;">📷</div>
                    <div>
                        <h3 style="font-size: 17.5px; font-weight: 700; color: #172033; margin: 0 0 3px 0;">Upload your study material</h3>
                        <p style="font-size: 13.5px; color: #64748B; margin: 0;">Upload an exam question, handwritten note, textbook page, code screenshot, mathematical problem, or technical diagram.</p>
                    </div>
                </div>
                <div style="font-size: 12px; font-weight: 600; color: #64748B; margin-bottom: 12px;">
                    Supported formats: <span style="background: #EEF2FF; color: #4F46E5; padding: 2px 8px; border-radius: 6px; font-weight: 700; border: 1px solid #C7D2FE;">PNG • JPG • WEBP</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            main_file = st.file_uploader(
                "Upload study material",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                key="main_landing_uploader",
                help="Drag and drop or browse files (PNG, JPG, WEBP)",
            )
            if main_file is not None:
                st.session_state.uploaded_image_bytes = main_file.getvalue()
                st.session_state.uploaded_image_mime = main_file.type
                st.session_state.uploaded_image_name = main_file.name
                st.session_state.messages = []
                st.session_state.last_summary = None
                st.rerun()

        # Feature Cards: Distinctly Tinted Educational Cards
        st.markdown(
            """
            <div style="font-size: 18px; font-weight: 700; color: #172033; margin-top: 14px; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">
                What can SnapStudy do?
            </div>
            """,
            unsafe_allow_html=True,
        )
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            st.markdown(
                """
                <div style="background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 14px; padding: 20px 18px; min-height: 175px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 1px 3px rgba(79, 70, 229, 0.05);">
                    <div>
                        <div style="font-size: 24px; background: #E0E7FF; border: 1px solid #C7D2FE; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">🧮</div>
                        <div style="font-size: 15px; font-weight: 700; color: #4F46E5; margin-bottom: 6px;">Solve Problems</div>
                        <div style="font-size: 13px; color: #64748B; line-height: 1.5;">Step-by-step solutions for mathematical, physics, and aptitude questions.</div>
                    </div>
                    <div style="margin-top: 14px;">
                        <span style="font-size: 11px; font-weight: 700; background: #E0E7FF; color: #4F46E5; padding: 3px 8px; border-radius: 6px; letter-spacing: 0.03em;">MATH & STEM</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with fc2:
            st.markdown(
                """
                <div style="background: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 14px; padding: 20px 18px; min-height: 175px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 1px 3px rgba(124, 58, 237, 0.05);">
                    <div>
                        <div style="font-size: 24px; background: #EDE9FE; border: 1px solid #DDD6FE; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">📝</div>
                        <div style="font-size: 15px; font-weight: 700; color: #7C3AED; margin-bottom: 6px;">Read Notes</div>
                        <div style="font-size: 13px; color: #64748B; line-height: 1.5;">Understand handwritten scribbles, blackboard photos, and textbook notes.</div>
                    </div>
                    <div style="margin-top: 14px;">
                        <span style="font-size: 11px; font-weight: 700; background: #EDE9FE; color: #7C3AED; padding: 3px 8px; border-radius: 6px; letter-spacing: 0.03em;">HANDWRITING</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with fc3:
            st.markdown(
                """
                <div style="background: #ECFEFF; border: 1px solid #A5F3FC; border-radius: 14px; padding: 20px 18px; min-height: 175px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 1px 3px rgba(6, 182, 212, 0.05);">
                    <div>
                        <div style="font-size: 24px; background: #CFFAFE; border: 1px solid #A5F3FC; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">💻</div>
                        <div style="font-size: 15px; font-weight: 700; color: #0891B2; margin-bottom: 6px;">Explain Code</div>
                        <div style="font-size: 13px; color: #64748B; line-height: 1.5;">Analyze programming code snippets, syntax errors, and algorithm logic.</div>
                    </div>
                    <div style="margin-top: 14px;">
                        <span style="font-size: 11px; font-weight: 700; background: #CFFAFE; color: #0891B2; padding: 3px 8px; border-radius: 6px; letter-spacing: 0.03em;">CODE & LOGIC</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with fc4:
            st.markdown(
                """
                <div style="background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 14px; padding: 20px 18px; min-height: 175px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 1px 3px rgba(249, 115, 22, 0.05);">
                    <div>
                        <div style="font-size: 24px; background: #FFEDD5; border: 1px solid #FED7AA; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">📐</div>
                        <div style="font-size: 15px; font-weight: 700; color: #EA580C; margin-bottom: 6px;">Understand Diagrams</div>
                        <div style="font-size: 13px; color: #64748B; line-height: 1.5;">Explain technical architectures, scientific diagrams, and flowcharts.</div>
                    </div>
                    <div style="margin-top: 14px;">
                        <span style="font-size: 11px; font-weight: 700; background: #FFEDD5; color: #EA580C; padding: 3px 8px; border-radius: 6px; letter-spacing: 0.03em;">DIAGRAMS</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==========================================================================
    # BRANCH 2: STAGING SCREEN (Material uploaded, waiting for "Analyze Material")
    # ==========================================================================
    elif has_image and not has_messages:
        col_preview, col_action = st.columns([1, 1], gap="large")

        with col_preview:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="workspace-card-header">
                        <h4 class="workspace-card-title">📷 Uploaded Study Material</h4>
                        <span style="font-size: 12px; background: #EEF2FF; color: #4F46E5; padding: 3px 10px; border-radius: 12px; font-weight: 600; border: 1px solid #C7D2FE;">{st.session_state.uploaded_image_name}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_image(st.session_state.uploaded_image_bytes)

        with col_action:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="workspace-card-header">
                        <h4 class="workspace-card-title">✨ Ready for AI Analysis</h4>
                    </div>
                    <p style="font-size: 14.5px; color: #475569; line-height: 1.6; margin-bottom: 22px;">
                        Your study material has been loaded. SnapStudy AI will inspect the contents, 
                        transcribe handwritten notes or formulas, provide step-by-step solutions, and prepare practice questions.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

                status_placeholder = st.empty()

                st.markdown('<div class="analyze-coral-btn">', unsafe_allow_html=True)
                analyze_clicked = st.button("✨ Analyze Material", type="primary", use_container_width=True, key="btn_analyze_staging")
                st.markdown('</div>', unsafe_allow_html=True)

                if analyze_clicked:
                    if not api_key:
                        st.error("Please enter a valid Gemini API key first.")
                    else:
                        # Multi-Step Loading State Card
                        status_placeholder.markdown(
                            """
                            <div class="stepper-card">
                                <div class="stepper-header">✨ Understanding your material</div>
                                <div class="stepper-row step-done"><span>✓</span> Reading uploaded image</div>
                                <div class="stepper-row step-done"><span>✓</span> Identifying questions and concepts</div>
                                <div class="stepper-row step-active"><span>⏳</span> Generating explanation...</div>
                                <div class="stepper-row step-wait"><span>○</span> Preparing practice questions</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        def on_fallback_status(msg: str):
                            status_placeholder.markdown(
                                """
                                <div class="stepper-card">
                                    <div class="stepper-header">✨ Understanding your material</div>
                                    <div class="stepper-row step-done"><span>✓</span> Reading uploaded image</div>
                                    <div class="stepper-row step-done"><span>✓</span> Identifying questions and concepts</div>
                                    <div class="stepper-row step-active"><span>⏳</span> Gemini is temporarily busy. Retrying automatically...</div>
                                    <div class="stepper-row step-wait"><span>○</span> Preparing practice questions</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        try:
                            client = get_gemini_client(api_key=api_key)
                            analysis = analyze_uploaded_material(
                                client=client,
                                image_bytes=st.session_state.uploaded_image_bytes,
                                mime_type=st.session_state.uploaded_image_mime,
                                models=active_models,
                                status_callback=on_fallback_status,
                            )
                            status_placeholder.empty()
                            st.session_state.messages.append(
                                {"role": "user", "content": "Please analyze this study material and explain it step-by-step."}
                            )
                            st.session_state.messages.append(
                                {"role": "assistant", "content": analysis}
                            )
                            st.rerun()
                        except Exception as e:
                            status_placeholder.empty()
                            st.error(format_friendly_error(e, primary_model))

    # ==========================================================================
    # BRANCH 3: STUDY WORKSPACE (Analysis completed & Interactive Tutoring)
    # ==========================================================================
    else:
        # Two-Column Workspace Layout
        col_material, col_solution = st.columns([5, 7], gap="large")

        # LEFT COLUMN: Study Material
        with col_material:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="workspace-card-header">
                        <h4 class="workspace-card-title">📷 Study Material</h4>
                        <span style="font-size: 11px; background: #EEF2FF; color: #4F46E5; padding: 2px 8px; border-radius: 8px; border: 1px solid #C7D2FE; font-weight: 600;">{st.session_state.uploaded_image_name or 'Uploaded File'}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.session_state.uploaded_image_bytes:
                    render_image(st.session_state.uploaded_image_bytes)

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("📤 Upload New", use_container_width=True, key="btn_upload_new_ws"):
                        st.session_state.messages = []
                        st.session_state.uploaded_image_bytes = None
                        st.session_state.uploaded_image_mime = None
                        st.session_state.uploaded_image_name = None
                        st.session_state.last_summary = None
                        st.rerun()
                with col_btn2:
                    if st.button("🔄 Clear All", use_container_width=True, key="btn_clear_ws"):
                        st.session_state.messages = []
                        st.session_state.uploaded_image_bytes = None
                        st.session_state.uploaded_image_mime = None
                        st.session_state.uploaded_image_name = None
                        st.session_state.last_summary = None
                        st.rerun()

        # RIGHT COLUMN: AI Analysis & Solutions
        with col_solution:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="workspace-card-header">
                        <h4 class="workspace-card-title">💡 SnapStudy Analysis & Solutions</h4>
                        <span style="font-size: 11px; background: #ECFDF5; color: #047857; padding: 2px 8px; border-radius: 8px; border: 1px solid #A7F3D0; font-weight: 600;">Structured Breakdown</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                primary_analysis = ""
                if len(st.session_state.messages) >= 2:
                    primary_analysis = st.session_state.messages[1]["content"]
                elif len(st.session_state.messages) == 1:
                    primary_analysis = st.session_state.messages[0]["content"]

                if primary_analysis:
                    render_structured_analysis(primary_analysis)

        # ----------------------------------------------------------------------
        # Interactive Chat & Follow-Up Q&A
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <h4 style="font-size: 16.5px; font-weight: 700; color: #172033; margin: 0; display: flex; align-items: center; gap: 8px;">
                    <span style="color: #7C3AED;">💬</span> Interactive Tutoring & Q&A
                </h4>
                <span style="font-size: 12px; color: #64748B;">Ask questions, request simpler explanations, or get more practice</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render subsequent conversation turns (from index 2 onwards)
        if len(st.session_state.messages) > 2:
            for msg in st.session_state.messages[2:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # View Last Generated Summary (if available)
        if st.session_state.last_summary:
            with st.expander("📄 View Last Generated Study Summary"):
                st.markdown(st.session_state.last_summary)

    # ==========================================================================
    # MODERN AI-ASSISTANT CHAT COMPOSER
    # ==========================================================================
    pill_prompt = None

    if has_image:
        # Context Label & Attached Material State
        st.markdown(
            f"""
            <div class="composer-meta-wrapper">
                <div class="composer-title-group">
                    <div class="composer-title">💬 Ask SnapStudy</div>
                    <div class="composer-subtitle">Ask questions, request explanations, or generate practice.</div>
                </div>
                <div class="composer-attached-pill">
                    📎 Study material attached: {st.session_state.uploaded_image_name or 'Uploaded File'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggestion Chips above the input
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        with c_col1:
            st.markdown('<div class="composer-chip-btn">', unsafe_allow_html=True)
            if st.button("💡 Explain this simply", use_container_width=True, key="pill_simple"):
                pill_prompt = "Explain this simply"
            st.markdown('</div>', unsafe_allow_html=True)
        with c_col2:
            st.markdown('<div class="composer-chip-btn">', unsafe_allow_html=True)
            if st.button("📝 Give me an example", use_container_width=True, key="pill_example"):
                pill_prompt = "Give me an example"
            st.markdown('</div>', unsafe_allow_html=True)
        with c_col3:
            st.markdown('<div class="composer-chip-btn">', unsafe_allow_html=True)
            if st.button("🎯 Create practice questions", use_container_width=True, key="pill_practice"):
                pill_prompt = "Create practice questions"
            st.markdown('</div>', unsafe_allow_html=True)
        with c_col4:
            st.markdown('<div class="composer-chip-btn">', unsafe_allow_html=True)
            if st.button("🔢 Explain step by step", use_container_width=True, key="pill_step"):
                pill_prompt = "Explain step by step"
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Empty State: subtle hint when no material is uploaded
        st.markdown(
            """
            <div class="composer-meta-wrapper">
                <div class="composer-title-group">
                    <div class="composer-title">💬 Ask SnapStudy</div>
                    <div class="composer-subtitle">Ask questions, request explanations, or generate practice.</div>
                </div>
                <div class="composer-empty-pill">
                    💡 Upload study material to ask questions about it
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Modern Sticky Chat Input
    typed_prompt = st.chat_input(
        "Ask SnapStudy anything about this material...",
        disabled=not bool(api_key),
    )

    active_query = pill_prompt or typed_prompt

    if active_query and active_query.strip():
        # Check profile completion
        if not st.session_state.onboarding_complete:
            st.warning("Please complete your profile (name & email) in the sidebar to keep your study sessions organized.")

        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            chat_status = st.empty()
            with st.spinner("SnapStudy is thinking..."):
                try:
                    client = get_gemini_client(api_key=api_key)
                    reply = generate_chat_reply(
                        client=client,
                        messages=st.session_state.messages,
                        image_bytes=st.session_state.uploaded_image_bytes,
                        mime_type=st.session_state.uploaded_image_mime,
                        models=active_models,
                        status_callback=lambda msg: chat_status.info("⏳ Gemini is temporarily busy. Retrying automatically..."),
                    )
                    chat_status.empty()
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception as e:
                    chat_status.empty()
                    st.error(format_friendly_error(e, primary_model))


if __name__ == "__main__":
    main()



