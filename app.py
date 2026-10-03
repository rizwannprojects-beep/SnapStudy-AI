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
        st.markdown(f'<div class="analysis-content analysis-scroll-box">\n\n{content}\n\n</div>', unsafe_allow_html=True)
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
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

            /* ==================================================================
               1. UNIFIED DESIGN SYSTEM (LIGHT ACADEMIC AI DASHBOARD)
               ================================================================== */
            :root {
                --primary: #4F46E5;
                --primary-hover: #4338CA;
                --accent: #06B6D4;
                --secondary: #7C3AED;
                --success: #10B981;
                --warning: #F59E0B;
                --error: #EF4444;

                --text-main: #172033;
                --text-secondary: #64748B;
                --text-muted: #94A3B8;

                --bg-page: #F5F7FF;
                --bg-card: #FFFFFF;
                --bg-sidebar: #EEF1FF;
                --bg-input: #FFFFFF;
                --input-text: #172033;
                --input-placeholder: #64748B;

                --border: #D9DDF0;
                --border-input: #CBD5E1;
                --border-soft: #E8ECF5;
                --border-focus: #4F46E5;

                --radius-card: 16px;
                --radius-control: 10px;
                --radius-badge: 20px;
                --radius-composer: 22px;
                --radius-sm: 6px;

                --shadow-card: 0 4px 16px rgba(23, 32, 51, 0.04);
                --shadow-elevated: 0 8px 24px rgba(23, 32, 51, 0.08);

                --content-max-width: 1400px;
                --chat-max-width: 1100px;
            }

            /* ==================================================================
               2. GLOBAL BASELINE & MATERIAL ICONS PROTECTION
               ================================================================== */
            html, body, .stApp {
                font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                color: var(--text-main) !important;
                background-color: var(--bg-page) !important;
                overflow-x: hidden !important;
            }

            /* Material Symbols font protection so icons never become text (e.g. 'uploadUpload') */
            [data-testid="stIconMaterial"],
            .material-symbols-rounded,
            [class*="material-symbols"] {
                font-family: 'Material Symbols Rounded' !important;
                font-weight: normal !important;
                font-style: normal !important;
                font-size: 20px !important;
                line-height: 1 !important;
                letter-spacing: normal !important;
                text-transform: none !important;
                display: inline-block !important;
                white-space: nowrap !important;
                word-wrap: normal !important;
                direction: ltr !important;
                -webkit-font-feature-settings: 'liga' !important;
                font-feature-settings: 'liga' !important;
            }

            /* Streamlit Decoration Line & Header Clean-up */
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

            footer {
                display: none !important;
            }

            /* Main Content Container & Alignment Grid */
            .stMainBlockContainer, .main .block-container {
                max-width: var(--content-max-width) !important;
                padding: 24px 32px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                box-sizing: border-box !important;
            }

            /* Typography Hierarchy */
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
                color: var(--text-main) !important;
                letter-spacing: -0.02em !important;
                margin-top: 0 !important;
            }

            h1 { font-size: 32px !important; font-weight: 700 !important; line-height: 1.25 !important; }
            h2 { font-size: 24px !important; font-weight: 700 !important; line-height: 1.3 !important; }
            h3 { font-size: 20px !important; font-weight: 600 !important; line-height: 1.35 !important; }
            h4 { font-size: 18px !important; font-weight: 600 !important; line-height: 1.4 !important; }

            p {
                font-size: 15px !important;
                line-height: 1.6 !important;
                color: var(--text-secondary);
                margin-bottom: 8px !important;
            }

            /* ==================================================================
               3. STREAMLIT INPUT OVERRIDES (GLOBAL)
               ================================================================== */
            input,
            textarea {
                color: var(--text-main) !important;
                background-color: #FFFFFF !important;
                font-family: inherit !important;
            }

            input::placeholder,
            textarea::placeholder {
                color: var(--input-placeholder) !important;
                opacity: 1 !important;
            }

            div[data-baseweb="input"],
            div[data-baseweb="base-input"] {
                background-color: #FFFFFF !important;
            }

            /* ==================================================================
               4. GLOBAL BUTTON CONTRAST & VISIBILITY (FIX FOR ALL BUTTONS)
               ================================================================== */
            /* Crucial: enforce child text/paragraphs to inherit button color */
            button,
            button *,
            button p,
            button span:not([data-testid="stIconMaterial"]),
            button div,
            div[data-testid="stButton"] button,
            div[data-testid="stButton"] button *,
            div[data-testid="stButton"] button p,
            div[data-testid="stButton"] button span:not([data-testid="stIconMaterial"]),
            div[data-testid="stFormSubmitButton"] button,
            div[data-testid="stFormSubmitButton"] button *,
            div[data-testid="stFormSubmitButton"] button p,
            button[data-testid^="stBaseButton"] * {
                color: inherit !important;
                margin: 0 !important;
            }

            /* Base button styling */
            div[data-testid="stButton"] button,
            div[data-testid="stFormSubmitButton"] button,
            button[data-testid^="stBaseButton"] {
                border-radius: var(--radius-control) !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                padding: 10px 18px !important;
                min-height: 44px !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                transition: all 0.15s ease !important;
                box-sizing: border-box !important;
                text-align: center !important;
                cursor: pointer !important;
            }

            /* Primary purple buttons */
            div[data-testid="stButton"] button[kind="primary"],
            div[data-testid="stButton"] button[data-testid*="primary"],
            div[data-testid="stFormSubmitButton"] button,
            button[kind="primaryFormSubmit"],
            button[data-testid="stBaseButton-primary"],
            button[data-testid="stBaseButton-primaryFormSubmit"] {
                background-color: var(--primary) !important;
                color: #FFFFFF !important;
                border: 1px solid var(--primary) !important;
                box-shadow: 0 1px 3px rgba(79, 70, 229, 0.2) !important;
            }

            div[data-testid="stButton"] button[kind="primary"]:hover,
            div[data-testid="stButton"] button[data-testid*="primary"]:hover,
            div[data-testid="stFormSubmitButton"] button:hover,
            button[kind="primaryFormSubmit"]:hover,
            button[data-testid="stBaseButton-primary"]:hover,
            button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
                background-color: var(--primary-hover) !important;
                border-color: var(--primary-hover) !important;
                color: #FFFFFF !important;
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
            }

            /* Secondary buttons */
            div[data-testid="stButton"] button[kind="secondary"],
            div[data-testid="stButton"] button[data-testid*="secondary"],
            button[data-testid="stBaseButton-secondary"],
            button[data-testid="stBaseButton-secondaryFormSubmit"],
            div[data-testid="stButton"] button:not([kind="primary"]):not(.analyze-coral-btn button) {
                background-color: #FFFFFF !important;
                color: var(--text-main) !important;
                border: 1px solid var(--border-input) !important;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            }

            div[data-testid="stButton"] button[kind="secondary"]:hover,
            div[data-testid="stButton"] button[data-testid*="secondary"]:hover,
            button[data-testid="stBaseButton-secondary"]:hover,
            button[data-testid="stBaseButton-secondaryFormSubmit"]:hover,
            div[data-testid="stButton"] button:not([kind="primary"]):hover {
                background-color: #F1F5F9 !important;
                color: var(--text-main) !important;
                border-color: var(--text-muted) !important;
            }

            /* Disabled buttons */
            button:disabled,
            button:disabled *,
            div[data-testid="stButton"] button:disabled,
            div[data-testid="stButton"] button:disabled *,
            div[data-testid="stFormSubmitButton"] button:disabled,
            div[data-testid="stFormSubmitButton"] button:disabled *,
            button[data-testid^="stBaseButton"]:disabled,
            button[data-testid^="stBaseButton"]:disabled * {
                background-color: #E2E8F0 !important;
                color: var(--text-muted) !important;
                border-color: var(--border-input) !important;
                cursor: not-allowed !important;
                box-shadow: none !important;
                opacity: 0.7 !important;
            }

            /* 5. FIX THE "ANALYZE MATERIAL" BUTTON */
            .analyze-coral-btn button,
            .analyze-coral-btn button * {
                background-color: #4F46E5 !important;
                color: #FFFFFF !important;
                font-weight: 600 !important;
                font-size: 15px !important;
                min-height: 52px !important;
                height: 52px !important;
                border-radius: 14px !important;
                border: 1px solid #4F46E5 !important;
                box-shadow: 0 4px 14px rgba(79, 70, 229, 0.28) !important;
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 8px !important;
            }

            .analyze-coral-btn button:hover,
            .analyze-coral-btn button:hover * {
                background-color: #4338CA !important;
                border-color: #4338CA !important;
                color: #FFFFFF !important;
                box-shadow: 0 6px 18px rgba(79, 70, 229, 0.38) !important;
                transform: translateY(-1px) !important;
            }

            /* ==================================================================
               5. SIDEBAR ALIGNMENT & DESIGN (320px - 340px)
               ================================================================== */
            section[data-testid="stSidebar"] {
                width: 330px !important;
                min-width: 320px !important;
                max-width: 340px !important;
                background-color: var(--bg-sidebar) !important;
                border-right: 1px solid var(--border) !important;
                box-shadow: none !important;
            }

            section[data-testid="stSidebar"] > div:first-child {
                padding: 24px 16px !important;
            }

            .sidebar-section-title {
                font-size: 12.5px;
                font-weight: 700;
                color: var(--text-main);
                letter-spacing: 0.04em;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 6px;
                text-transform: uppercase;
            }

            /* Sidebar Card Containers */
            section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: var(--bg-card) !important;
                border: 1px solid var(--border) !important;
                border-radius: 14px !important;
                padding: 18px 16px !important;
                margin-bottom: 16px !important;
                box-shadow: 0 2px 8px rgba(23, 32, 51, 0.03) !important;
            }

            /* 6. FIX PROFILE FORM INPUTS */
            section[data-testid="stSidebar"] div[data-testid="stTextInput"] {
                width: 100% !important;
                margin-bottom: 12px !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
                background-color: #FFFFFF !important;
                border: 1px solid var(--border-input) !important;
                border-radius: var(--radius-control) !important;
                height: 48px !important;
                min-height: 48px !important;
                transition: all 0.15s ease !important;
                box-sizing: border-box !important;
                padding: 0 !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {
                border: 2px solid var(--primary) !important;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
                background-color: #FFFFFF !important;
                color: var(--text-main) !important;
                caret-color: var(--primary) !important;
                font-size: 14px !important;
                font-weight: 500 !important;
                padding: 10px 14px !important;
                width: 100% !important;
                height: 100% !important;
                box-sizing: border-box !important;
                border: none !important;
                border-radius: var(--radius-control) !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stTextInput"] input::placeholder {
                color: var(--input-placeholder) !important;
                opacity: 1 !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stTextInput"] label {
                font-size: 13px !important;
                font-weight: 600 !important;
                color: var(--text-main) !important;
                margin-bottom: 6px !important;
                display: block !important;
            }

            /* Hide form instructions like 'Press Enter to submit form' */
            form[aria-label="sidebar_onboarding_form"] [data-testid="InputInstructions"],
            div[data-testid="stTextInput"] [data-testid="InputInstructions"],
            div[data-testid="InputInstructions"] {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                overflow: hidden !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            /* Save Profile Button (Full Width, Purple, White Text) */
            section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] button,
            section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] button * {
                width: 100% !important;
                height: 48px !important;
                min-height: 48px !important;
                background-color: var(--primary) !important;
                color: #FFFFFF !important;
                border: 1px solid var(--primary) !important;
                border-radius: var(--radius-control) !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
                transition: all 0.15s ease !important;
                box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25) !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] button:hover,
            section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] button:hover * {
                background-color: var(--primary-hover) !important;
                border-color: var(--primary-hover) !important;
                color: #FFFFFF !important;
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
            }

            /* Graceful truncation for long filenames */
            section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                max-width: 100% !important;
            }

            /* ==================================================================
               6. TOP HEADER CARD
               ================================================================== */
            .snapstudy-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: var(--radius-card);
                padding: 16px 24px;
                box-shadow: var(--shadow-card);
                margin-bottom: 20px;
                box-sizing: border-box;
                width: 100%;
            }

            .header-brand {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .header-brand-icon {
                font-size: 22px;
                background: #EEF2FF;
                border: 1px solid #C7D2FE;
                border-radius: 10px;
                width: 42px;
                height: 42px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }

            .header-brand-title {
                font-size: 19px;
                font-weight: 700;
                color: var(--primary);
                line-height: 1.2;
                letter-spacing: -0.02em;
            }

            .header-brand-subtitle {
                font-size: 12.5px;
                font-weight: 500;
                color: var(--text-secondary);
                line-height: 1.2;
                margin-top: 2px;
            }

            .header-status {
                display: inline-flex;
                align-items: center;
                gap: 7px;
                background: #ECFDF5;
                color: #065F46;
                border: 1px solid #A7F3D0;
                padding: 6px 14px;
                border-radius: var(--radius-badge);
                font-size: 12.5px;
                font-weight: 600;
                white-space: nowrap;
                flex-shrink: 0;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--success);
                display: inline-block;
                box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
            }

            .header-accent-line {
                height: 3px;
                background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 50%, var(--secondary) 100%);
                border-radius: 3px;
                margin-top: -12px;
                margin-bottom: 24px;
            }

            /* ==================================================================
               7. WELCOME & CARDS SYSTEM
               ================================================================== */
            .hero-wrapper {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: var(--radius-card);
                padding: 32px;
                box-shadow: var(--shadow-card);
                margin-bottom: 24px;
                box-sizing: border-box;
                width: 100%;
            }

            .hero-heading {
                font-size: 32px !important;
                font-weight: 700 !important;
                color: var(--text-main) !important;
                margin: 0 0 8px 0 !important;
                line-height: 1.25 !important;
            }

            .hero-subheading {
                font-size: 16px !important;
                color: var(--text-secondary) !important;
                margin: 0 !important;
                line-height: 1.5 !important;
            }

            /* Container cards in main content */
            .main div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--bg-card) !important;
                border: 1px solid var(--border) !important;
                border-radius: var(--radius-card) !important;
                box-shadow: var(--shadow-card) !important;
                padding: 24px !important;
                margin-bottom: 24px !important;
                box-sizing: border-box !important;
            }

            .workspace-card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding-bottom: 14px;
                border-bottom: 1px solid var(--border-soft);
                margin-bottom: 16px;
            }

            .workspace-card-title {
                font-size: 16.5px;
                font-weight: 700;
                color: var(--text-main);
                margin: 0;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            /* ==================================================================
               8. STUDY MATERIAL & FILE UPLOADER (7. FIX UPLOAD SECTION)
               ================================================================== */
            div[data-testid="stImage"] {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                width: 100% !important;
                margin: 0 auto !important;
            }

            div[data-testid="stImage"] img {
                border-radius: 12px !important;
                border: 1px solid var(--border) !important;
                max-width: 100% !important;
                max-height: 520px !important;
                object-fit: contain !important;
                box-shadow: 0 2px 8px rgba(23, 32, 51, 0.04) !important;
            }

            /* Dropzone Light Styling with dashed border & centered layout */
            [data-testid="stFileUploaderDropzone"],
            [data-testid="stFileUploadDropzone"],
            section[data-testid="stFileUploadDropzone"],
            div[data-testid="stFileUploader"] section {
                background-color: #FAFAFF !important;
                border: 2px dashed var(--border-input) !important;
                border-radius: 14px !important;
                padding: 24px 16px !important;
                text-align: center !important;
                color: var(--text-secondary) !important;
                transition: all 0.2s ease !important;
                box-sizing: border-box !important;
            }

            [data-testid="stFileUploaderDropzone"]:hover,
            [data-testid="stFileUploadDropzone"]:hover,
            section[data-testid="stFileUploadDropzone"]:hover,
            div[data-testid="stFileUploader"] section:hover {
                border-color: var(--primary) !important;
                background-color: #F5F7FF !important;
            }

            /* Single, clean Upload button inside dropzone */
            div[data-testid="stFileUploader"] section button,
            div[data-testid="stFileUploader"] section button * {
                background-color: #FFFFFF !important;
                color: var(--text-main) !important;
                border: 1px solid var(--border-input) !important;
                border-radius: 8px !important;
                font-size: 13.5px !important;
                font-weight: 600 !important;
                padding: 8px 18px !important;
                margin: 0 auto !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
            }

            div[data-testid="stFileUploader"] section button:hover,
            div[data-testid="stFileUploader"] section button:hover * {
                background-color: #F1F5F9 !important;
                border-color: var(--text-muted) !important;
                color: var(--text-main) !important;
            }

            div[data-testid="stFileUploader"] section button span[data-testid="stIconMaterial"] {
                color: var(--primary) !important;
                font-size: 18px !important;
            }

            div[data-testid="stFileUploader"] section div,
            div[data-testid="stFileUploader"] section span:not([data-testid="stIconMaterial"]),
            div[data-testid="stFileUploader"] section small {
                color: var(--text-secondary) !important;
                font-size: 13px !important;
            }

            /* ==================================================================
               9. FEATURE CARDS RESPONSIVE GRID
               ================================================================== */
            .feature-section-title {
                font-size: 20px;
                font-weight: 700;
                color: var(--text-main);
                margin-top: 24px;
                margin-bottom: 14px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .feature-cards-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                margin-bottom: 24px;
                width: 100%;
                box-sizing: border-box;
            }

            @media (max-width: 1100px) {
                .feature-cards-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
            }

            @media (max-width: 600px) {
                .feature-cards-grid {
                    grid-template-columns: 1fr;
                }
            }

            .feature-card-item {
                border-radius: var(--radius-card);
                padding: 22px 20px;
                min-height: 180px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-shadow: 0 2px 8px rgba(23, 32, 51, 0.04);
                box-sizing: border-box;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }

            .feature-card-item:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(23, 32, 51, 0.08);
            }

            .card-math { background: #EEF2FF; border: 1px solid #C7D2FE; }
            .card-notes { background: #F5F3FF; border: 1px solid #DDD6FE; }
            .card-code { background: #ECFEFF; border: 1px solid #A5F3FC; }
            .card-diagrams { background: #FFF7ED; border: 1px solid #FED7AA; }

            .feature-icon-box {
                font-size: 24px;
                width: 44px;
                height: 44px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 10px;
                margin-bottom: 12px;
            }
            .card-math .feature-icon-box { background: #E0E7FF; border: 1px solid #C7D2FE; }
            .card-notes .feature-icon-box { background: #EDE9FE; border: 1px solid #DDD6FE; }
            .card-code .feature-icon-box { background: #CFFAFE; border: 1px solid #A5F3FC; }
            .card-diagrams .feature-icon-box { background: #FFEDD5; border: 1px solid #FED7AA; }

            .feature-title {
                font-size: 15.5px;
                font-weight: 700;
                margin-bottom: 6px;
            }
            .card-math .feature-title { color: var(--primary); }
            .card-notes .feature-title { color: var(--secondary); }
            .card-code .feature-title { color: #0891B2; }
            .card-diagrams .feature-title { color: #EA580C; }

            .feature-desc {
                font-size: 13.5px;
                color: var(--text-secondary);
                line-height: 1.5;
            }

            .feature-tag-wrapper {
                margin-top: 14px;
            }

            .feature-tag {
                font-size: 11px;
                font-weight: 700;
                padding: 3px 8px;
                border-radius: 6px;
                letter-spacing: 0.03em;
                display: inline-block;
            }
            .tag-indigo { background: #E0E7FF; color: var(--primary); }
            .tag-violet { background: #EDE9FE; color: var(--secondary); }
            .tag-cyan   { background: #CFFAFE; color: #0891B2; }
            .tag-coral  { background: #FFEDD5; color: #EA580C; }

            /* ==================================================================
               10. ANALYSIS SECTION (WITH SCROLL BOX)
               ================================================================== */
            .analysis-scroll-box {
                max-height: 640px;
                overflow-y: auto;
                padding-right: 8px;
            }
            .analysis-scroll-box::-webkit-scrollbar {
                width: 6px;
            }
            .analysis-scroll-box::-webkit-scrollbar-thumb {
                background: #CBD5E1;
                border-radius: 4px;
            }

            .analysis-content {
                font-size: 15px;
                line-height: 1.65;
                color: var(--text-main);
                word-break: break-word;
                overflow-wrap: break-word;
            }

            .analysis-content h2, .analysis-content h3 {
                color: var(--text-main) !important;
                font-weight: 700 !important;
                margin-top: 20px !important;
                margin-bottom: 8px !important;
            }

            .analysis-section-header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-top: 14px;
                margin-bottom: 8px;
                padding-bottom: 6px;
            }

            .analysis-section-icon {
                font-size: 18px;
                line-height: 1;
            }

            .analysis-section-title {
                font-size: 16px;
                font-weight: 700;
                color: var(--text-main);
            }

            .analysis-section-pill {
                font-size: 11px;
                font-weight: 700;
                padding: 2px 8px;
                border-radius: var(--radius-badge);
                letter-spacing: 0.02em;
                margin-left: 4px;
            }

            .pill-indigo { background: #EEF2FF; color: var(--primary); border: 1px solid #C7D2FE; }
            .pill-violet { background: #F5F3FF; color: var(--secondary); border: 1px solid #DDD6FE; }
            .pill-cyan   { background: #ECFEFF; color: #0891B2; border: 1px solid #A5F3FC; }
            .pill-coral  { background: #FFF7ED; color: #EA580C; border: 1px solid #FED7AA; }

            .analysis-divider {
                height: 1px;
                background-color: var(--border-soft);
                margin: 20px 0;
            }

            /* Final Answer Highlight Box */
            .final-answer-card {
                margin-top: 20px;
                margin-bottom: 4px;
            }

            .final-answer-header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 6px;
            }

            .final-answer-icon {
                font-size: 18px;
            }

            .final-answer-title {
                font-size: 15px;
                font-weight: 700;
                color: #065F46;
            }

            .final-answer-box {
                background: #ECFDF5;
                border: 1px solid var(--success);
                border-radius: 12px;
                padding: 14px 16px;
                color: #065F46;
                font-size: 15px;
                font-weight: 600;
                line-height: 1.5;
                margin-bottom: 20px;
            }

            /* Practice Card Box */
            .practice-card-box {
                background: #F8FAFC;
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 14px 16px;
                margin-bottom: 12px;
                font-size: 14px;
                line-height: 1.6;
                color: var(--text-main);
            }

            .practice-q-badge {
                display: inline-block;
                font-size: 11px;
                font-weight: 700;
                background: #FFF7ED;
                color: #EA580C;
                border: 1px solid #FED7AA;
                padding: 2px 8px;
                border-radius: var(--radius-badge);
                margin-bottom: 8px;
            }

            /* Stepper Card */
            .stepper-card {
                background: #F8FAFC;
                border: 1px solid var(--border);
                border-radius: var(--radius-control);
                padding: 16px;
                margin-bottom: 16px;
            }

            .stepper-header {
                font-size: 14px;
                font-weight: 700;
                color: var(--primary);
                margin-bottom: 10px;
            }

            .stepper-row {
                font-size: 13px;
                margin-bottom: 6px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .step-done { color: var(--success); font-weight: 600; }
            .step-active { color: var(--primary); font-weight: 700; }
            .step-wait { color: var(--text-muted); }

            /* Code Blocks */
            .analysis-content pre, .stMainBlockContainer pre {
                background-color: #1F2937 !important;
                color: #F8FAFC !important;
                border: 1px solid #374151 !important;
                border-radius: var(--radius-control) !important;
                padding: 14px 16px !important;
                overflow-x: auto !important;
                font-size: 13.5px !important;
                line-height: 1.5 !important;
            }

            .analysis-content code:not(pre code) {
                background-color: #EEF2FF !important;
                color: var(--primary) !important;
                padding: 2px 6px !important;
                border-radius: 4px !important;
                border: 1px solid #C7D2FE !important;
                font-size: 13px !important;
            }

            /* ==================================================================
               11. CHAT CONTAINER & MODERN CHATGPT-STYLE COMPOSER (2 & 3)
               ================================================================== */
            /* 3. CHAT SECTION CONTAINER */
            .chat-section {
                width: 100% !important;
                max-width: var(--chat-max-width) !important;
                margin: 0 auto !important;
                box-sizing: border-box !important;
                padding: 0 !important;
            }

            .chat-header {
                width: 100% !important;
                margin-bottom: 12px !important;
            }

            /* Distinct Chat Message Cards */
            div[data-testid="stChatMessage"] {
                background-color: #FFFFFF !important;
                border: 1px solid var(--border) !important;
                border-radius: 16px !important;
                padding: 16px 20px !important;
                margin-bottom: 16px !important;
                box-shadow: 0 2px 8px rgba(23, 32, 51, 0.03) !important;
                max-width: var(--chat-max-width) !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            div[data-testid="stChatMessage"][data-testid*="user"],
            div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
                background-color: #EEF2FF !important;
                border-color: #C7D2FE !important;
            }

            .composer-meta-wrapper {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-top: 24px;
                margin-bottom: 12px;
                padding: 0 4px;
                width: 100%;
                max-width: var(--chat-max-width);
                margin-left: auto;
                margin-right: auto;
                box-sizing: border-box;
            }

            .composer-title-group {
                display: flex;
                flex-direction: column;
            }

            .composer-title {
                font-size: 16px;
                font-weight: 700;
                color: var(--text-main);
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .composer-subtitle {
                font-size: 13px;
                color: var(--text-secondary);
                margin-top: 2px;
            }

            .composer-attached-pill {
                font-size: 12px;
                background: #EEF2FF;
                color: var(--primary);
                border: 1px solid #C7D2FE;
                padding: 4px 12px;
                border-radius: var(--radius-badge);
                font-weight: 600;
            }

            .composer-empty-pill {
                font-size: 12px;
                background: #F1F5F9;
                color: var(--text-secondary);
                border: 1px solid #CBD5E1;
                padding: 4px 12px;
                border-radius: var(--radius-badge);
                font-weight: 500;
            }

            /* Suggestion Chips */
            .composer-chips-container {
                max-width: var(--chat-max-width) !important;
                margin: 0 auto 12px auto !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            .composer-chip-btn button {
                background: #FFFFFF !important;
                border: 1px solid var(--border) !important;
                color: var(--text-main) !important;
                border-radius: 20px !important;
                font-size: 13px !important;
                font-weight: 500 !important;
                padding: 6px 14px !important;
                min-height: 38px !important;
                height: 38px !important;
                margin-bottom: 8px !important;
                box-shadow: 0 1px 3px rgba(23, 32, 51, 0.03) !important;
                transition: all 0.15s ease !important;
                width: 100% !important;
            }

            .composer-chip-btn button:hover {
                border-color: var(--primary) !important;
                color: var(--primary) !important;
                background: #EEF2FF !important;
                box-shadow: 0 2px 6px rgba(79, 70, 229, 0.1) !important;
            }

            /* 2. CHATGPT-STYLE CHAT COMPOSER OVERRIDES */
            /* Streamlit stBottom: Natural document flow, cleanly connected to chat */
            div[data-testid="stBottom"] {
                position: relative !important;
                bottom: auto !important;
                left: auto !important;
                right: auto !important;
                width: 100% !important;
                max-width: var(--chat-max-width) !important;
                margin: 16px auto 32px auto !important;
                padding: 0 !important;
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                box-sizing: border-box !important;
                z-index: 10 !important;
            }

            div[data-testid="stBottom"] > div,
            div[data-testid="stBottomBlockContainer"] {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                max-width: var(--chat-max-width) !important;
                width: 100% !important;
                padding: 0 !important;
                margin: 0 auto !important;
                box-sizing: border-box !important;
            }

            /* Eliminate any accidental card wrapper from stBottom */
            div[data-testid="stBottom"] div[data-testid="stVerticalBlockBorderWrapper"],
            div[data-testid="stBottom"] div[data-testid="stVerticalBlock"] {
                background: transparent !important;
                border: none !important;
                border-radius: 0 !important;
                box-shadow: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }

            /* Outer chat input element (Oe) */
            div[data-testid="stChatInput"] {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
                max-width: var(--chat-max-width) !important;
                width: 100% !important;
                margin: 0 auto !important;
                box-sizing: border-box !important;
                display: flex !important;
                flex-direction: column !important;
            }

            /* Inner Composer Card (ke): Modern ChatGPT style composer */
            div[data-testid="stChatInput"] > div {
                background-color: #FFFFFF !important;
                border: 1px solid var(--border) !important;
                border-radius: var(--radius-composer) !important;
                box-shadow: 0 4px 20px rgba(23, 32, 51, 0.06) !important;
                min-height: 64px !important;
                height: auto !important;
                padding: 8px 16px !important;
                width: 100% !important;
                max-width: var(--chat-max-width) !important;
                margin: 0 auto !important;
                box-sizing: border-box !important;
                display: flex !important;
                flex-direction: row !important;
                align-items: center !important;
                justify-content: space-between !important;
                gap: 12px !important;
                transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
            }

            div[data-testid="stChatInput"] > div:focus-within {
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15), 0 4px 20px rgba(79, 70, 229, 0.08) !important;
            }

            /* Textarea container stretch */
            div[data-testid="stChatInput"] > div > div:first-child,
            div[data-testid="stChatInput"] [data-baseweb="textarea"],
            div[data-testid="stChatInput"] form {
                flex: 1 1 auto !important;
                width: 100% !important;
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                display: flex !important;
                align-items: center !important;
            }

            /* Composer Textarea */
            div[data-testid="stChatInput"] textarea,
            [data-testid="stChatInputTextArea"] {
                background: transparent !important;
                color: var(--text-main) !important;
                caret-color: var(--primary) !important;
                font-size: 15px !important;
                font-family: inherit !important;
                line-height: 1.5 !important;
                border: none !important;
                box-shadow: none !important;
                padding: 8px 10px !important;
                resize: none !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            div[data-testid="stChatInput"] textarea::placeholder,
            [data-testid="stChatInputTextArea"]::placeholder {
                color: var(--input-placeholder) !important;
                opacity: 1 !important;
            }

            /* Composer Send Button (Inside composer on the right, ~44-48px square/circular) */
            div[data-testid="stChatInput"] button,
            button[data-testid="stChatInputSubmitButton"] {
                background-color: var(--primary) !important;
                color: #FFFFFF !important;
                border-radius: 50% !important;
                width: 44px !important;
                height: 44px !important;
                min-width: 44px !important;
                min-height: 44px !important;
                border: none !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                cursor: pointer !important;
                transition: all 0.15s ease !important;
                flex-shrink: 0 !important;
                box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25) !important;
                padding: 0 !important;
                margin: 0 !important;
            }

            div[data-testid="stChatInput"] button:hover,
            button[data-testid="stChatInputSubmitButton"]:hover {
                background-color: var(--primary-hover) !important;
                transform: scale(1.05) !important;
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
            }

            div[data-testid="stChatInput"] button:disabled,
            button[data-testid="stChatInputSubmitButton"]:disabled {
                background-color: #CBD5E1 !important;
                color: var(--text-muted) !important;
                cursor: not-allowed !important;
                transform: none !important;
                box-shadow: none !important;
            }

            div[data-testid="stChatInput"] button svg,
            button[data-testid="stChatInputSubmitButton"] svg {
                fill: #FFFFFF !important;
                color: #FFFFFF !important;
                width: 20px !important;
                height: 20px !important;
            }

            /* ==================================================================
               12. ALERTS & EXPANDERS
               ================================================================== */
            div[data-testid="stAlert"] {
                border-radius: var(--radius-control) !important;
                padding: 12px 16px !important;
                font-size: 14px !important;
                box-shadow: none !important;
            }

            div[data-testid="stNotification"] {
                border-radius: var(--radius-control) !important;
            }

            div[data-testid="stExpander"] {
                border: 1px solid var(--border) !important;
                border-radius: 14px !important;
                background: var(--bg-card) !important;
                box-shadow: var(--shadow-card) !important;
            }

            /* ==================================================================
               13. RESPONSIVE DESIGN
               ================================================================== */
            @media (max-width: 1024px) {
                section[data-testid="stSidebar"] {
                    width: 290px !important;
                    min-width: 270px !important;
                }
                .stMainBlockContainer, .main .block-container {
                    padding: 20px 20px !important;
                }
                .hero-heading {
                    font-size: 28px !important;
                }
                .hero-wrapper {
                    padding: 24px !important;
                }
            }

            @media (max-width: 768px) {
                .snapstudy-header {
                    flex-direction: column !important;
                    align-items: flex-start !important;
                    gap: 12px !important;
                    padding: 14px 18px !important;
                }

                .header-status {
                    align-self: flex-start !important;
                }

                .composer-meta-wrapper {
                    flex-direction: column !important;
                    align-items: flex-start !important;
                    gap: 6px !important;
                }

                .workspace-card-header {
                    flex-direction: column !important;
                    align-items: flex-start !important;
                    gap: 6px !important;
                }

                .stMainBlockContainer, .main .block-container {
                    padding: 16px 14px !important;
                }

                div[data-testid="stChatInput"] > div {
                    padding: 6px 12px !important;
                    min-height: 56px !important;
                }
            }

            @media (max-width: 480px) {
                .header-brand-title {
                    font-size: 16px !important;
                }

                .hero-heading {
                    font-size: 22px !important;
                }

                div[data-testid="stChatInput"] > div {
                    padding: 4px 10px !important;
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

        # Feature Cards: Responsive Grid Container
        st.markdown(
            """
            <div class="feature-section-title">✨ What can SnapStudy do?</div>
            <div class="feature-cards-grid">
                <div class="feature-card-item card-math">
                    <div>
                        <div class="feature-icon-box">🧮</div>
                        <div class="feature-title">Solve Problems</div>
                        <div class="feature-desc">Step-by-step solutions for mathematical, physics, and aptitude questions.</div>
                    </div>
                    <div class="feature-tag-wrapper">
                        <span class="feature-tag tag-indigo">MATH & STEM</span>
                    </div>
                </div>
                <div class="feature-card-item card-notes">
                    <div>
                        <div class="feature-icon-box">📝</div>
                        <div class="feature-title">Read Notes</div>
                        <div class="feature-desc">Understand handwritten scribbles, blackboard photos, and textbook notes.</div>
                    </div>
                    <div class="feature-tag-wrapper">
                        <span class="feature-tag tag-violet">HANDWRITING</span>
                    </div>
                </div>
                <div class="feature-card-item card-code">
                    <div>
                        <div class="feature-icon-box">💻</div>
                        <div class="feature-title">Explain Code</div>
                        <div class="feature-desc">Analyze programming code snippets, syntax errors, and algorithm logic.</div>
                    </div>
                    <div class="feature-tag-wrapper">
                        <span class="feature-tag tag-cyan">CODE & LOGIC</span>
                    </div>
                </div>
                <div class="feature-card-item card-diagrams">
                    <div>
                        <div class="feature-icon-box">📐</div>
                        <div class="feature-title">Understand Diagrams</div>
                        <div class="feature-desc">Explain technical architectures, scientific diagrams, and flowcharts.</div>
                    </div>
                    <div class="feature-tag-wrapper">
                        <span class="feature-tag tag-coral">DIAGRAMS</span>
                    </div>
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
        col_material, col_solution = st.columns([42, 58], gap="large")

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
            <div class="chat-section">
                <div class="chat-header">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <h4 style="font-size: 16.5px; font-weight: 700; color: #172033; margin: 0; display: flex; align-items: center; gap: 8px;">
                            <span style="color: #7C3AED;">💬</span> Interactive Tutoring & Q&A
                        </h4>
                        <span style="font-size: 12px; color: #64748B;">Ask questions, request simpler explanations, or get more practice</span>
                    </div>
                </div>
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
            <div class="chat-section">
                <div class="composer-meta-wrapper">
                    <div class="composer-title-group">
                        <div class="composer-title">💬 Ask SnapStudy</div>
                        <div class="composer-subtitle">Ask questions, request explanations, or generate practice.</div>
                    </div>
                    <div class="composer-attached-pill">
                        📎 Study material attached: {st.session_state.uploaded_image_name or 'Uploaded File'}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggestion Chips above the input
        st.markdown('<div class="composer-chips-container">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Empty State: subtle hint when no material is uploaded
        st.markdown(
            """
            <div class="chat-section">
                <div class="composer-meta-wrapper">
                    <div class="composer-title-group">
                        <div class="composer-title">💬 Ask SnapStudy</div>
                        <div class="composer-subtitle">Ask questions, request explanations, or generate practice.</div>
                    </div>
                    <div class="composer-empty-pill">
                        💡 Upload study material to ask questions about it
                    </div>
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



