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
            "Your Gemini API key appears invalid or expired. Please check GEMINI_API_KEY in .streamlit/secrets.toml.",
            401,
        )

    # 2. 403 Forbidden / Permission Denied
    if code == 403 or "permission_denied" in err_str:
        return (
            False,
            "Your Gemini API key or permissions need to be checked. Ensure the Gemini API is enabled for your project.",
            403,
        )

    # 3. 404 Model Not Found
    if code == 404 or "not_found" in err_str or ("model" in err_str and "not found" in err_str):
        return (
            False,
            f"The configured Gemini model '{model_name}' is unavailable or not found. Please check GEMINI_MODEL.",
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
            "Gemini is temporarily busy. Please try again in a few moments.",
            503,
        )

    # 6. 429 Rate Limit / Resource Exhausted
    if code == 429 or "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str or "quota" in err_str:
        return (
            True,
            "Gemini request limit reached temporarily. Please wait a little and try again.",
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
        "Gemini is temporarily unavailable. Please wait a few seconds and try again.",
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
# 4. Streamlit Application Interface
# ==============================================================================

def main():
    st.set_page_config(
        page_title="SnapStudy AI - Visual Study Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load configuration
    secret_gemini_key = get_secret("GEMINI_API_KEY", "")
    active_models = get_model_pipeline()
    primary_model = active_models[0]
    smtp_email = get_secret("SMTP_EMAIL", "")
    smtp_password = get_secret("SMTP_APP_PASSWORD", "")
    smtp_server = get_secret("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(get_secret("SMTP_PORT", "587"))

    # Session State Initialization
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

    # Sidebar: Branding, Profile & Settings
    with st.sidebar:
        st.title("📚 SnapStudy AI")
        st.caption("Visual Learning & Study Assistant powered by Google Gemini Vision")

        st.divider()

        # API Key handling: Secret or manual fallback
        api_key = secret_gemini_key
        if not api_key:
            st.warning("⚠️ `GEMINI_API_KEY` not found in `.streamlit/secrets.toml`.")
            api_key = st.text_input(
                "Enter Gemini API Key (temporary):",
                type="password",
                help="Obtain one from https://aistudio.google.com/ or configure .streamlit/secrets.toml",
            )

        # Student Profile Card
        st.subheader("👤 Student Profile")
        if st.session_state.onboarding_complete:
            st.markdown(f"**Name:** {st.session_state.student_name}")
            st.markdown(f"**Email:** `{st.session_state.student_email}`")
            if st.button("✏️ Edit Profile", use_container_width=True):
                st.session_state.onboarding_complete = False
                st.rerun()
        else:
            with st.form("onboarding_form"):
                name_input = st.text_input(
                    "Your Full Name", value=st.session_state.student_name, placeholder="e.g. Alex Johnson"
                )
                email_input = st.text_input(
                    "Your Email Address",
                    value=st.session_state.student_email,
                    placeholder="e.g. alex@university.edu",
                    help="Summaries will be sent to this email address.",
                )
                submitted = st.form_submit_button("Save Profile", use_container_width=True)
                if submitted:
                    if not name_input.strip():
                        st.error("Please provide your name.")
                    elif not is_valid_email(email_input):
                        st.error("Please provide a valid email address.")
                    else:
                        st.session_state.student_name = name_input.strip()
                        st.session_state.student_email = email_input.strip()
                        st.session_state.onboarding_complete = True
                        st.success("Profile saved!")
                        st.rerun()

        st.divider()

        # Image Upload Widget
        st.subheader("📷 Study Material")
        uploaded_file = st.file_uploader(
            "Upload question, notes, code, or diagram",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload textbook problems, handwritten notes, diagrams, or programming code.",
        )

        if uploaded_file is not None:
            # Check if this is a newly uploaded file
            if st.session_state.uploaded_image_name != uploaded_file.name:
                st.session_state.uploaded_image_bytes = uploaded_file.getvalue()
                st.session_state.uploaded_image_mime = uploaded_file.type
                st.session_state.uploaded_image_name = uploaded_file.name
                st.session_state.messages = []  # Reset chat for the new study item
                st.session_state.last_summary = None

            try:
                st.image(
                    st.session_state.uploaded_image_bytes,
                    caption=f"Uploaded: {uploaded_file.name}",
                    width="stretch",
                )
            except TypeError:
                st.image(
                    st.session_state.uploaded_image_bytes,
                    caption=f"Uploaded: {uploaded_file.name}",
                    use_container_width=True,
                )

        st.divider()

        # Session Actions
        st.subheader("⚙️ Actions")

        # Send Summary to Email Button
        can_send_summary = (
            st.session_state.onboarding_complete
            and len(st.session_state.messages) > 0
            and bool(api_key)
        )

        if st.button(
            "📧 Send Summary to Email",
            disabled=not can_send_summary,
            use_container_width=True,
            help="Generates high-yield study revision notes and emails them to your registered email.",
        ):
            if not smtp_email or not smtp_password:
                st.error(
                    "SMTP credentials are not configured in `.streamlit/secrets.toml`. "
                    "Please provide `SMTP_EMAIL` and `SMTP_APP_PASSWORD`."
                )
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
                    except GeminiServiceError as e:
                        summary_status.empty()
                        st.error(e.user_message)
                    except Exception as e:
                        summary_status.empty()
                        _, friendly_msg, _ = classify_gemini_error(e, primary_model)
                        st.error(friendly_msg)

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

        # Clear / Reset Session
        if st.button("🔄 Clear Chat & Reset", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_image_bytes = None
            st.session_state.uploaded_image_mime = None
            st.session_state.uploaded_image_name = None
            st.session_state.last_summary = None
            st.rerun()

        # System info footer
        st.divider()
        st.caption(f"🧠 Primary Model: `{primary_model}`")
        if len(active_models) > 1:
            st.caption(f"🛡️ Fallbacks: {', '.join(f'`{m}`' for m in active_models[1:])}")
        if smtp_email:
            st.caption(f"✉️ SMTP Sender: `{smtp_email}`")
        else:
            st.caption("✉️ SMTP: *Not configured*")

    # ==========================================================================
    # Main Chat & Study Area
    # ==========================================================================

    st.header("SnapStudy AI 💡 Visual Study Assistant", divider="rainbow")

    # Onboarding Prompt if not yet filled
    if not st.session_state.onboarding_complete:
        st.info("👋 **Welcome to SnapStudy AI!** Please enter your name and email in the sidebar to get started.")

    # Missing API Key Warning Banner
    if not api_key:
        st.error(
            "🔑 **Gemini API Key Required**\n\n"
            "Please configure your `GEMINI_API_KEY` in `.streamlit/secrets.toml` "
            "(copy from `.streamlit/secrets.toml.example`) or enter it in the sidebar to begin."
        )

    # Initial Analysis Trigger if image uploaded but no messages yet
    if st.session_state.uploaded_image_bytes and len(st.session_state.messages) == 0 and api_key:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                "📸 **Study material uploaded!** Click **Analyze Material** to have Gemini inspect the problem, "
                "transcribe notes, explain concepts, and provide a step-by-step breakdown."
            )
        with col2:
            if st.button("🚀 Analyze Material", type="primary", use_container_width=True):
                status_placeholder = st.empty()
                with st.spinner("Gemini Vision is analyzing your study material..."):
                    try:
                        client = get_gemini_client(api_key=api_key)
                        analysis = analyze_uploaded_material(
                            client=client,
                            image_bytes=st.session_state.uploaded_image_bytes,
                            mime_type=st.session_state.uploaded_image_mime,
                            models=active_models,
                            status_callback=lambda msg: status_placeholder.info(f"⏳ {msg}"),
                        )
                        status_placeholder.empty()
                        st.session_state.messages.append(
                            {"role": "user", "content": "Please analyze this study material and explain it step-by-step."}
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": analysis}
                        )
                        st.rerun()
                    except GeminiServiceError as e:
                        status_placeholder.empty()
                        st.error(e.user_message)
                    except Exception as e:
                        status_placeholder.empty()
                        _, friendly_msg, _ = classify_gemini_error(e, primary_model)
                        st.error(friendly_msg)

    # Empty State Guidance
    if not st.session_state.uploaded_image_bytes and len(st.session_state.messages) == 0:
        st.markdown(
            """
            ### How to use SnapStudy AI:
            1. **Set your profile** in the sidebar (Name and Email for study notes).
            2. **Upload an image** of:
               - 📖 Textbook exercises or exam questions
               - ✍️ Handwritten notes or whiteboard scribbles
               - ➗ Mathematical formulas or calculus problems
               - 💻 Programming code snippets or terminal errors
               - 🔬 Technical, scientific, or architectural diagrams
            3. **Chat interactively** with Gemini to clarify doubts, request hints, or break down difficult concepts.
            4. **Click 'Send Summary to Email'** to receive a structured revision sheet in your inbox.
            """
        )

    # Display Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box
    user_prompt = st.chat_input(
        "Ask a question about the uploaded material, request a simpler explanation, or ask for practice...",
        disabled=not bool(api_key),
    )

    if user_prompt and user_prompt.strip():
        if not st.session_state.onboarding_complete:
            st.warning("Please complete your profile (name & email) in the sidebar first.")
        else:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Generate Gemini response
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        client = get_gemini_client(api_key=api_key)
                        reply = generate_chat_reply(
                            client=client,
                            messages=st.session_state.messages,
                            image_bytes=st.session_state.uploaded_image_bytes,
                            mime_type=st.session_state.uploaded_image_mime,
                            models=active_models,
                            status_callback=lambda msg: status_placeholder.info(f"⏳ {msg}"),
                        )
                        status_placeholder.empty()
                        st.markdown(reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply}
                        )
                    except GeminiServiceError as e:
                        status_placeholder.empty()
                        st.error(e.user_message)
                    except Exception as e:
                        status_placeholder.empty()
                        _, friendly_msg, _ = classify_gemini_error(e, primary_model)
                        st.error(friendly_msg)

    # Show preview of last generated summary if available
    if st.session_state.last_summary:
        with st.expander("📄 View Last Generated Study Summary"):
            st.markdown(st.session_state.last_summary)


if __name__ == "__main__":
    main()
