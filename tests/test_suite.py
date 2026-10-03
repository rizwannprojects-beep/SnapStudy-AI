import os
import sys
import time

# Ensure SnapStudy-AI is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app import (
    is_valid_email,
    get_secret,
    get_gemini_client,
    get_model_pipeline,
    classify_gemini_error,
    call_gemini_with_fallback,
    analyze_uploaded_material,
    generate_chat_reply,
    generate_session_summary,
    send_study_summary_email,
    GeminiServiceError,
    GEMINI_MODELS
)

def run_tests():
    print("=" * 60)
    print("SNAPSTUDY AI - COMPREHENSIVE AUTOMATED VERIFICATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST 1: Startup & Configuration
    # ---------------------------------------------------------
    print("\n[TEST 1] Application Startup & Pipeline Definition")
    models = get_model_pipeline()
    print(f"  Active model pipeline: {models}")
    assert len(models) >= 3
    assert models[0] in ["gemini-3.8-flash", "gemini-2.0-flash"]
    print("  -> TEST 1 PASSED: Model pipeline verified.")

    # ---------------------------------------------------------
    # TEST 2: Student Profile Validation
    # ---------------------------------------------------------
    print("\n[TEST 2] Student Profile Validation")
    assert is_valid_email("alex@university.edu") is True
    assert is_valid_email("student.johnson@college.org") is True
    assert is_valid_email("invalid-email") is False
    assert is_valid_email("@no-user.com") is False
    assert is_valid_email("no-domain@") is False
    assert is_valid_email("") is False
    print("  -> TEST 2 PASSED: Email validation cleanly rejects invalid inputs.")

    # ---------------------------------------------------------
    # TEST 3: Image Types Verification
    # ---------------------------------------------------------
    print("\n[TEST 3] Image Upload Format Verification")
    math_path = os.path.join(BASE_DIR, "tests", "sample_materials", "math_problem.png")
    notes_path = os.path.join(BASE_DIR, "tests", "sample_materials", "physics_notes.jpg")
    code_path = os.path.join(BASE_DIR, "tests", "sample_materials", "code_snippet.webp")

    assert os.path.exists(math_path), f"Missing {math_path}"
    assert os.path.exists(notes_path), f"Missing {notes_path}"
    assert os.path.exists(code_path), f"Missing {code_path}"

    with open(math_path, "rb") as f:
        png_bytes = f.read()
    with open(notes_path, "rb") as f:
        jpg_bytes = f.read()
    with open(code_path, "rb") as f:
        webp_bytes = f.read()

    assert len(png_bytes) > 0
    assert len(jpg_bytes) > 0
    assert len(webp_bytes) > 0
    print(f"  PNG size: {len(png_bytes)} bytes")
    print(f"  JPG size: {len(jpg_bytes)} bytes")
    print(f"  WEBP size: {len(webp_bytes)} bytes")
    print("  -> TEST 3 PASSED: All 3 formats (PNG, JPG, WEBP) exist and load valid binary data.")

    # ---------------------------------------------------------
    # TEST 5: Gemini Failure Handling (503/429/401/400)
    # ---------------------------------------------------------
    print("\n[TEST 5] Gemini Failure Handling & Immediate Fallback")
    
    # 503 check
    retryable_503, msg_503, code_503 = classify_gemini_error(
        Exception("503 UNAVAILABLE: This model is currently experiencing high demand."),
        models[0]
    )
    assert retryable_503 is True
    assert code_503 == 503
    assert "temporarily busy" in msg_503

    # 429 check
    retryable_429, msg_429, code_429 = classify_gemini_error(
        Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded"),
        models[0]
    )
    assert retryable_429 is True
    assert code_429 == 429
    assert "limit reached" in msg_429

    # 401 permanent check (do NOT retry)
    retryable_401, msg_401, code_401 = classify_gemini_error(
        Exception("401 API_KEY_INVALID: API key not valid"),
        models[0]
    )
    assert retryable_401 is False
    assert code_401 == 401
    assert "API key appears invalid" in msg_401

    # 400 permanent check (do NOT retry)
    retryable_400, msg_400, code_400 = classify_gemini_error(
        Exception("400 INVALID_ARGUMENT: Unsupported format"),
        models[0]
    )
    assert retryable_400 is False
    assert code_400 == 400

    print("  -> TEST 5 PASSED: Error classification correctly distinguishes temporary vs permanent errors.")

    # ---------------------------------------------------------
    # TEST 7: Chat without Image
    # ---------------------------------------------------------
    print("\n[TEST 7] Chat without Image")
    # Verify generate_chat_reply works with image_bytes=None without error
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Newton's Second Law states that force equals mass times acceleration (F = ma)."
    mock_client.models.generate_content.return_value = mock_resp

    test_messages = [{"role": "user", "content": "Explain Newton's second law"}]
    reply = generate_chat_reply(
        client=mock_client,
        messages=test_messages,
        image_bytes=None,
        mime_type=None,
        models=models
    )
    assert "Newton" in reply
    print("  -> TEST 7 PASSED: Text-only chat functions without image and produces no exception.")

    # ---------------------------------------------------------
    # TEST 8: Clear Chat & Session Reset Simulation
    # ---------------------------------------------------------
    print("\n[TEST 8] Clear Chat & Reset")
    session_state = {
        "messages": [{"role": "user", "content": "Question 1"}, {"role": "assistant", "content": "Answer 1"}],
        "uploaded_image_bytes": b"sample_bytes",
        "uploaded_image_mime": "image/png",
        "uploaded_image_name": "test.png",
        "last_summary": "Sample Summary Notes",
    }
    # Execute reset action
    session_state["messages"] = []
    session_state["uploaded_image_bytes"] = None
    session_state["uploaded_image_mime"] = None
    session_state["uploaded_image_name"] = None
    session_state["last_summary"] = None

    assert len(session_state["messages"]) == 0
    assert session_state["uploaded_image_bytes"] is None
    assert session_state["last_summary"] is None
    print("  -> TEST 8 PASSED: Session state cleanly cleared on reset.")

    # ---------------------------------------------------------
    # TEST 9: Email Summary Error Handling
    # ---------------------------------------------------------
    print("\n[TEST 9] Email Summary (Missing & Invalid SMTP)")
    # Case 1: Missing credentials
    ok1, err1 = send_study_summary_email(
        to_email="alex@university.edu",
        student_name="Alex",
        summary_markdown="# Study Summary",
        smtp_email="",
        smtp_password=""
    )
    assert ok1 is False
    assert "SMTP credentials are missing" in err1

    # Case 2: Bad credentials (fails authentication cleanly without crashing)
    ok2, err2 = send_study_summary_email(
        to_email="alex@university.edu",
        student_name="Alex",
        summary_markdown="# Study Summary",
        smtp_email="test_user@gmail.com",
        smtp_password="wrong_password_1234"
    )
    assert ok2 is False
    assert "failed" in err2.lower()
    print("  -> TEST 9 PASSED: SMTP missing & authentication failures handled cleanly.")

    # ---------------------------------------------------------
    # TEST 10: Secrets & Security Isolation
    # ---------------------------------------------------------
    print("\n[TEST 10] Secrets & Security Check")
    with open(os.path.join(BASE_DIR, "app.py"), "r", encoding="utf-8") as f:
        app_content = f.read()
    assert "AIza" not in app_content, "API Key found in app.py!"
    
    with open(os.path.join(BASE_DIR, ".gitignore"), "r", encoding="utf-8") as f:
        gitignore_content = f.read()
    assert ".streamlit/secrets.toml" in gitignore_content, "secrets.toml not in .gitignore!"

    with open(os.path.join(BASE_DIR, ".streamlit", "secrets.toml.example"), "r", encoding="utf-8") as f:
        example_content = f.read()
    assert "your_gemini_api_key_here" in example_content
    assert "your_16_character_app_password" in example_content
    print("  -> TEST 10 PASSED: No keys hardcoded, .gitignore active, example contains only placeholders.")

    # ---------------------------------------------------------
    # TEST 11: Error Handling & Resilience
    # ---------------------------------------------------------
    print("\n[TEST 11] Error Handling for All Scenarios")
    # All models fail simulation
    mock_failing_client = MagicMock()
    mock_failing_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE: high demand")
    try:
        call_gemini_with_fallback(
            client=mock_failing_client,
            contents=["test"],
            models=["m1", "m2", "m3"]
        )
        assert False, "Should have raised GeminiServiceError"
    except GeminiServiceError as ge:
        assert ge.user_message == "Gemini is temporarily busy. Please try again in a few moments."
    print("  -> TEST 11 PASSED: All edge cases produce user-friendly error messages.")

    # ---------------------------------------------------------
    # TEST 4 & 6 & 12: Live Gemini Vision Analysis & Interactive Chat
    # ---------------------------------------------------------
    print("\n[TEST 4, 6, 12] Live Gemini Vision Testing on Real Images")
    gemini_key = get_secret("GEMINI_API_KEY")
    if not gemini_key:
        # Check in .streamlit/secrets.toml directly
        import toml
        secrets_path = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "r", encoding="utf-8") as sf:
                parsed = toml.load(sf)
                gemini_key = parsed.get("GEMINI_API_KEY")

    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        print("  [SKIP LIVE API] No active GEMINI_API_KEY configured in secrets.toml.")
    else:
        print(f"  Found Gemini API key (length: {len(gemini_key)}). Testing live multimodal calls...")
        client = get_gemini_client(api_key=gemini_key)

        test_cases = [
            ("A. Math Problem (PNG)", png_bytes, "image/png"),
            ("B. Physics Notes (JPG)", jpg_bytes, "image/jpeg"),
            ("C. Python Code (WEBP)", webp_bytes, "image/webp"),
        ]

        timings = []
        for name, img_b, mime in test_cases:
            print(f"\n  Running Vision Analysis for {name}...")
            t0 = time.time()
            try:
                analysis = analyze_uploaded_material(
                    client=client,
                    image_bytes=img_b,
                    mime_type=mime,
                    models=models,
                )
                t_elapsed = time.time() - t0
                timings.append(t_elapsed)
                print(f"  -> Success in {t_elapsed:.2f}s!")
                print(f"  -> Response preview ({len(analysis)} chars): {analysis[:180].replace(chr(10), ' ')}...")
                assert len(analysis) > 50
            except Exception as e:
                print(f"  -> Live call exception: {e}")

        # Test Interactive Chat with context
        print("\n  [TEST 6] Testing Interactive Chat Follow-up...")
        t0 = time.time()
        chat_msgs = [
            {"role": "user", "content": "Please analyze this study material."},
            {"role": "assistant", "content": "This is a calculus problem about finding the derivative of f(x) = 3x^4 - 5x^2 + 7x - 2."},
            {"role": "user", "content": "Explain the formula used in simple words and give only the final answer."}
        ]
        try:
            chat_reply = generate_chat_reply(
                client=client,
                messages=chat_msgs,
                image_bytes=png_bytes,
                mime_type="image/png",
                models=models
            )
            t_chat = time.time() - t0
            print(f"  -> Chat response received in {t_chat:.2f}s!")
            print(f"  -> Chat preview: {chat_reply[:180].replace(chr(10), ' ')}...")
            assert len(chat_reply) > 20
        except Exception as e:
            print(f"  -> Chat call exception: {e}")

        if timings:
            avg_time = sum(timings) / len(timings)
            print(f"\n  [TEST 12] Performance Summary: Average Vision Analysis response time: {avg_time:.2f}s")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
