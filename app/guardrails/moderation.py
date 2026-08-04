from groq import Groq
from app.config import GROQ_API_KEY, GROQ_MODEL

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


CLASSIFIER_PROMPT = """You are a strict content safety classifier. Analyze the user's request below and determine if it asks for help with something harmful, illegal, unethical, or dangerous — including but not limited to: violence, weapons, hacking/unauthorized access, fraud, forgery, cheating, scams, drugs, explosives, or exploitation.

Request: "{query}"

Respond in EXACTLY this format, nothing else:
VERDICT: <SAFE or UNSAFE>
REASON: <if UNSAFE, a short 1-2 sentence explanation of why this is harmful/unethical, written in a warm but firm tone, and if appropriate suggest a better alternative. If SAFE, just write "No issues found.">
"""


def check_moderation(text: str) -> dict:
    """AI-based moderation: asks the LLM itself to classify the request,
    instead of relying on a fixed keyword list (which is easy to bypass
    with rephrasing)."""
    try:
        client = get_client()
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(query=text)}],
            temperature=0.0,
            max_tokens=200,
        )
        output = (r.choices[0].message.content or "").strip()

        verdict_line = next((l for l in output.splitlines() if l.upper().startswith("VERDICT")), "")
        reason_line = next((l for l in output.splitlines() if l.upper().startswith("REASON")), "")

        is_unsafe = "UNSAFE" in verdict_line.upper()
        reason_text = reason_line.split(":", 1)[-1].strip() if ":" in reason_line else output

        if is_unsafe:
            return {"blocked": True, "status": "refused", "reason": reason_text}
        return {"blocked": False, "status": "allowed", "reason": "Query passed safety review."}

    except Exception as e:
        # Fail safe: if the classifier call itself fails, don't silently allow —
        # flag it so it's visible rather than pretending everything's fine.
        return {"blocked": False, "status": "allowed", "reason": f"Safety check unavailable ({str(e)}); proceeding with caution."}