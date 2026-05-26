import json
from typing import Optional

import google.genai as genai
from app.config import settings

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


REPORT_SCHEMA = {
    "overall_risk_score": "integer 1-100",
    "critical_conflicts": [
        {
            "patent_id": "string",
            "patent_title": "string",
            "conflicting_claim": "string — excerpt from patent claim",
            "user_text_excerpt": "string — excerpt from user document",
            "conflict_explanation": "string",
            "severity": "High | Medium | Low",
        }
    ],
    "design_workaround_suggestions": ["string"],
    "summary": "string — 2-3 sentence executive summary",
}

SYSTEM_PROMPT = f"""You are a patent attorney AI assistant specializing in prior art analysis and infringement risk assessment.

Analyze the user's invention description against the retrieved prior art and produce a structured JSON report.
Your response MUST be valid JSON exactly matching this schema:
{json.dumps(REPORT_SCHEMA, indent=2)}

Scoring guidance for overall_risk_score:
- 1-30: Low risk — few or superficial overlaps with prior art
- 31-60: Moderate risk — notable similarities but meaningful differences exist
- 61-85: High risk — significant claim overlaps requiring design changes
- 86-100: Critical risk — near-identical claims found; strong infringement likelihood

Be precise, cite specific claim language, and make workaround suggestions actionable."""


def generate_report(user_text: str, prior_art_hits: list[dict]) -> dict:
    """Call Gemini to produce an infringement risk report in structured JSON."""
    context_blocks = []
    for i, hit in enumerate(prior_art_hits, 1):
        context_blocks.append(
            f"[Patent {i}: {hit['patent_id']} — {hit['title']}]\n"
            f"Section: {hit['section']} | Claim: {hit.get('claim_number', 'N/A')}\n"
            f"Similarity: {hit['similarity_score']:.2%}\n"
            f"Text: {hit['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    user_message = (
        f"## User's Invention Description\n\n{user_text}\n\n"
        f"## Retrieved Prior Art\n\n{context}\n\n"
        "Produce the infringement risk report JSON now."
    )

    client = _get_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_message}]}
        ],
        config={"temperature": 0.2},
    )

    raw = response.text or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse model response", "raw": raw}
