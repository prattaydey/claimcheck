"""
Per-paragraph detailed analysis and clause-like breakdown.
Provides granular risk assessment for each paragraph of the user's invention.
"""

import json
import re
from typing import Optional
import google.genai as genai
from app.config import settings

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


PARAGRAPH_ANALYSIS_PROMPT = """Analyze a paragraph from an invention description and its matched prior art.
Provide a detailed clause-like breakdown (like patent claims) that breaks down:
1. Technical components described
2. Functional relationships between components
3. Specific claim-like elements that could be patentable
4. Vulnerability assessment against prior art

Return JSON:
{
    "paragraph_id": "string",
    "technical_components": ["component1", "component2"],
    "functional_flow": "string - describe how components interact",
    "claim_elements": [
        {
            "element": "string - patentable element",
            "description": "string - specific wording",
            "strength": "Strong | Moderate | Weak"
        }
    ],
    "prior_art_exposure": {
        "most_relevant_patent": "string",
        "overlap_level": "None | Low | Medium | High | Critical",
        "vulnerable_elements": ["element1", "element2"],
        "distinguishing_features": ["feature1", "feature2"]
    },
    "risk_score": "integer 1-100",
    "mitigation": "string - how to reduce exposure"
}"""


def analyze_paragraph(paragraph: dict, prior_art_matches: list[dict]) -> dict:
    """
    Analyze a single paragraph against its prior art matches.

    Args:
        paragraph: {paragraph_id, text}
        prior_art_matches: List of prior art results for this paragraph

    Returns:
        Detailed clause-like breakdown with risk assessment
    """
    if not prior_art_matches:
        return {
            "paragraph_id": paragraph.get("paragraph_id", ""),
            "technical_components": [],
            "functional_flow": "No prior art matches found",
            "claim_elements": [],
            "prior_art_exposure": {
                "most_relevant_patent": None,
                "overlap_level": "None",
                "vulnerable_elements": [],
                "distinguishing_features": [],
            },
            "risk_score": 0,
            "mitigation": "No immediate prior art conflicts identified",
        }

    # Build context with top 3 prior art matches
    prior_art_context = "\n".join(
        [
            f"- {art.get('patent_id', 'Unknown')}: {art.get('title', '')} "
            f"(similarity: {art.get('similarity_score', 0):.0%})\n"
            f"  Excerpt: {art.get('text', '')[:200]}..."
            for art in prior_art_matches[:3]
        ]
    )

    user_message = (
        f"Paragraph ID: {paragraph.get('paragraph_id', '')}\n\n"
        f"Paragraph Text:\n{paragraph.get('text', '')}\n\n"
        f"Matched Prior Art:\n{prior_art_context}\n\n"
        f"Provide detailed clause-like breakdown and risk assessment."
    )

    client = _get_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            {"role": "user", "parts": [{"text": PARAGRAPH_ANALYSIS_PROMPT + "\n\n" + user_message}]}
        ],
        config={"temperature": 0.3},
    )

    raw = response.text or "{}"

    # Extract JSON from markdown code blocks if present
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if match:
        raw = match.group(1)

    try:
        result = json.loads(raw)
        result["paragraph_id"] = paragraph.get("paragraph_id", "")
        return result
    except json.JSONDecodeError:
        return {
            "paragraph_id": paragraph.get("paragraph_id", ""),
            "technical_components": [],
            "functional_flow": "Analysis unavailable",
            "claim_elements": [],
            "prior_art_exposure": {
                "most_relevant_patent": None,
                "overlap_level": "Unknown",
                "vulnerable_elements": [],
                "distinguishing_features": [],
            },
            "risk_score": 50,
            "mitigation": "Manual review recommended",
        }


def analyze_paragraphs_batch(paragraphs: list[dict], prior_art_by_paragraph: dict) -> list[dict]:
    """
    Analyze multiple paragraphs in sequence.

    Args:
        paragraphs: List of {paragraph_id, text}
        prior_art_by_paragraph: Dict mapping paragraph_id -> list of prior art results

    Returns:
        List of detailed analyses, one per paragraph
    """
    analyses = []
    for para in paragraphs:
        para_id = para.get("paragraph_id", "")
        prior_arts = prior_art_by_paragraph.get(para_id, [])
        analysis = analyze_paragraph(para, prior_arts)
        analyses.append(analysis)

    return analyses


def calculate_paragraph_risk_profile(analyses: list[dict]) -> dict:
    """
    Calculate overall risk profile from paragraph analyses.

    Returns:
        {
            "avg_risk_score": float,
            "highest_risk_paragraph": dict,
            "critical_exposure_count": int,
            "risk_distribution": "string - description of risk spread"
        }
    """
    if not analyses:
        return {
            "avg_risk_score": 0,
            "highest_risk_paragraph": None,
            "critical_exposure_count": 0,
            "risk_distribution": "No paragraphs analyzed",
        }

    risk_scores = [a.get("risk_score", 0) for a in analyses]
    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0

    highest_risk = max(analyses, key=lambda x: x.get("risk_score", 0)) if analyses else None

    critical_count = len([a for a in analyses if a.get("prior_art_exposure", {}).get("overlap_level") == "Critical"])

    if avg_risk < 25:
        distribution = "Low overall risk, well-differentiated from prior art"
    elif avg_risk < 50:
        distribution = "Moderate risk, some overlaps with prior art in specific areas"
    elif avg_risk < 75:
        distribution = "High risk, significant overlaps across multiple paragraphs"
    else:
        distribution = "Critical risk, pervasive prior art conflicts"

    return {
        "avg_risk_score": round(avg_risk, 1),
        "highest_risk_paragraph": highest_risk,
        "critical_exposure_count": critical_count,
        "risk_distribution": distribution,
    }
