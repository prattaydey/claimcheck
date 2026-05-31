"""
Phase 3: Agent Synthesis
Uses Gemini to analyze prior art patterns across multiple paragraphs,
identify conflict clusters, and generate actionable workaround suggestions.
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


PATTERN_DETECTION_PROMPT = """You are a patent analysis AI expert. Analyze the provided invention paragraphs and their retrieved prior art to identify:

1. **Cross-paragraph patterns**: What technical themes repeat across multiple paragraphs?
2. **Conflict clusters**: Which patents/claims appear multiple times in different paragraphs' results? (Higher frequency = stronger conflict)
3. **Core exposure areas**: What are the 2-3 most vulnerable aspects of the invention?
4. **Workaround feasibility**: For each exposure, can the invention be redesigned to avoid it?

Input format:
- List of paragraphs with their IDs and matched prior art
- Each prior art match includes similarity score and patent details

Return a JSON object with this structure:
{
    "cross_paragraph_themes": [
        {
            "theme": "string - technical concept appearing across paragraphs",
            "paragraphs_affected": ["para_id1", "para_id2"],
            "frequency": "integer - how many paragraphs mention this",
            "severity": "Low | Medium | High"
        }
    ],
    "conflict_clusters": [
        {
            "patent_id": "string",
            "patent_title": "string",
            "frequency": "integer - how many paragraphs matched this patent",
            "average_similarity": "float - 0.0-1.0",
            "affected_paragraphs": ["para_id1", "para_id2"],
            "core_conflict": "string - what specific aspect conflicts"
        }
    ],
    "core_exposures": [
        {
            "rank": 1,
            "exposure": "string - describe the vulnerability",
            "paragraphs_involved": ["para_id1", "para_id2"],
            "severity": "Critical | High | Medium",
            "redesign_feasible": true/false,
            "rationale": "string - why it is/isn't redesignable",
            "workaround_options": ["option1", "option2"]
        }
    ],
    "narrative_summary": "string - 3-4 sentence executive summary of the invention's exposure landscape"
}"""


def analyze_patterns(paragraphs: list[dict], prior_art_by_paragraph: dict) -> dict:
    """
    Analyze patterns across paragraphs and their prior art matches.

    Args:
        paragraphs: List of {paragraph_id, text}
        prior_art_by_paragraph: Dict mapping paragraph_id -> list of prior art results

    Returns:
        {
            "cross_paragraph_themes": [...],
            "conflict_clusters": [...],
            "core_exposures": [...],
            "narrative_summary": "..."
        }
    """
    # Build analysis context
    context_blocks = []
    for para in paragraphs:
        para_id = para.get("paragraph_id", "")
        para_text = para.get("text", "")[:300]  # Truncate for context
        prior_arts = prior_art_by_paragraph.get(para_id, [])

        context_blocks.append(
            f"[Paragraph {para_id}]\n"
            f"Excerpt: {para_text}...\n"
            f"Prior Art Matches ({len(prior_arts)} found):\n"
        )

        for art in prior_arts[:3]:  # Include top 3 matches per paragraph
            context_blocks.append(
                f"  - {art.get('patent_id', 'Unknown')}: {art.get('title', '')} "
                f"(similarity: {art.get('similarity_score', 0):.1%})\n"
            )

    context = "\n".join(context_blocks)

    user_message = (
        f"Analyze the following invention paragraphs and their prior art matches:\n\n"
        f"{context}\n\n"
        f"Identify patterns, conflict clusters, and core exposures."
    )

    client = _get_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            {"role": "user", "parts": [{"text": PATTERN_DETECTION_PROMPT + "\n\n" + user_message}]}
        ],
        config={"temperature": 0.3},
    )

    raw = response.text or "{}"

    # Extract JSON from markdown code blocks if present
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if match:
        raw = match.group(1)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "cross_paragraph_themes": [],
            "conflict_clusters": [],
            "core_exposures": [],
            "narrative_summary": "Unable to analyze patterns from prior art results.",
        }


WORKAROUND_PROMPT = """You are a patent strategy expert. Given a prior art conflict, suggest concrete design changes to avoid infringement.

For each core exposure identified, provide 2-3 specific, actionable workarounds that:
1. Are technically feasible
2. Preserve the core value of the invention
3. Don't require complete redesign

Input: Exposure description with patent claims and affected paragraphs

Return a JSON object with this structure:
{
    "exposure_summary": "string - restate the exposure for clarity",
    "redesign_scope": "Minor | Moderate | Substantial",
    "workarounds": [
        {
            "number": 1,
            "title": "string - short title of workaround",
            "description": "string - 2-3 sentences describing the change",
            "technical_changes": ["change1", "change2"],
            "impact": "string - how this affects the invention",
            "effort": "Low | Medium | High",
            "risk": "Low | Medium | High"
        }
    ],
    "recommended_path": "string - which workaround(s) to pursue and in what order"
}"""


def suggest_workarounds(exposure: str, patent_claims: list[str]) -> dict:
    """
    Generate workaround suggestions for a specific exposure.

    Args:
        exposure: Description of the vulnerability
        patent_claims: List of conflicting patent claims

    Returns:
        {
            "exposure_summary": "...",
            "redesign_scope": "...",
            "workarounds": [...],
            "recommended_path": "..."
        }
    """
    claims_context = "\n".join([f"- {claim}" for claim in patent_claims[:3]])

    user_message = (
        f"Exposure: {exposure}\n\n"
        f"Conflicting Prior Art Claims:\n{claims_context}\n\n"
        f"Suggest concrete workarounds to design around this conflict."
    )

    client = _get_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            {"role": "user", "parts": [{"text": WORKAROUND_PROMPT + "\n\n" + user_message}]}
        ],
        config={"temperature": 0.4},
    )

    raw = response.text or "{}"

    # Extract JSON from markdown code blocks if present
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if match:
        raw = match.group(1)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "exposure_summary": exposure,
            "redesign_scope": "Unknown",
            "workarounds": [],
            "recommended_path": "Unable to generate workaround suggestions.",
        }


def build_narrative(
    invention_summary: str,
    primary_domain: str,
    core_exposures: list[dict],
    conflict_clusters: list[dict],
) -> str:
    """
    Build a narrative summary of the invention's infringement exposure.

    Returns a 4-5 sentence narrative that tells the story of:
    1. What the invention does
    2. What domains it touches
    3. Where it's most exposed
    4. Whether exposure is redesignable
    """
    if not core_exposures or not conflict_clusters:
        return (
            f"This {primary_domain} invention shows limited prior art overlap. "
            "Further analysis recommended with expanded patent databases."
        )

    most_severe = core_exposures[0] if core_exposures else {}
    most_frequent = conflict_clusters[0] if conflict_clusters else {}

    narrative = (
        f"This {primary_domain} invention is most exposed through {most_severe.get('exposure', 'its core technical approach')}. "
        f"The patent {most_frequent.get('patent_id', '')} appears as the strongest prior art conflict, "
        f"matching {most_frequent.get('frequency', 0)} distinct aspects of the invention. "
    )

    if most_severe.get("redesign_feasible"):
        narrative += (
            f"The exposure appears {most_severe.get('severity', 'moderate').lower()}, "
            f"with feasible redesign options available to mitigate conflict."
        )
    else:
        narrative += (
            f"However, the core exposure is {most_severe.get('severity', 'moderate').lower()}, "
            f"and may require significant redesign to fully avoid infringement."
        )

    return narrative
