"""
Phase 1: Extract & Classify
Extracts technical terms and classifies document into tech domains.
Used to pre-filter noise and enable domain-specific RAG searches.
"""

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


# Domain taxonomy with keywords for pre-classification
DOMAIN_KEYWORDS = {
    "software": [
        "algorithm", "software", "code", "program", "application", "API", "framework",
        "database", "query", "processing", "computation", "memory", "CPU", "GPU",
        "machine learning", "neural network", "deep learning", "classifier", "model",
        "encryption", "hash", "compression", "parser", "compiler", "interpreter"
    ],
    "hardware": [
        "circuit", "chip", "semiconductor", "transistor", "PCB", "microprocessor",
        "sensor", "actuator", "motor", "device", "apparatus", "mechanism", "mechanical",
        "electrical", "power", "voltage", "current", "resistance", "capacitor",
        "integrated circuit", "IC", "FPGA", "ASIC"
    ],
    "biotech": [
        "protein", "DNA", "RNA", "gene", "genetic", "cell", "biological", "enzyme",
        "pharmaceutical", "drug", "vaccine", "antibody", "genomic", "genome",
        "mutation", "sequencing", "CRISPR", "immunotherapy", "molecular", "compound"
    ],
    "mechanical": [
        "gear", "bearing", "hinge", "lever", "pulley", "spring", "valve", "pump",
        "turbine", "compressor", "combustion", "engine", "transmission", "shaft",
        "lubrication", "friction", "wear", "fatigue"
    ],
    "business_method": [
        "method", "process", "workflow", "transaction", "payment", "contract",
        "protocol", "negotiation", "settlement", "order", "inventory", "supply chain"
    ],
    "network_telecom": [
        "network", "wireless", "cellular", "5G", "6G", "WiFi", "Bluetooth", "protocol",
        "transmission", "signal", "modulation", "antenna", "router", "switch",
        "packet", "routing", "bandwidth", "latency", "throughput"
    ],
    "optical": [
        "optical", "laser", "photon", "wavelength", "fiber optic", "lens", "prism",
        "reflection", "refraction", "imaging", "hologram", "spectroscopy", "polarization"
    ],
}

# Inverse index for quick domain lookup
KEYWORD_TO_DOMAIN = {}
for domain, keywords in DOMAIN_KEYWORDS.items():
    for keyword in keywords:
        KEYWORD_TO_DOMAIN[keyword.lower()] = domain


CLASSIFICATION_PROMPT = """You are a patent domain classifier. Analyze the provided text and:
1. Identify the primary technology domain (software, hardware, biotech, mechanical, business_method, network_telecom, optical, or other)
2. Extract up to 15 key technical terms specific to this invention
3. Identify any secondary domains (if applicable)

Return a JSON object with this structure:
{
    "primary_domain": "string",
    "confidence": 0.0-1.0,
    "secondary_domains": ["string"],
    "key_terms": ["term1", "term2", ...],
    "technical_summary": "1-2 sentence description of the technical core"
}

Be precise and cite specific technical language from the text."""


def classify_document(text: str) -> dict:
    """
    Classify a document into tech domains and extract key technical terms.
    Uses both keyword matching and LLM classification for accuracy.

    Returns:
        {
            "primary_domain": str,
            "confidence": float (0.0-1.0),
            "secondary_domains": list[str],
            "key_terms": list[str],
            "technical_summary": str,
            "extraction_method": str ("keyword" or "llm")
        }
    """
    # Step 1: Quick keyword-based pre-classification
    domain_scores = {}
    text_lower = text.lower()

    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            domain_scores[domain] = matches

    # If strong keyword signal, use it (fast path)
    if domain_scores:
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_domains[0][0]
        confidence = min(sorted_domains[0][1] / 10, 0.95)  # Cap at 0.95 for keyword-only
        secondaries = [d[0] for d in sorted_domains[1:3]]

        # Extract key terms from text
        key_terms = _extract_key_terms(text, primary)

        return {
            "primary_domain": primary,
            "confidence": confidence,
            "secondary_domains": secondaries,
            "key_terms": key_terms,
            "technical_summary": f"Document spans {primary} domain",
            "extraction_method": "keyword"
        }

    # Step 2: Fall back to LLM if no keywords matched
    return _classify_with_llm(text)


def _extract_key_terms(text: str, domain: str) -> list[str]:
    """Extract technical terms specific to the domain."""
    # Look for domain keywords and nearby context
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    found_terms = []
    text_lower = text.lower()

    for keyword in keywords:
        if keyword in text_lower:
            found_terms.append(keyword)

    # Also extract capitalized terms (likely proper nouns/technical terms)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    found_terms.extend(capitalized[:5])  # Add top 5 capitalized terms

    # Remove duplicates, keep order, limit to 15
    seen = set()
    unique_terms = []
    for term in found_terms:
        if term.lower() not in seen:
            unique_terms.append(term)
            seen.add(term.lower())
            if len(unique_terms) >= 15:
                break

    return unique_terms


def _classify_with_llm(text: str) -> dict:
    """Use Gemini to classify when keyword matching fails."""
    client = _get_client()

    # Truncate text if too long (keep first 5000 chars for efficiency)
    text_sample = text[:5000]

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            {"role": "user", "parts": [{"text": CLASSIFICATION_PROMPT + "\n\nText to classify:\n\n" + text_sample}]}
        ],
        config={"temperature": 0.3},
    )

    raw = response.text or "{}"

    # Extract JSON from markdown code blocks if present
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if match:
        raw = match.group(1)

    try:
        import json
        result = json.loads(raw)
        result["extraction_method"] = "llm"
        return result
    except:
        # Fallback if JSON parsing fails
        return {
            "primary_domain": "other",
            "confidence": 0.3,
            "secondary_domains": [],
            "key_terms": [],
            "technical_summary": "Unable to classify",
            "extraction_method": "fallback"
        }


def extract_sections(text: str) -> dict:
    """
    Attempt to extract document sections (abstract, claims, detailed description).
    Useful for user documents that may be structured like patents.

    Returns:
        {
            "abstract": str,
            "claims": str,
            "detailed_description": str,
            "other": str
        }
    """
    sections = {
        "abstract": "",
        "claims": "",
        "detailed_description": "",
        "other": text
    }

    # Simple regex-based section detection
    abstract_match = re.search(
        r'(?:ABSTRACT|Summary|Overview)[\s\n:]*(.+?)(?=(?:CLAIMS|Claims|DESCRIPTION|Detailed Description|$))',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if abstract_match:
        sections["abstract"] = abstract_match.group(1).strip()
        sections["other"] = text.replace(abstract_match.group(0), "")

    claims_match = re.search(
        r'(?:CLAIMS|Claims)[\s\n:]*(.+?)(?=(?:DETAILED DESCRIPTION|Detailed Description|DESCRIPTION|$))',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if claims_match:
        sections["claims"] = claims_match.group(1).strip()
        sections["other"] = sections["other"].replace(claims_match.group(0), "")

    desc_match = re.search(
        r'(?:DETAILED DESCRIPTION|Detailed Description|DESCRIPTION)[\s\n:]*(.+?)$',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if desc_match:
        sections["detailed_description"] = desc_match.group(1).strip()
        sections["other"] = sections["other"].replace(desc_match.group(0), "")

    return sections


def get_domain_description(domain: str) -> str:
    """Get a human-readable description of a tech domain."""
    descriptions = {
        "software": "Software, algorithms, data processing, and machine learning",
        "hardware": "Physical electronics, circuits, semiconductors, and components",
        "biotech": "Biological systems, pharmaceuticals, genetic engineering, and life sciences",
        "mechanical": "Mechanical systems, mechanisms, engines, and physical devices",
        "business_method": "Business processes, workflows, and transaction methods",
        "network_telecom": "Network systems, wireless communication, and telecommunications",
        "optical": "Optical systems, lasers, photonics, and imaging",
    }
    return descriptions.get(domain, "General technology")
