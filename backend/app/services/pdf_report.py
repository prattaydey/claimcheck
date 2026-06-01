"""
PDF report generation for ClaimCheck analysis results.
Creates professional, downloadable patent infringement reports.
"""

from datetime import datetime
from typing import Optional, BinaryIO
import io


def generate_pdf_report(
    invention_title: str,
    classification: dict,
    risk_score: int,
    risk_breakdown: dict,
    paragraph_analyses: list[dict],
    conflict_clusters: list[dict],
    core_exposures: list[dict],
    action_items: list[dict],
    patent_timeline: dict,
) -> bytes:
    """
    Generate a professional PDF report.
    Uses a simple text-based approach (can be enhanced with reportlab/weasyprint).

    Returns:
        PDF bytes ready to download
    """
    # For now, generate a structured text report
    # In production, use reportlab or weasyprint for true PDF
    report_text = _build_text_report(
        invention_title,
        classification,
        risk_score,
        risk_breakdown,
        paragraph_analyses,
        conflict_clusters,
        core_exposures,
        action_items,
        patent_timeline,
    )

    return report_text.encode("utf-8")


def _build_text_report(
    invention_title: str,
    classification: dict,
    risk_score: int,
    risk_breakdown: dict,
    paragraph_analyses: list[dict],
    conflict_clusters: list[dict],
    core_exposures: list[dict],
    action_items: list[dict],
    patent_timeline: dict,
) -> str:
    """Build structured text report (foundation for PDF)."""

    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("PATENT INFRINGEMENT RISK ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Metadata
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Invention: {invention_title}")
    lines.append("")

    # Executive Summary
    lines.append("-" * 80)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Domain: {classification.get('primary_domain', 'Unknown').upper()}")
    lines.append(
        f"Confidence: {int(classification.get('confidence', 0) * 100)}%"
    )
    lines.append(
        f"Overall Risk Score: {risk_score}/100 ({_risk_label(risk_score)})"
    )
    lines.append("")
    lines.append(f"Summary: {classification.get('technical_summary', 'N/A')}")
    lines.append("")

    # Risk Breakdown
    if risk_breakdown:
        lines.append("-" * 80)
        lines.append("RISK BREAKDOWN")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"Claims Overlap: {risk_breakdown.get('claims_overlap', 0)}/100")
        lines.append(f"Design Freedom: {risk_breakdown.get('design_freedom', 0)}/100")
        lines.append(f"Invalidation Risk: {risk_breakdown.get('invalidation_risk', 0)}/100")
        lines.append(f"License Cost: {risk_breakdown.get('license_cost', 0)}/100")
        lines.append("")

    # Patent Timeline
    if patent_timeline and patent_timeline.get("milestones"):
        lines.append("-" * 80)
        lines.append("KEY DATES & MILESTONES")
        lines.append("-" * 80)
        lines.append("")
        for milestone in patent_timeline.get("milestones", []):
            lines.append(f"{milestone.get('date', 'N/A')}: {milestone.get('event', '')}")
            lines.append(f"  → {milestone.get('significance', '')}")
        lines.append("")

    # Strongest Conflicts
    if conflict_clusters:
        lines.append("-" * 80)
        lines.append("STRONGEST CONFLICTS")
        lines.append("-" * 80)
        lines.append("")
        for i, cluster in enumerate(conflict_clusters[:5], 1):
            lines.append(f"{i}. {cluster.get('patent_id', 'Unknown')}")
            lines.append(f"   Title: {cluster.get('patent_title', '')}")
            lines.append(
                f"   Appearances: {cluster.get('frequency', 0)} paragraphs"
            )
            lines.append(
                f"   Avg Similarity: {int(cluster.get('average_similarity', 0) * 100)}%"
            )
            lines.append(f"   Conflict: {cluster.get('core_conflict', '')}")
            lines.append("")

    # Core Exposures
    if core_exposures:
        lines.append("-" * 80)
        lines.append("CORE EXPOSURES")
        lines.append("-" * 80)
        lines.append("")
        for exposure in core_exposures:
            lines.append(f"• {exposure.get('exposure', '')}")
            lines.append(f"  Severity: {exposure.get('severity', 'Unknown')}")
            lines.append(
                f"  Redesignable: {'Yes' if exposure.get('redesign_feasible') else 'No'}"
            )
            lines.append(f"  Rationale: {exposure.get('rationale', '')}")
            if exposure.get("workaround_options"):
                lines.append("  Workaround Options:")
                for option in exposure.get("workaround_options", []):
                    lines.append(f"    - {option}")
            lines.append("")

    # Action Items
    if action_items:
        lines.append("-" * 80)
        lines.append("ACTION ITEMS")
        lines.append("-" * 80)
        lines.append("")
        for item in action_items:
            priority = item.get("priority", "MEDIUM")
            lines.append(f"[{priority}] {item.get('action', '')}")
            lines.append(f"Why: {item.get('reason', '')}")
            lines.append("")

    # Per-Paragraph Analysis
    if paragraph_analyses:
        lines.append("-" * 80)
        lines.append("PER-PARAGRAPH ANALYSIS")
        lines.append("-" * 80)
        lines.append("")
        for analysis in paragraph_analyses:
            para_id = analysis.get("paragraph_id", "Unknown")
            risk = analysis.get("risk_score", 0)
            lines.append(f"Paragraph {para_id} (Risk: {risk}/100)")
            lines.append(f"  Components: {', '.join(analysis.get('technical_components', []))}")
            lines.append(f"  Flow: {analysis.get('functional_flow', '')}")
            exposure = analysis.get("prior_art_exposure", {})
            lines.append(f"  Overlap: {exposure.get('overlap_level', 'None')}")
            if exposure.get("vulnerable_elements"):
                lines.append(f"  Vulnerable: {', '.join(exposure.get('vulnerable_elements', []))}")
            if exposure.get("distinguishing_features"):
                lines.append(f"  Differentiators: {', '.join(exposure.get('distinguishing_features', []))}")
            lines.append("")

    # Footer
    lines.append("=" * 80)
    lines.append("End of Report")
    lines.append("=" * 80)

    return "\n".join(lines)


def _risk_label(score: int) -> str:
    """Convert risk score to label."""
    if score >= 70:
        return "HIGH RISK"
    elif score >= 40:
        return "MODERATE RISK"
    else:
        return "LOW RISK"


def export_as_txt(report_dict: dict, filename: str = "claimcheck_report.txt") -> bytes:
    """Export report as plain text file."""
    report_text = _build_text_report(
        report_dict.get("invention_title", "Untitled Invention"),
        report_dict.get("classification", {}),
        report_dict.get("overall_risk_score", 0),
        report_dict.get("risk_breakdown", {}),
        report_dict.get("paragraph_analyses", []),
        report_dict.get("conflict_clusters", []),
        report_dict.get("core_exposures", []),
        report_dict.get("action_items", []),
        report_dict.get("patent_timeline", {}),
    )
    return report_text.encode("utf-8")
