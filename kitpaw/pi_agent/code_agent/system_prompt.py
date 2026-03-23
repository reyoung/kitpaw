from __future__ import annotations

from xml.sax.saxutils import escape as xml_escape, quoteattr as xml_quoteattr

from .types import Skill


def format_skills_for_prompt(skills: list[Skill]) -> str:
    if not skills:
        return ""
    blocks = [
        f'<skill name={xml_quoteattr(skill.name)} location={xml_quoteattr(skill.base_dir)}>{xml_escape(skill.description)}</skill>'
        for skill in skills
        if not skill.disable_model_invocation
    ]
    if not blocks:
        return ""
    return "Available skills:\n" + "\n".join(blocks)


def default_build_system_prompt(base_prompt: str | None, skills: list[Skill]) -> str:
    parts = [
        base_prompt
        or (
            "You are pi, a terminal coding agent. Use tools when needed. Be concise, precise, and prefer reading before editing."
        )
    ]
    skill_text = format_skills_for_prompt(skills)
    if skill_text:
        parts.append(skill_text)
    return "\n\n".join(parts)
