"""Prompt templates for local LLM judges."""

from __future__ import annotations

POINTWISE_PROMPT = """You are evaluating two candidate answers to a user question.

Question:
{question}

Response A:
{response_a}

Response B:
{response_b}

Do not answer the user question yourself.
Do not solve the task.
Do not provide chain-of-thought or any explanation.
Only compare Response A and Response B.
Return exactly one final line in exactly this format:
FINAL VERDICT: A
or
FINAL VERDICT: B
"""

POINTWISE_COT_PROMPT = """You are evaluating two candidate answers to a user question.

Question:
{question}

Response A:
{response_a}

Response B:
{response_b}

Do not answer the user question yourself.
Do not solve the task.
Compare Response A and Response B on factual correctness, instruction-following,
completeness, and clarity.
Keep the reasoning brief.
After your reasoning, end with exactly one final line in this format:
FINAL VERDICT: A
or
FINAL VERDICT: B
"""

PAIRWISE_PROMPT = """Please act as an impartial judge and compare the quality of two
assistant responses.

User prompt:
{question}

Assistant A:
{response_a}

Assistant B:
{response_b}

Do not answer the user prompt yourself.
Do not solve the task.
Do not provide chain-of-thought or any explanation.
Only compare Assistant A and Assistant B.
Output exactly one final line:
FINAL VERDICT: A
or
FINAL VERDICT: B
"""

PAIRWISE_COT_PROMPT = """Please act as an impartial judge and compare two assistant
responses to the same user prompt.

User prompt:
{question}

Assistant A:
{response_a}

Assistant B:
{response_b}

Do not answer the user prompt yourself.
Do not solve the task.
Begin with a short comparison of correctness, helpfulness, relevance, and concision.
Keep the reasoning brief.
End with exactly one final line in this format:
FINAL VERDICT: A
or
FINAL VERDICT: B
"""

PROMPT_TEMPLATES = {
    "pointwise": POINTWISE_PROMPT,
    "pointwise_cot": POINTWISE_COT_PROMPT,
    "pairwise": PAIRWISE_PROMPT,
    "pairwise_cot": PAIRWISE_COT_PROMPT,
}


def format_prompt(
    template_name: str,
    *,
    question: str,
    response_a: str,
    response_b: str,
) -> str:
    """Format a prompt for the requested template."""

    try:
        template = PROMPT_TEMPLATES[template_name]
    except KeyError as exc:
        supported = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt template '{template_name}'. Supported: {supported}"
        ) from exc
    return template.format(
        question=question.strip(),
        response_a=response_a.strip(),
        response_b=response_b.strip(),
    )
