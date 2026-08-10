import os
import litellm
litellm.drop_params = True

# --- Workaround for CrewAI bug #5886: cache_breakpoint injected for non-Anthropic providers ---
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg
# -----------------------------------------------------------------------------------------------

from crewai import Agent, Task, Crew, Process, LLM

from app.config import GROQ_API_KEY, GROQ_MODEL

os.environ.setdefault("GROQ_API_KEY", GROQ_API_KEY)


def get_llm():
    return LLM(model=f"groq/{GROQ_MODEL}", api_key=GROQ_API_KEY, temperature=0.3)


def run_verification_crew(query: str, majority_answer: str, evidence: list) -> dict:
    llm = get_llm()

    evidence_text = "\n".join(
        f"- {e['text'][:300]} (source: {e['source']})" for e in evidence
    ) or "No supporting documents were retrieved for this query."

    fact_checker = Agent(
        role="Fact-Checker",
        goal="Verify whether the draft answer is supported by the retrieved evidence.",
        backstory="A meticulous researcher who cross-checks every claim against sources before accepting it.",
        llm=llm,
        verbose=False,
        cache=False,
    )
    skeptic = Agent(
        role="Skeptic",
        goal="Challenge the draft answer and surface weak evidence or logical gaps.",
        backstory="A critical thinker who assumes claims are wrong until proven otherwise.",
        llm=llm,
        verbose=False,
        cache=False,
    )
    judge_task = Task(
    description=(
        f"Claim/Question: {query}\n"
        f"Draft answer: {majority_answer}\n\n"
        "Using the fact-checker's and skeptic's findings, write a final answer "
        "to the original question, formatted as 3-6 concise bullet points "
        "(use '- ' at the start of each line). If the topic is contested or "
        "subjective, use separate bullet points to present each distinct "
        "perspective, clearly labeled. Do not write in paragraph form. "
        "Do not describe your process — just give the bulleted answer."
    ),
    expected_output="A final, bullet-point answer to the original question, "
                     "with each point on its own line starting with '- '.",
    agent=judge,
    context=[fact_check_task, skeptic_task],
)

    fact_check_task = Task(
        description=(
            f"Claim/Question: {query}\n"
            f"Draft answer: {majority_answer}\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Check whether the draft answer is supported by the evidence above. "
            "List any unsupported claims explicitly, in 2-3 sentences."
        ),
        expected_output="A short assessment of which parts of the draft answer are supported and which are not.",
        agent=fact_checker,
    )
    skeptic_task = Task(
        description=(
            f"Claim/Question: {query}\n"
            f"Draft answer: {majority_answer}\n\n"
            "Challenge this answer in 2-3 sentences. Point out any logical gaps, "
            "missing context, or reasons to doubt it, even if it looks correct at first glance."
        ),
        expected_output="A short list of doubts or challenges to the draft answer.",
        agent=skeptic,
    )
    judge_task = Task(
        description=(
            f"Claim/Question: {query}\n"
            f"Draft answer: {majority_answer}\n\n"
            "Using the fact-checker's and skeptic's findings, write ONE final, concise "
            "answer (3-5 sentences max) to the original question. If the topic is "
            "contested or subjective, briefly present multiple perspectives instead of "
            "a single one-sided verdict. Do not describe your process — just give the answer."
        ),
        expected_output="A final, concise, user-facing answer to the original question.",
        agent=judge,
        context=[fact_check_task, skeptic_task],
    )

    crew = Crew(
        agents=[fact_checker, skeptic, judge],
        tasks=[fact_check_task, skeptic_task, judge_task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        result = crew.kickoff()
    except Exception as e:
        return {
            "fact_check": str(fact_check_task.output) if fact_check_task.output else "",
            "skeptic_review": str(skeptic_task.output) if skeptic_task.output else "",
            "final_verdict": f"Verification failed due to an internal error: {e}",
        }

    return {
        "fact_check": str(fact_check_task.output) if fact_check_task.output else "",
        "skeptic_review": str(skeptic_task.output) if skeptic_task.output else "",
        "final_verdict": str(result),
    }