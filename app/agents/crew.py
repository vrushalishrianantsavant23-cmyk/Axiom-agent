import os
import litellm
litellm.drop_params = True

import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Agent, Task, Crew, Process, LLM

from app.config import GROQ_API_KEY, GROQ_MODEL

os.environ.setdefault("GROQ_API_KEY", GROQ_API_KEY)


def get_llm():
    return LLM(model=f"groq/{GROQ_MODEL}", api_key=GROQ_API_KEY, temperature=0.3, max_tokens=300)


def run_verification_crew(query: str, majority_answer: str, evidence: list) -> dict:
    llm = get_llm()

    evidence_text = "\n".join(
        f"- {e['text'][:300]} (source: {e['source']})" for e in evidence
    ) or "No supporting documents were retrieved for this query."

    fact_checker = Agent(
        role="Fact-Checker",
        goal="Verify whether the draft answer is accurate, using retrieved evidence when relevant.",
        backstory="A meticulous researcher who checks claims against sources when available, and against general knowledge otherwise.",
        llm=llm,
        verbose=False,
        cache=False,
    )
    skeptic = Agent(
        role="Skeptic",
        goal="Challenge the draft answer only where there is a genuine reason to doubt it.",
        backstory="A critical thinker who raises real concerns, not manufactured ones, and is comfortable agreeing when an answer is solid.",
        llm=llm,
        verbose=False,
        cache=False,
    )
    judge = Agent(
        role="Judge",
        goal="Deliver a final, natural, user-facing answer using the fact-checker and skeptic's findings.",
        backstory="An impartial adjudicator who gives clear, direct, confident answers like a helpful AI assistant.",
        llm=llm,
        verbose=False,
        cache=False,
    )

    fact_check_task = Task(
        description=(
            f"Claim/Question: {query}\n"
            f"Draft answer: {majority_answer}\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Check whether the draft answer is accurate. If document evidence was "
            "retrieved above, check whether the answer aligns with it. If NO document "
            "evidence was retrieved (or it says 'No supporting documents were retrieved'), "
            "that is completely normal for general knowledge questions unrelated to any "
            "uploaded document — in that case, judge the answer's accuracy using your own "
            "general knowledge instead, and do NOT flag it as 'unsupported' just because "
            "no document was found. Only raise concerns if the answer itself contains a "
            "factual error. Respond in 2-3 sentences."
        ),
        expected_output="A short assessment of whether the answer is factually accurate.",
        agent=fact_checker,
    )
    skeptic_task = Task(
        description=(
            f"Claim/Question: {query}\n"
            f"Draft answer: {majority_answer}\n\n"
            "Challenge this answer only if there is a genuine reason to doubt it — a "
            "factual error, an oversimplification, or missing important nuance. Do NOT "
            "invent doubts just because no document evidence exists; for general "
            "knowledge questions, judge based on whether the answer itself is correct. "
            "If the answer is solid, say so briefly instead of manufacturing concerns. "
            "Respond in 2-3 sentences."
        ),
        expected_output="A short, genuine assessment — real concerns if any exist, or a "
                         "brief confirmation that the answer holds up.",
        agent=skeptic,
    )
    judge_task = Task(
        description=(
            f"Claim/Question: {query}\n"
            f"Draft answer: {majority_answer}\n\n"
            "Using the fact-checker's and skeptic's findings as background context, write "
            "ONE clear, direct, natural answer to the original question — the way a helpful "
            "assistant like ChatGPT would answer. Do NOT write in a skeptical or "
            "fact-checking tone, do NOT say things are 'unsupported' unless there is a "
            "genuine factual problem with the answer. If the fact-checker and skeptic found "
            "no real issues, just give the straightforward, confident answer. Only mention "
            "uncertainty or lack of evidence if the fact-checker/skeptic raised a genuine, "
            "substantive concern — not for well-established facts like basic math, general "
            "knowledge, or common definitions. If the topic is genuinely contested or "
            "subjective, briefly note the different perspectives — otherwise just answer "
            "directly. Do not describe your process — just give the answer, in normal "
            "prose (not bullet points unless the question itself calls for a list)."
        ),
        expected_output="A direct, natural, confident answer to the original question, "
                         "written the way a helpful AI assistant would normally respond.",
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