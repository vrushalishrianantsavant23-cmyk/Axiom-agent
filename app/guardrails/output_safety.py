def validate_output(answer: str, sources: list) -> dict:
    """Post-generation validation. If the answer is empty/too short/looks like an
    error, fall back to an explicit 'insufficient evidence' response instead of
    silently returning something misleading."""
    if not answer or len(answer.strip()) < 3:
        return {"valid": False, "fallback": "Insufficient evidence to provide a confident answer."}
    if answer.strip().startswith("[error"):
        return {"valid": False, "fallback": "Insufficient evidence to provide a confident answer."}
    return {"valid": True, "fallback": None}
