NEUTRALITY_INSTRUCTION = (
    "If this topic is contested, political, religious, or subjective, present multiple "
    "perspectives fairly instead of a single one-sided verdict. Do not favor any "
    "ideological, political, or personal viewpoint."
)

CONTESTED_MARKERS = [
    "should", "better than", "is it right", "morally", "politics", "political",
    "religion", "abortion", "election", "government policy", "opinion on",
    "who is correct", "which is superior",
]


def is_contested_topic(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in CONTESTED_MARKERS)
