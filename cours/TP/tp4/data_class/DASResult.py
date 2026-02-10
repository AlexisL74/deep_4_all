from dataclasses import dataclass

@dataclass
class DASResult:
    prompt: str
    teacher_response: str
    teacher_mean_logprob: float
    student_mean_logprob: float
    divergence: float
    keep: bool