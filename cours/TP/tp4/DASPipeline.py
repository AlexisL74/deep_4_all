from StudentModel import StudentModel
from Promptotron import PrompteResponse
from dataset_utils import load_prompt_responses
from data_class.DASResult import DASResult

class DASPipeline:
    def __init__(self, student_model: StudentModel, divergence_threshold: float = 0.1, pertinence_threshold: float = 0.5) -> None:
        self.student_model = student_model
        self.divergence_threshold = divergence_threshold
        self.pertinence_threshold = pertinence_threshold
        self.verbose = True

    def score(self, teacher_response: PrompteResponse) -> DASResult:
        student_data = self.student_model.get_logprobs(teacher_response.input, teacher_response.output)
        
        teacher_logprob = teacher_response.mean_logprob
        student_logprob = student_data.get("mean_logprob", 0.0)
        divergence = teacher_logprob - student_logprob
        keep = (divergence > self.divergence_threshold) and (teacher_logprob >= self.pertinence_threshold)

        if self.verbose:
            print(f"Teacher LogProb: {teacher_logprob:.6f}, Student LogProb: {student_logprob:.6f}")
            print(f"Divergence (Teacher - Student): {divergence:.6f}")

        return DASResult(
            prompt=teacher_response.input,
            teacher_response=teacher_response.output,
            teacher_mean_logprob=teacher_logprob,
            student_mean_logprob=student_logprob,
            divergence=divergence,
            keep=keep,
        )

    def filter_dataset(self, dataset: list[PromptResponse]) -> list[DASResult]:
        kept = []
        total = len(dataset)

        for idx, teacher_response in enumerate(dataset):
            result = self.score(teacher_response)
            
            if self.verbose:
                status = "KEEP" if result.keep else "SKIP"
                print(
                    f"[{idx + 1}/{total}] {status} | "
                    f"div={result.divergence:.4f} | "
                    f"teacher={result.teacher_mean_logprob:.4f} | "
                    f"student={result.student_mean_logprob:.4f} | "
                    f"prompt='{result.prompt[:60]}...'"
                )
            
            if result.keep:
                kept.append(result)
            
        if self.verbose:
            print(f"\n{len(kept)}/{total} exemples gardés (seuil >= {self.divergence_threshold})")
        
        return kept

    def response_filtering(self, low_temp_path: str, high_temp_path: str) -> dict[str, list[DASResult]]:
        low_temp_dataset = load_prompt_responses(low_temp_path)
        high_temp_dataset = load_prompt_responses(high_temp_path)

        print(f"Datasets chargés : {len(low_temp_dataset)} low_temp, {len(high_temp_dataset)} high_temp")

        print("\n" + "=" * 60)
        print("[DASPipeline] Filtrage du dataset LOW TEMPERATURE")
        print("=" * 60)
        low_kept = self.filter_dataset(low_temp_dataset)

        print("\n" + "=" * 60)
        print("[DASPipeline] Filtrage du dataset HIGH TEMPERATURE")
        print("=" * 60)
        high_kept = self.filter_dataset(high_temp_dataset)

        print("\n" + "=" * 60)
        print(f"[RÉSUMÉ] LOW: {len(low_kept)}/{len(low_temp_dataset)} | HIGH: {len(high_kept)}/{len(high_temp_dataset)}")
        print("=" * 60)

        return {"low_temp_kept": low_kept, "high_temp_kept": high_kept}

    def temperature_scheduling_learning