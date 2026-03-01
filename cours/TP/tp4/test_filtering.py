from StudentModel import StudentModel
from DASPipeline import DASPipeline
from dataset_utils import save_das_results, save_sharegpt
from transformers import BitsAndBytesConfig
import torch


if __name__ == "__main__":
    # 1. Charger le Student
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    student = StudentModel(model_id="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", bnb_config=bnb_config)

    # 2. Pipeline DAS
    pipeline = DASPipeline(student_model=student, divergence_threshold=0.1)

    # 3. Filtrage sur les deux datasets existants
    results = pipeline.response_filtering(
        low_temp_path="data/low_temperature_dataset.json",
        high_temp_path="data/high_temperature_dataset.json",
    )

    low_kept = results["low_temp_kept"]
    high_kept = results["high_temp_kept"]

    # 4. Sauvegarder en ShareGPT
    save_sharegpt(low_kept, "data/train_stage_1.json")
    save_sharegpt(high_kept, "data/train_stage_2.json")

    # 5. Aperçu
    # print("\n--- Aperçu ShareGPT ---")
    # for item in all_kept[:3]:
    #     print(f"  [{item.divergence:.4f}] {item.prompt[:80]}...")