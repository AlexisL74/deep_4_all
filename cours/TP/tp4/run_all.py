import torch
from transformers import BitsAndBytesConfig
from StudentModel import StudentModel
from DASPipeline import DASPipeline
from DASTrainer import DASTrainer
from dataset_utils import save_sharegpt


if __name__ == "__main__":
    # ============================================
    # ÉTAPE 1 : Filtrage DAS
    # ============================================
    print("=" * 60)
    print("ÉTAPE 1 : Response Filtering (DAS)")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    student = StudentModel(
        model_id="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        bnb_config=bnb_config,
    )

    pipeline = DASPipeline(student_model=student, divergence_threshold=0.1)
    results = pipeline.response_filtering(
        low_temp_path="data/low_temperature_responses.json",
        high_temp_path="data/high_temperature_responses.json",
    )

    save_sharegpt(results["low_temp_kept"], "data/train_stage_1.json")
    save_sharegpt(results["high_temp_kept"], "data/train_stage_2.json")

    del student
    del pipeline
    torch.cuda.empty_cache()

    # ============================================
    # ÉTAPE 2 : Entraînement LLaMA-Factory
    # ============================================
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Training (LLaMA-Factory)")
    print("=" * 60)

    trainer = DASTrainer(
        model_name="Qwen/Qwen3-4B-Instruct",
        output_dir="output",
        llamafactory_data_dir="data",
        lora_rank=16,
        lora_alpha=32,
        num_epochs_stage1=3.0,
        num_epochs_stage2=2.0,
        lr_stage1=2e-5,
        lr_stage2=1e-5,
        batch_size=1,
        gradient_accumulation_steps=8,
        cutoff_len=2048,
    )

    train_results = trainer.train_full_pipeline(
        stage1_path="data/train_stage_1.json",
        stage2_path="data/train_stage_2.json",
        export=True,
    )

    print("\n=== PIPELINE COMPLET TERMINÉ ===")
    for k, v in train_results.items():
        print(f"  {k}: {v}")