from DASTrainer import DASTrainer

if __name__ == "__main__":
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
        template="qwen3",
    )

    # Pipeline complet
    results = trainer.train_full_pipeline(
        stage1_path="data/train_stage_1.json",
        stage2_path="data/train_stage_2.json",
        export=True,
    )

    print("\n=== RÉSULTATS ===")
    print(f"Stage 1 adapter : {results['stage1_adapter']}")
    print(f"Stage 2 adapter : {results['stage2_adapter']}")
    if "merged_model" in results:
        print(f"Merged model    : {results['merged_model']}")