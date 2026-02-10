import json
import os
from pathlib import Path
from llamafactory.train.tuner import export_model, run_exp


class DASTrainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct",
        output_dir: str = "output",
        llamafactory_data_dir: str = "data",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        num_epochs_stage1: float = 3.0,
        num_epochs_stage2: float = 2.0,
        lr_stage1: float = 2e-5,
        lr_stage2: float = 1e-5,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        cutoff_len: int = 2048,
        template: str = "qwen3",
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.data_dir = Path(llamafactory_data_dir)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_epochs_stage1 = num_epochs_stage1
        self.num_epochs_stage2 = num_epochs_stage2
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.cutoff_len = cutoff_len
        self.template = template

        self.stage1_output = str(self.output_dir / "stage1")
        self.stage2_output = str(self.output_dir / "stage2")
        self.merged_output = str(self.output_dir / "merged")

    def _register_datasets(self, stage1_path: str, stage2_path: str) -> None:
        """
        Crée/met à jour le dataset_info.json pour LLaMA-Factory.
        """
        dataset_info_path = self.data_dir / "dataset_info.json"

        dataset_info = {}
        if dataset_info_path.exists():
            with open(dataset_info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)

        for name, path in [("das_stage1", stage1_path), ("das_stage2", stage2_path)]:
            dataset_info[name] = {
                "file_name": str(Path(path).resolve()),
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                },
            }

        dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        print(f"[DASTrainer] dataset_info.json mis à jour : {dataset_info_path}")

    def _build_args(self, stage: int, dataset_name: str, adapter_path: str | None = None) -> dict:
        is_stage1 = stage == 1
        args = {
            # Model
            "model_name_or_path": self.model_name,
            "template": self.template,
            # Method
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_target": "all",
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": 0.05,
            # Dataset
            "dataset": dataset_name,
            "dataset_dir": str(self.data_dir.resolve()),
            "cutoff_len": self.cutoff_len,
            "preprocessing_num_workers": 4,
            # Output
            "output_dir": self.stage1_output if is_stage1 else self.stage2_output,
            "logging_steps": 5,
            "save_steps": 100,
            "save_total_limit": 2,
            "overwrite_output_dir": True,
            # Train
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.lr_stage1 if is_stage1 else self.lr_stage2,
            "num_train_epochs": self.num_epochs_stage1 if is_stage1 else self.num_epochs_stage2,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
            "gradient_checkpointing": True,
            # Misc
            "report_to": "none",
        }

        if adapter_path:
            args["adapter_name_or_path"] = adapter_path

        return args

    def train_stage1(self, stage1_sharegpt_path: str) -> str:
        self._register_datasets(stage1_sharegpt_path, stage1_sharegpt_path)

        args = self._build_args(stage=1, dataset_name="das_stage1")

        print("\n" + "=" * 60)
        print("[DASTrainer] STAGE 1 — Low Temperature Training")
        print("=" * 60)
        print(f"  Model:    {self.model_name}")
        print(f"  Dataset:  {stage1_sharegpt_path}")
        print(f"  LR:       {self.lr_stage1}")
        print(f"  Epochs:   {self.num_epochs_stage1}")
        print(f"  Output:   {self.stage1_output}")
        print("=" * 60)

        run_exp(args)

        print(f"\n[DASTrainer] Stage 1 terminé. Adapter sauvegardé dans : {self.stage1_output}")
        return self.stage1_output

    def train_stage2(self, stage2_sharegpt_path: str, stage1_adapter_path: str | None = None) -> str:
        from llamafactory.train.tuner import run_exp

        if stage1_adapter_path is None:
            stage1_adapter_path = self.stage1_output

        if not Path(stage1_adapter_path).exists():
            raise FileNotFoundError(
                f"Adapter Stage 1 introuvable : {stage1_adapter_path}. "
                "Lancez train_stage1() d'abord."
            )

        self._register_datasets(stage2_sharegpt_path, stage2_sharegpt_path)

        args = self._build_args(
            stage=2,
            dataset_name="das_stage2",
            adapter_path=stage1_adapter_path,
        )

        print("\n" + "=" * 60)
        print("[DASTrainer] STAGE 2 — High Temperature Training")
        print("=" * 60)
        print(f"  Model:    {self.model_name}")
        print(f"  Adapter:  {stage1_adapter_path}")
        print(f"  Dataset:  {stage2_sharegpt_path}")
        print(f"  LR:       {self.lr_stage2}")
        print(f"  Epochs:   {self.num_epochs_stage2}")
        print(f"  Output:   {self.stage2_output}")
        print("=" * 60)

        run_exp(args)

        print(f"\n[DASTrainer] Stage 2 terminé. Adapter sauvegardé dans : {self.stage2_output}")
        return self.stage2_output

    def export_merged(self, adapter_path: str | None = None) -> str:
        if adapter_path is None:
            adapter_path = self.stage2_output

        args = {
            "model_name_or_path": self.model_name,
            "adapter_name_or_path": adapter_path,
            "template": self.template,
            "finetuning_type": "lora",
            "export_dir": self.merged_output,
            "export_size": 4,
            "export_device": "cpu",
            "export_legacy_format": False,
        }

        print("\n" + "=" * 60)
        print("[DASTrainer] Export du modèle fusionné")
        print(f"  Adapter:  {adapter_path}")
        print(f"  Output:   {self.merged_output}")
        print("=" * 60)

        export_model(args)

        print(f"\n[DASTrainer] Modèle fusionné exporté dans : {self.merged_output}")
        return self.merged_output

    def train_full_pipeline(self, stage1_path: str, stage2_path: str, export: bool = False) -> dict:
        stage1_adapter = self.train_stage1(stage1_path)
        stage2_adapter = self.train_stage2(stage2_path, stage1_adapter)

        result = {
            "stage1_adapter": stage1_adapter,
            "stage2_adapter": stage2_adapter,
        }

        if export:
            merged_path = self.export_merged(stage2_adapter)
            result["merged_model"] = merged_path

        return result