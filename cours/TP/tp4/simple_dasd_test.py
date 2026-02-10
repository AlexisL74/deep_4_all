#!/usr/bin/env python3
"""
Script minimal pour tester que l'API OpenAI répond et que la clé est correcte.

Usage:
  export OPENAI_API_KEY="sk-..."
  python3 cours/TP/tp4/simple_dasd_test.py "Explique pourquoi le ciel est bleu."

Le script capte les erreurs d'authentification et affiche un message utile.
"""
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


def main(prompt: str):
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")

    print(f"Test local model load: {model_name}")

    bnb_cfg = None
    if BitsAndBytesConfig is not None:
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("BitsAndBytesConfig disponible : tentative de chargement en 4-bit.")
        except Exception:
            bnb_cfg = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if bnb_cfg is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # Fallback : tenter de charger en fp16 si 4-bit non disponible
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=128, do_sample=False)

        out = tokenizer.decode(gen[0], skip_special_tokens=True)
        print("--- Génération (local) ---")
        print(out)

    except Exception as e:
        print("Erreur lors du chargement ou de l'inférence du modèle local :")
        print(str(e))
        print("")
        print("Conseils :")
        print(" - Vérifiez que vous avez accès au modèle '" + model_name + "' depuis Hugging Face.")
        print(" - Installez 'transformers', 'torch' et 'bitsandbytes' si vous voulez le chargement 4-bit.")
        sys.exit(2)


if __name__ == "__main__":
    prompt = "Explique pourquoi le ciel est bleu de manière concise et scientifique." if len(sys.argv) == 1 else " ".join(sys.argv[1:])
    main(prompt)
