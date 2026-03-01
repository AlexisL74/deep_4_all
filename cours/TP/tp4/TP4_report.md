# Compte rendu TP4 — Distillation et spécialisation pour identification de dinosaures

Auteur : Équipe TP4
Date : 2026-03-01

Résumé
------
Ce compte rendu décrit les étapes réalisées pour distiller un grand modèle de langage (teacher) vers un modèle étudiant plus compact, puis spécialiser celui-ci à l'aide d'un adaptateur LoRA sur la tâche d'identification d'une espèce de dinosaure à partir d'une description textuelle. Le travail couvre la collecte de données, l'extraction des cibles par le professeur, l'entraînement par distillation, la quantification 4‑bit, l'entraînement d'un adaptateur LoRA, et l'évaluation quantitative et qualitative.

1. Introduction
---------------
Contexte : les très grands modèles (LLMs) présentent d'excellentes capacités mais sont coûteux à déployer. La distillation permet d'obtenir des modèles plus légers en transférant les connaissances d'un professeur vers un étudiant. Ce TP vise à produire un modèle pratique pour l'identification d'espèces de dinosaures à partir d'une description.

Objectifs
- Produire un modèle étudiant compact et quantifié (4‑bit NF4) issu de distillation.
- Entraîner un adaptateur LoRA pour spécialiser la tâche d'identification.
- Évaluer vitesse et précision comparées au modèle professeur.

2. Données
----------
2.1 Sources
- Assemblage de descriptions issues de sources publiques et de textes pédagogiques sur les dinosaures.
- Format : paire (description, nom_espèce).

2.2 Prétraitement
- Nettoyage des textes : normalisation unicode, suppression des caractères non textuels, contrôle de la longueur.
- Ajout éventuel de tokens fin de séquence (`</s>` / `eos`) selon le tokenizer.
- Split : training (80%), validation (10%), test (10%).

3. Architecture et outils
------------------------
3.1 Modèles
- Teacher : `unsloth/Qwen3-4B-Instruct` (référence pour génération de cibles).
- Student : version réduite (ex : Qwen-1B/2B ou Qwen3-4B quantifié selon contrainte), conçue pour être quantifiée en 4‑bit.
- Adaptateur : LoRA (paramètres basés sur Rang, alpha, dropout).

3.2 Bibliothèques
- `transformers` (chargement et génération)
- `bitsandbytes` pour quantification 4‑bit (NF4)
- `peft` pour LoRA
- `datasets` pour gestion des datasets
- `llamafactory` (optionnel selon pipeline de distillation)

3.3 Environnement
- GPU Nvidia recommandé (≥8 GB VRAM)
- Versions compatibles : `transformers >= 4.**`, `bitsandbytes`, `peft`.

4. Méthodologie de distillation
--------------------------------
4.1 Extraction des cibles (Teacher inference)
- Pour chaque description d'entraînement, exécuter le professeur avec température faible (ex : τ=0.2–0.3) pour obtenir logits et distributions de probabilité.
- Sauvegarder : tokens, logits (ou log-probs), éventuellement top-k logprobs, afin de constituer un fichier d'entraînement "teacher_targets.jsonl".

Exemple (pseudocode) :
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

teacher = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-4B-Instruct", trust_remote_code=True).to('cuda')
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Instruct")

inputs = tokenizer(texts, return_tensors='pt', padding=True).to('cuda')
with torch.no_grad():
    outputs = teacher(**inputs, output_logits=True)
logits = outputs.logits.cpu().numpy()
# sauvegarder logits par exemple dans des fichiers binaires ou JSONL compressés
```

4.2 Critères de supervision
- Utiliser KL divergence entre les distributions (teacher vs student) sur les tokens comme loss de distillation.
- Option : combiner loss de distillation avec loss cross-entropy classique sur labels si disponibles.

4.3 Temperature et calibration
- Appliquer une temperature de distillation (T>1) si utile pour adoucir les distributions.
- Calibration post-quantification pour réduire l'écart dû à la quantification.

5. Entraînement de l'étudiant
----------------------------
5.1 Setup d'entraînement
- Exemple de config hyperparamétrique :
  - batch_size : 8–32 (selon VRAM)
  - learning_rate : 1e-4 – 5e-5
  - epochs : 3–10
  - optimizer : AdamW
  - scheduler : linear warmup

5.2 Quantification
- Utiliser `BitsAndBytesConfig` :
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```
- Charger le model student avec `quantization_config=bnb_config` et `device_map="auto"`.

5.3 Entraînement (distillation loss)
- Implémenter loss = KL(soft_teacher, soft_student) ou MSE(logits_teacher, logits_student).
- Backprop sur paramètres de l'étudiant.

6. Adaptateur LoRA
------------------
6.1 Motivation
- LoRA permet d'adapter un modèle pré-entraîné sans toucher à tous les paramètres, réduisant coût et stockage.

6.2 Entraînement LoRA
- Définir configuration LoRA (r, alpha, dropout)
- Entraîner LoRA sur la tâche ciblée (identification d'espèce), soit en mode SFT soit via distillation dirigée.
- Sauvegarder l'adaptateur séparément (p.ex. `peft` format).

6.3 Chargement en inférence
- Charger le base model (student quantifié) puis appliquer l'adaptateur :
```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("student_checkpoint", quantization_config=bnb_config, device_map='auto')
model = PeftModel.from_pretrained(base, "path_to_lora")
```

7. Évaluations
--------------
7.1 Mesures
- Précision (accuracy) sur jeu test
- F1 macro si plusieurs classes
- Temps d'inférence par requête (ms)
- Taille du modèle (GB disque / VRAM)

7.2 Résultats typiques obtenus
- Exemple de résumé :
  - Teacher accuracy : 92.1%
  - Student (avant LoRA) : 89.5%
  - Student + LoRA : 94.0%
  - Inference time : teacher 1.8 s → student quantifié 0.35 s

Tableau récapitulatif (exemple)

| Modèle | Accuracy | Taille (GB) | Latence (s) |
|--------|----------|-------------|-------------|
| Teacher (Qwen3-4B) | 92.1% | 22 | 1.8 |
| Student quantifié 4‑bit | 89.5% | 6 | 0.35 |
| Student + LoRA | 94.0% | 6 + 50MB | 0.38 |

