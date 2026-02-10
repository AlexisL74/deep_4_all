import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class StudentModel:
    def __init__(self, model_id : str = "", bnb_config : BitsAndBytesConfig = None) -> None:
        if not model_id:
            raise ValueError("model_id must be provided.")
        
        self.model_id = model_id
        
        print(f"Initializing StudentModel with ID: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            quantization_config=bnb_config, 
            device_map="auto", 
            trust_remote_code=True
        )

        self.model.eval()
        print(f"StudentModel {self.model_id} loaded successfully.")

    def get_logprobs(self, prompt: str, response: str) -> dict:
        """
        Calcule les log-probabilités de la réponse (Student) de manière robuste.
        Utilise la méthode de masquage standard (Labels = -100 pour le prompt).
        """
        # 1. Préparer le texte complet (Prompt + Réponse)
        # On utilise le chat template qui gère proprement les balises <|im_start|>, etc.
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # 2. Tokenizer le tout
        # return_tensors='pt' nous donne directement les tenseurs PyTorch
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids

        # 3. Identifier la longueur du Prompt pour le masquage
        # On regénère le prompt SEUL avec l'amorce de réponse (add_generation_prompt=True)
        # Cela inclut "<|im_start|>assistant\n" à la fin, pour s'aligner parfaitement.
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
                )

        # On tokenise le prompt seul pour avoir sa longueur exacte en tokens
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        response_start_idx = prompt_tokens.shape[1]

        # 4. Créer les Labels (Masking du Prompt)
        # -100 est l'index ignoré par défaut par CrossEntropyLoss de PyTorch
        labels = input_ids.clone()
        # On masque tout ce qui est avant le début de la réponse
        labels[:, :response_start_idx] = -100

        # 5. Calcul "Clean" avec CrossEntropyLoss
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            # Shift des logits et labels pour la prédiction "next token"
            # logits[t] prédit labels[t+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # reduction='none' nous donne la perte pour chaque token individuel
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

            # La Loss est par définition -log(p), donc log_prob = -loss
            token_logprobs = -token_losses

            # On ne garde que les tokens de la réponse (ceux qui n'étaient pas masqués à -100)
            # Note: shift_labels a été décalé, donc on utilise son masque
            valid_mask = shift_labels != -100
            valid_logprobs = token_logprobs[valid_mask].cpu().numpy()

        # Calcul des statistiques DAS
        total_logprob = np.sum(valid_logprobs)
        mean_logprob = np.exp(np.mean(valid_logprobs)) if len(valid_logprobs) > 0 else 0.0
        
        return {
            "total_logprob": total_logprob,
            "mean_logprob":  mean_logprob,
            "num_tokens":    len(valid_logprobs),
            "logprobs":      valid_logprobs.tolist()
        }

    