import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

# Promptotron import (teacher)
try:
    from promptotron import Promptotron
except Exception:
    Promptotron = None

# Local configuration
STUDENT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
LORA_ADAPTER = "Djapamal/DinoQwen3-4B-DASD"

# Small dataset of 10 example descriptions with expected species (examples)
PROMPTS = [
    {"id": 1, "text": "Large bipedal carnivore with tiny forelimbs and huge jaws, often considered the king of the tyrant dinosaurs.", "label": "tyrannosaurus rex"},
    {"id": 2, "text": "Long neck, massive body, columnar legs, herbivore that browsed high vegetation.", "label": "brachiosaurus"},
    {"id": 3, "text": "Armored dinosaur with a clubbed tail and plates or spikes along its back.", "label": "ankylosaurus"},
    {"id": 4, "text": "Two-horned face with a large frill, herbivorous, often found in herds.", "label": "triceratops"},
    {"id": 5, "text": "Medium-sized, fast, sickle-clawed theropod, likely pack hunter with a long tail for balance.", "label": "velociraptor"},
    {"id": 6, "text": "Large sail-backed reptile with elongated neural spines on its back, semi-aquatic predator.", "label": "spinosaurus"},
    {"id": 7, "text": "Small to medium bird-like dinosaur with feathers and long arms — considered close to the origin of birds.", "label": "archaeopteryx"},
    {"id": 8, "text": "Huge armored plates along the back and a very long tail, often mistaken in popular culture with stegosaurus-like plates.", "label": "stegosaurus"},
    {"id": 9, "text": "Small herbivorous ornithopod known for its duck-billed snout and herd behavior.", "label": "hadrosaurus"},
    {"id": 10, "text": "A nimble, small-bodied predator with a crested head and likely strong eyesight.", "label": "allosaurus"}
]

RESULTS_FILE = "data/benchmark_results.json"

# Helper: normalize text

def norm(s):
    return s.lower().strip()


def run_teacher(prompts):    
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("API_URL")

    print(api_key, base_url)

    pt = Promptotron(api_key=api_key, base_url=base_url)
    teacher_results = {}
    for p in prompts:
        try:
            # Promptotron.double_temperature_prompt returns (low, high)
            low, high = pt.double_temperature_prompt(p['text'], p['id'])
            # choose low-temperature response for teacher label
            teacher_results[p['id']] = {
                'text': low.output,
                'mean_logprob': low.mean_logprob,
                'temperature': low.temperature
            }
            print(f"Teacher OK for prompt {p['id']}")
        except Exception as e:
            print(f"Teacher error for prompt {p['id']}: {e}")
            teacher_results[p['id']] = {'text': '', 'error': str(e)}
    return teacher_results


def load_student_model(model_name):
    print(f"Loading student model {model_name} (may be large)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def run_model_on_prompts(tokenizer, model, prompts, max_new_tokens=40):
    results = {}
    for p in prompts:
        text = p['text']
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        latency = time.perf_counter() - start
        gen = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        results[p['id']] = {'text': gen, 'latency_s': latency}
        print(f"Model {getattr(model,'__class__',type(model)).__name__} prompt {p['id']} done in {latency:.3f}s")
    return results


def judge_results(prompts, model_outputs):
    correct = 0
    judged = {}
    for p in prompts:
        out = model_outputs.get(p['id'], {}).get('text','')
        match = norm(p['label']) in norm(out)
        judged[p['id']] = {'expected': p['label'], 'output': out, 'match': match}
        if match:
            correct += 1
    accuracy = correct / len(prompts)
    return accuracy, judged


def main():
    load_dotenv() 
    # 1) Teacher
    teacher_results = run_teacher(PROMPTS)

    # 2) Student base
    try:
        tokenizer_s, student = load_student_model(STUDENT_MODEL)
    except Exception as e:
        print(f"Could not load student model: {e}")
        tokenizer_s, student = None, None

    student_results = None
    if student is not None:
        student_results = run_model_on_prompts(tokenizer_s, student, PROMPTS)

    # 3) Student + LoRA
    student_lora_results = None
    try:
        # reload base for LoRA wrapper (safer)
        tokenizer_s2 = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
        base_for_lora = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
            trust_remote_code=True,
        )
        lora_model = PeftModel.from_pretrained(base_for_lora, LORA_ADAPTER, trust_remote_code=True)
        lora_model.eval()
        student_lora_results = run_model_on_prompts(tokenizer_s2, lora_model, PROMPTS)
    except Exception as e:
        print(f"Could not load/apply LoRA adapter: {e}")

    # Judge
    report = {'prompts': PROMPTS, 'teacher': teacher_results, 'student': student_results, 'student_lora': student_lora_results}

    summary = {}
    if teacher_results is not None:
        # teacher may not have gold labels, but we can still store outputs
        summary['teacher_available'] = True
    else:
        summary['teacher_available'] = False

    if student_results is not None:
        acc_s, judged_s = judge_results(PROMPTS, student_results)
        summary['student_accuracy'] = acc_s
        report['student_judged'] = judged_s
    if student_lora_results is not None:
        acc_l, judged_l = judge_results(PROMPTS, student_lora_results)
        summary['student_lora_accuracy'] = acc_l
        report['student_lora_judged'] = judged_l

    # Save
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'report': report, 'summary': summary}, f, ensure_ascii=False, indent=2)

    print("\nBenchmark summary:")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()
