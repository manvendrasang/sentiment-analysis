import csv
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === CONFIGURATIONS ===
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_JSON_FILE = "input file path"
OUTPUT_CSV_FILE = "output file path"

# === LOAD LLAMA MODEL AND TOKENIZER ===
print("⏳ Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("✅ Model loaded.")

# === TRANSLATE ONE SENTENCE ===
def translate_to_hindi(sentence):
    prompt = f"Translate the following sentence from English to Hindi:\n\n\"{sentence}\"\n\nHindi Translation:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                do_sample=False,
                temperature=0
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # EXTRACT TRANSLATION AFTER PROMPT
        if "Hindi Translation:" in output_text:
            translated = output_text.split("Hindi Translation:")[-1].strip()
        else:
            translated = output_text.strip()

        return translated
    except Exception as e:
        print(f"❌ Error translating: {sentence[:30]} - {e}")
        return ""

# === MAIN PROCESSING ===
def process_file(input_path, output_path, start, limit):
    translated_sentences = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = json.load(f)

    if not isinstance(lines, list):
        print("❌ JSON must be a list of strings.")
        return

    # CLEAN AND SLICE
    lines = [line.strip() for line in lines if line.strip()]
    lines_to_translate = lines[start:start+limit]

    for idx, line in enumerate(lines_to_translate, start=start):
        print(f"🔁 Translating line [{idx}]: {line[:50]}...")
        translated = translate_to_hindi(line)
        if translated:
            translated_sentences.append({"Sentences": translated})

    # SAVE TO CSV
    with open(output_path, "w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Sentences"])
        writer.writeheader()
        writer.writerows(translated_sentences)

    print(f"\n✅ Translated output saved to: {output_path}")

# === RUN(DRIVER) ===
if __name__ == "__main__":
    try:
        start = int(input("🔢 Start from which line (0-based index)? "))
        limit = int(input("🔢 How many lines to translate? "))
        process_file(INPUT_JSON_FILE, OUTPUT_CSV_FILE, start, limit)
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
