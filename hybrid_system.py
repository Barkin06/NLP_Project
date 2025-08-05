import re
import torch
import pickle
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoTokenizer as LLMTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ---------------- CONFIG ----------------
INTENT_MODEL_PATH = "./deberta_intent_model"
LABEL_ENCODER_PATH = "./label_encoder.pkl"
LLM_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Hugging Face'den çekilecek

CONF_THRESH = 0.70
MAX_NEW_TOKENS = 80
MAX_SENTENCES = 2
LLM_MAX_INPUT_TOK = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using {DEVICE.upper()}")

# ---------------- LOAD MODELS ----------------
# Intent Model (Local)
intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH).to(DEVICE)

# Label Encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
id2label = {i: l for i, l in enumerate(label_encoder.classes_)}

# Phi-3 Model (Quantized, Online from Hugging Face)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

llm_tokenizer = LLMTokenizer.from_pretrained(
    LLM_NAME,
    trust_remote_code=True  # Özel kodlar için zorunlu
)

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    trust_remote_code=True,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.float16
)

# Tokenizer için pad token ayarı
if llm_tokenizer.pad_token_id is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

# ---------------- PROMPT ----------------
def build_prompt(question: str, intent: str = None):
    extra = f"Intent: {intent}\n" if intent else ""
    return (
        "<|system|>\n"
        "You are a helpful assistant. Follow these rules strictly:\n"
        "- ONLY output the final answer, nothing else.\n"
        "- Use MAX 2 short sentences.\n"
        "- Do NOT repeat the question or add examples.\n"
        "- Do NOT change numbers.\n"
        "- If unsure, reply: I don't know.\n"
        f"{extra}<|end|>"
        f"<|user|>\nQuestion: {question}\n<|end|>"
        "<|assistant|>\nAnswer:"
    )

# ---------------- FUNCTIONS ----------------
def predict_intent(text: str):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = intent_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return id2label[idx], float(probs[idx])

def extract_answer(decoded: str) -> str:
    m = re.search(r"Answer:\s*(.*)", decoded, re.DOTALL)
    raw = m.group(1).strip() if m else decoded.strip()
    raw = re.sub(r"(Intent|Improved|User question|Revised).*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw)
    sentences = re.split(r'(?<=[.!?]) +', raw)
    return " ".join(sentences[:MAX_SENTENCES]).strip() or "I don't know."

def generate_llm(question: str, intent: str = None) -> str:
    prompt = build_prompt(question, intent)
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=LLM_MAX_INPUT_TOK).to(llm_model.device)

    out = llm_model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=llm_tokenizer.eos_token_id,
        pad_token_id=llm_tokenizer.pad_token_id
    )

    decoded = llm_tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_answer(decoded)

def hybrid_answer(user_query: str):
    intent_label, conf = predict_intent(user_query)
    use_intent = intent_label if (conf >= CONF_THRESH and intent_label not in ["no", "none", "oos"]) else None
    llm_answer = generate_llm(user_query, use_intent)
    return {
        "query": user_query,
        "intent": use_intent or "fallback",
        "confidence": round(conf, 3),
        "answer": llm_answer
    }

# ---------------- INTERACTIVE MODE ----------------
if __name__ == "__main__":
    print("\n✅ Hybrid System Ready! Type your question (or 'exit' to quit):\n")
    while True:
        user_q = input("❓ Your question: ").strip()
        if user_q.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        result = hybrid_answer(user_q)
        print(f"👉 Intent: {result['intent']} (conf: {result['confidence']})")
        print(f"💡 Answer: {result['answer']}\n")
