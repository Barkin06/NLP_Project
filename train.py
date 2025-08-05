# train.py
import json
import pickle
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)
import inspect

# ----------------- 0. Seed -----------------
set_seed(42)

# ----------------- 1. Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()} | #GPU: {torch.cuda.device_count()}")

# ----------------- 2. Veri Yükleme -----------------
with open("data_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_data = data["train"]
val_data   = data["val"] + data["oos_val"]
test_data  = data["test"] + data["oos_test"]

train_texts  = [x[0] for x in train_data]
train_labels = [x[1] for x in train_data]
val_texts    = [x[0] for x in val_data]
val_labels   = [x[1] for x in val_data]
test_texts   = [x[0] for x in test_data]
test_labels  = [x[1] for x in test_data]

# ----------------- 3. LabelEncoder ----------------- fit + transform
#fit() ID lere alfabetik olarak çevirir(integer)
label_encoder = LabelEncoder()
label_encoder.fit(train_labels + val_labels + test_labels)
num_labels = len(label_encoder.classes_)
print(f"✅ Total classes: {num_labels}")

#transform() bizim için çevirlmiş olan sayıları stringlerle mapler
train_labels = label_encoder.transform(train_labels)
val_labels   = label_encoder.transform(val_labels)
test_labels  = label_encoder.transform(test_labels)

# ----------------- 4. Tokenizer / Model -----------------
model_name = "microsoft/deberta-v3-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True) #AutoTokenizer() stringleri token id ye çevirir.

id2label = {i: l for i, l in enumerate(label_encoder.classes_)} #------------------------------!!!!!!!!!!!!!!!!!!!!!!!
label2id = {l: i for i, l in enumerate(label_encoder.classes_)}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label, #tahmin ve çıktı
    label2id=label2id #training ve loss hesabı
).to(device)

# ----------------- 5. Tokenization -----------------
def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

# ----------------- 6. HF Dataset ----------------- 
train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
val_dataset   = Dataset.from_dict({"text": val_texts,   "labels": val_labels})
test_dataset  = Dataset.from_dict({"text": test_texts,  "labels": test_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch",   columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch",  columns=["input_ids", "attention_mask", "labels"]) #torch (tensor) formatına çevir ki train edilebilssin

# ----------------- 7. Metrics -----------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted") #kesinlik ve duyarlılığın mean değeri
    }

# ----------------- 8. TrainingArguments (try/except) -----------------
def make_training_args():
    # Ortak paramlar
    base_kwargs = dict(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50
    )

    # 1) Modern argümanları dene
    try:
        return TrainingArguments(
            **base_kwargs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_dir="./logs",
            report_to="none"
        )
    except TypeError:
        pass

    # 2) evaluate_during_training vs. save_steps vs. eval_steps dene
    try:
        return TrainingArguments(
            **base_kwargs,
            save_steps=500,
            eval_steps=500
            # eski sürümlerde evaluate_during_training varsa eklemek istersek:
            # evaluate_during_training=True
        )
    except TypeError:
        pass

    # 3) En minimal fallback (parametre yoksa)
    return TrainingArguments(**base_kwargs)

training_args = make_training_args()

data_collator = DataCollatorWithPadding(tokenizer)

# ----------------- 9. Trainer -----------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # eski sürüm evaluationStrategy yoksa da değer verilebilir
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ----------------- 10. Train -----------------
trainer.train()

# ----------------- 11. Validation -----------------
val_metrics = trainer.evaluate()
print("✅ Validation Metrics:", val_metrics)

# ----------------- 12. Test -----------------
test_metrics = trainer.evaluate(eval_dataset=test_dataset)
print("✅ Test Metrics:", test_metrics)

# ----------------- 13. Save -----------------
save_dir = "./deberta_intent_model"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Eğitim tamamlandı, en iyi model ve label encoder kaydedildi!")
