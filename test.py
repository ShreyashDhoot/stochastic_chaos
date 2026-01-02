import torch
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from datasets import load_dataset
import matplotlib.pyplot as plt


model_name = "meta-llama/Meta-Llama-3-8B"

ID_TO_LABEL = {
    0: "negative",
    1: "positive",
    2: "neutral",
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="./offload",
        offload_state_dict=True
    )
    model.generation_config.do_sample = True
    model.eval()
    return model, tokenizer


def build_fewshot_prompt(demos, query_text, id_to_label):
    prompt = ""
    for text, label_id in demos:
        prompt += f"Text: {text}\nLabel: {id_to_label[label_id]}\n\n"
    prompt += f"Text: {query_text}\nLabel:"
    return prompt


def extract_label_id(generated_text, label_to_id):
    text = generated_text.lower()
    if "label:" in text:
        text = text.split("label:")[-1]

    for label_str, label_id in label_to_id.items():
        if label_str in text:
            return label_id
    return None


@torch.no_grad()
def greedy_decode(model, tokenizer, prompt, max_new_tokens=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@torch.no_grad()
def stochastic_decode(
    model,
    tokenizer,
    prompt,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=5,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


MAX_NEW_TOKENS = 5
TOP_P = 0.9
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
TEMPS = np.linspace(0.05, 1.0, 20)

def evaluate_example(
    model,
    tokenizer,
    demos,
    x_i,
    y_i,
    id_to_label,
    label_to_id,
    k,
    temperature
):

    prompt = build_fewshot_prompt(demos, x_i, id_to_label)

   
    greedy_text = greedy_decode(model, tokenizer, prompt)
    greedy_pred = extract_label_id(greedy_text, label_to_id)
    greedy_correct = (greedy_pred == y_i)

    sampled_preds = []
    for _ in range(k):
        text = stochastic_decode(
            model,
            tokenizer,
            prompt,
            temperature=temperature,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS
        )

        pred = extract_label_id(text, label_to_id)
        if pred is not None:
            sampled_preds.append(pred)

    if len(sampled_preds) == 0:
        best_correct = False
    else:
        majority_pred = Counter(sampled_preds).most_common(1)[0][0]
        best_correct = (majority_pred == y_i)

    return {
        "greedy_correct": greedy_correct,
        "best_correct": best_correct,
        "sampled_preds": sampled_preds
    }



def sample_demos(train_set, k_shot, seed=0):
    random.seed(seed)
    idxs = random.sample(range(len(train_set)), k_shot)
    return [
        (train_set[i]["text"], train_set[i]["label"])
        for i in idxs
    ]

dataset=load_dataset("unswnlporg/BESSTIE")

train_set = dataset["train"]
eval_set  = dataset["validation"]

eval_subset = eval_set.select(range(100))
demos = sample_demos(train_set, k_shot=8)


T = 0.7
k = 16
model, tokenizer = load_model(model_name)

correct = []
for ex in eval_subset:
    r = evaluate_example(
        model,
        tokenizer,
        demos,
        ex["text"],
        ex["label"],
        ID_TO_LABEL,
        LABEL_TO_ID,
        k=k,
        temperature=T
    )
    correct.append(r["best_correct"])

print("Acc:", np.mean(correct))


greedy_correct = []

for ex in eval_set:
    r = evaluate_example(
        model,
        tokenizer,
        demos,
        ex["text"],
        ex["label"],
        ID_TO_LABEL,
        LABEL_TO_ID,
        k=1,
        temperature=0.7
    )
    greedy_correct.append(r["greedy_correct"])

acc_greedy = np.mean(greedy_correct)
print("Greedy accuracy:", acc_greedy)


EG = np.zeros((len(TEMPS), len(K_VALUES)))

for i, T in enumerate(TEMPS):
    print(f"Temperature {T:.2f}")

    for j, k in enumerate(K_VALUES):
        correct = []

        for ex in eval_subset:
            r = evaluate_example(
                model,
                tokenizer,
                demos,
                ex["text"],
                ex["label"],
                ID_TO_LABEL,
                LABEL_TO_ID,
                k=k,
                temperature=T
            )
            correct.append(r["best_correct"])

        acc_best_k = np.mean(correct)
        EG[i, j] = acc_best_k - acc_greedy

        print(f"  k={k:>2}, ΔAcc={EG[i,j]:.4f}")


plt.figure(figsize=(7, 5))

plt.imshow(
    EG,
    origin="lower",
    aspect="auto",
    extent=[
        np.log2(K_VALUES[0]),
        np.log2(K_VALUES[-1]),
        TEMPS[0],
        TEMPS[-1]
    ],
    vmin=0,
    vmax=0.25,
    cmap="viridis"
)

plt.colorbar(label="Exploration Gain (ΔAcc)")
plt.xlabel("log₂ k")
plt.ylabel("Temperature T")
plt.title("Exploration–ICL Landscape (Vicuna on BESSTIE)")

plt.show()