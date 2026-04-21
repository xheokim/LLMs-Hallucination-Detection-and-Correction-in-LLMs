'''1. 环境与全局设置
2. 加载 HaluEval 数据集
3. 4bit 量化配置（解决显存报错）
4. 加载 Qwen7B 开源模型
5. Grok API 配置
6. 批量评测 + 输出幻觉率 + 保存CSV'''
# ==============================
# Import required libraries / 导入所需库
# ==============================
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI
import json
import random
import os

# ==============================
# Force use GPU (CUDA) / 强制使用GPU
# ==============================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==============================
# Global Experiment Settings / 全局实验设置
# ==============================
GROK_API_KEY = "xai-SC1************"
TEMPERATURE = 0
MAX_NEW_TOKENS = 128
SAMPLE_COUNT = 300

# ==============================
# Load HaluEval Dataset from local / 本地加载HaluEval数据集
# ==============================
test_data = []
with open("data/qa_data.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        test_data.append({
            "question": item["question"],
            "correct_answer": item["right_answer"]
        })

test_data = random.sample(test_data, SAMPLE_COUNT)
print(f"Successfully loaded HaluEval dataset: {len(test_data)} samples.")

# ==============================
# 4-bit Quantization Config / 4位量化配置
# ==============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

# ==============================
# Load Open-Source Model: Qwen2.5-7B-Instruct / 加载开源模型
# ==============================
print("\nLoading open-source model: Qwen2.5-7B-Instruct")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True
)

# ==============================
# Grok API Client / Grok API客户端
# ==============================
client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1"
)

# ==============================
# Model Inference Functions / 模型推理函数
# ==============================
def infer_qwen(prompt):
    messages = [{"role": "user", "content": prompt}]


    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def infer_grok3_mini(prompt):
    completion = client.chat.completions.create(
        model="grok-3-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE
    )
    return completion.choices[0].message.content.strip()

# ==============================
# Hallucination Detection / 幻觉检测函数
# ==============================
def is_answer_correct(model_answer, ground_truth):
    if not model_answer:
        return False
    return ground_truth.lower() in model_answer.lower()

# ==============================
# Run Experiment / 执行实验
# ==============================
results = []
print("\nStarting evaluation...")

for item in tqdm(test_data):
    question = item["question"]
    gt = item["correct_answer"]

    ans_qwen = infer_qwen(question)
    ans_grok = infer_grok3_mini(question)

    correct_qwen = is_answer_correct(ans_qwen, gt)
    correct_grok = is_answer_correct(ans_grok, gt)

    results.append([
        question, gt,
        ans_qwen, ans_grok,
        correct_qwen, correct_grok
    ])

# ==============================
# Save Results to CSV / 保存结果到CSV
# ==============================
df = pd.DataFrame(results, columns=[
    "question",
    "ground_truth",
    "qwen7b_answer",
    "grok3mini_answer",
    "qwen7b_correct",
    "grok3mini_correct"
])

# ==============================
# Calculate Metrics / 计算指标
# ==============================
acc_qwen = df["qwen7b_correct"].mean()
acc_grok = df["grok3mini_correct"].mean()
hallucination_qwen = 1 - acc_qwen
hallucination_grok = 1 - acc_grok

# ==============================
# Print Final Results (English Only) / 输出最终结果（仅英文）
# ==============================
print("\n============================================================")
print("Experiment Result: Qwen2.5-7B vs Grok 3 Mini (Hallucination Rate)")
print("============================================================")
print(f"Qwen2.5-7B    Accuracy: {acc_qwen:.2%}   Hallucination Rate: {hallucination_qwen:.2%}")
print(f"Grok 3 Mini   Accuracy: {acc_grok:.2%}   Hallucination Rate: {hallucination_grok:.2%}")
print("============================================================")

df.to_csv("hallucination_experiment_result.csv", index=False, encoding="utf-8-sig")
print("\nExperiment finished. Results saved to hallucination_experiment_result.csv")