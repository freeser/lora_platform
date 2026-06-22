"""
使用微调+量化后的模型进行多轮对话
需要GPU环境运行，支持INT4/INT8/FP16量化
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2___5-0___5B-Instruct"
FINETUNED_INT4_PATH = PROJECT_ROOT / "models" / "Qwen2___5-0___5B-Instruct_finetuned_int4"

def load_model_and_tokenizer():
    print("加载量化后的模型和分词器...")
    print(f"模型路径: {FINETUNED_INT4_PATH}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_INT4_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_INT4_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("模型加载完成！")
    base_model.eval()

    return base_model, tokenizer

def build_conversation_prompt(history):
    """构建多轮对话提示"""
    prompt = "<|im_start|>system\n你是一个名为沐雪的AI女孩子<|im_end|>\n"
    for item in history:
        role = item["role"]
        content = item["content"]
        if role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        else:
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def chat(model, tokenizer):
    """多轮对话函数"""
    print("\n" + "="*50)
    print("量化模型多轮对话 (输入 'quit' 退出)")
    print("="*50 + "\n")

    history = []

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == 'quit':
            print("对话结束，再见！")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        prompt = build_conversation_prompt(history)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = response.strip()

        print(f"助手: {response}")
        history.append({"role": "assistant", "content": response})

        print()

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    chat(model, tokenizer)
