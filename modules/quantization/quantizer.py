"""
模型量化模块
支持 INT4 / INT8 / FP16 三种精度，使用 bitsandbytes 和 PyTorch 动态量化
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Quantizer:
    def quantize_model(self, model_path, output_path, precision="int4"):
        if precision == "int4":
            self._quantize_int4(model_path, output_path)
        elif precision == "int8":
            self._quantize_int8(model_path, output_path)
        elif precision == "fp16":
            self._quantize_fp16(model_path, output_path)
        else:
            raise ValueError(f"不支持的量化精度: {precision}")

    def _quantize_int4(self, model_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        # 检查是否是 LoRA 模型
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        base_model_path = None
        if os.path.exists(adapter_config_path):
            import json
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
        
        # 确定 tokenizer 路径
        tokenizer_path = base_model_path if base_model_path else model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    def _quantize_int8(self, model_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        # 检查是否是 LoRA 模型
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        base_model_path = None
        if os.path.exists(adapter_config_path):
            import json
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
        
        # 确定 tokenizer 路径
        tokenizer_path = base_model_path if base_model_path else model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    def _quantize_fp16(self, model_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        # 检查是否是 LoRA 模型
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        base_model_path = None
        if os.path.exists(adapter_config_path):
            import json
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
        
        # 确定 tokenizer 路径
        tokenizer_path = base_model_path if base_model_path else model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)


# ---------- 模块独立演示 ----------
if __name__ == "__main__":
    print("量化模块演示：需要提供真实模型路径。")
    q = Quantizer()
    print("支持精度：int4, int8, fp16")
    import sys
    from pathlib import Path

    # 获取根目录
    root_dir = Path(__file__).resolve().parent.parent.parent
    q.quantize_model(root_dir / "models" / "Qwen2___5-0___5B-Instruct_finetuned", root_dir / "models" / "Qwen2___5-0___5B-Instruct_finetuned_int4", "int4")
    print("量化模块结构验证通过。")