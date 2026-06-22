"""
模型微调模块
提供大模型的LoRA微调功能，支持多种预训练模型
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

class Finetuner:
    """模型微调器类，负责模型的加载、配置和训练"""
    
    def __init__(self, model_name):
        """
        初始化微调器
        :param model_name: 预训练模型名称或路径
        """
        self.model_name = model_name
        self.accelerator = Accelerator()
    
    def load_model(self):
        """
        加载预训练模型和分词器
        :return: 模型和分词器
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    
    def setup_lora(self, model, lora_rank=8, lora_alpha=16):
        """
        设置LoRA配置并应用到模型
        :param model: 预训练模型
        :param lora_rank: LoRA秩，控制参数量
        :param lora_alpha: LoRA缩放因子
        :return: 配置了LoRA的模型
        """
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        return get_peft_model(model, lora_config)
    
    def train(self, model, tokenizer, train_data, val_data, epochs=3, learning_rate=1e-4, batch_size=2):
        """
        训练模型
        :param model: 配置了LoRA的模型
        :param tokenizer: 分词器
        :param train_data: 训练数据列表
        :param val_data: 验证数据列表（暂未使用）
        :param epochs: 训练轮数
        :param learning_rate: 学习率
        :param batch_size: 批次大小
        """
        from torch.utils.data import Dataset, DataLoader
        from transformers import get_linear_schedule_with_warmup
        
        # 定义数据集类
        class TextDataset(Dataset):
            """文本数据集类，用于模型训练"""
            
            def __init__(self, data, tokenizer, max_length=256):
                """
                初始化数据集
                :param data: 数据列表
                :param tokenizer: 分词器
                :param max_length: 最大序列长度
                """
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                """返回数据集大小"""
                return len(self.data)
            
            def __getitem__(self, idx):
                """
                获取单个数据样本
                :param idx: 数据索引
                :return: 编码后的数据字典
                """
                item = self.data[idx]
                system_msg = item.get("system", "") or ""
                conversations = item.get("conversation", [])

                if not isinstance(conversations, list):
                    conversations = []

                text = f"<|im_start|>system\n{system_msg}<|im_end|>\n"

                for conv in conversations:
                    if not isinstance(conv, dict):
                        continue
                    human = conv.get("human", "")
                    assistant = conv.get("assistant", "")
                    if human:
                        text += f"<|im_start|>user\n{human}<|im_end|>\n"
                    if assistant:
                        text += f"<|im_start|>assistant\n{assistant}<|im_end|>\n"

                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                labels = encoding['input_ids'].clone()

                return {
                    "input_ids": encoding['input_ids'].flatten(),
                    "attention_mask": encoding['attention_mask'].flatten(),
                    "labels": labels.flatten()
                }
        
        # 创建数据集和数据加载器
        train_dataset = TextDataset(train_data, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 配置优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # 配置学习率调度器
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps * 0.1,
            num_training_steps=total_steps
        )
        
        # 使用accelerator准备模型和数据
        model, optimizer, train_dataloader = self.accelerator.prepare(
            model, optimizer, train_dataloader
        )
        
        # 训练循环
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # 前向传播
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                
                # 反向传播
                self.accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # 打印训练进度
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def save_model(self, model, save_path):
        """
        保存微调后的模型
        :param model: 训练完成的模型
        :param save_path: 保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)



# ---------- 模块独立演示（需有 GPU 和预训练模型） ----------
if __name__ == "__main__":
    print("微调模块演示：请确保已安装 transformers 且有 GPU 环境。")
    import sys
    from pathlib import Path

    # 获取根目录
    root_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(root_dir))
    from config import Config
    from modules.data.data_processor import DataProcessor
    # 此处仅展示初始化流程，实际训练需要真实数据
    finetuner = Finetuner(Config.DEFAULT_MODEL)
    print("模型名称:", finetuner.model_name)
    model, tokenizer = finetuner.load_model()
    model = finetuner.setup_lora(model)
    train_path = os.path.join(Config.DATA_DIR, "test.jsonl")
    processor = DataProcessor()
    train_data = processor.load_data(train_path)
    train_data = processor.clean_data(train_data)
    finetuner.train(
        model, tokenizer, train_data,
        val_data=[],
        epochs=Config.DEFAULT_EPOCHS,
        learning_rate=Config.DEFAULT_LEARNING_RATE
    )
    # 构建绝对路径
    save_path = root_dir / "models" / f"{os.path.basename(Config.DEFAULT_MODEL)}_finetuned"
    finetuner.save_model(model, str(save_path))
    print("微调完成。")

