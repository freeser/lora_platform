"""
数据处理器模块
负责数据集的加载、清洗、格式转换和划分
支持 JSON、JSONL 和 CSV 格式
"""
import os
import json
import pandas as pd
import ast
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext == ".jsonl":
            records = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError:
                            continue
            return records
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            records = []
            for _, row in df.iterrows():
                record = {"system": row.get("system", ""), "conversation": []}
                conv_str = row.get("conversation", "[]")
                if isinstance(conv_str, str):
                    try:
                        conv_list = ast.literal_eval(conv_str)
                        record["conversation"] = conv_list
                    except (ValueError, SyntaxError):
                        record["conversation"] = []
                elif isinstance(conv_str, list):
                    record["conversation"] = conv_str
                records.append(record)
            return records
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def clean_data(self, data):
        cleaned = []
        for item in data:
            if isinstance(item, dict):
                clean_item = {k: v for k, v in item.items() if v is not None and v != ''}
                if "conversation" in clean_item and isinstance(clean_item["conversation"], str):
                    try:
                        clean_item["conversation"] = ast.literal_eval(clean_item["conversation"])
                    except (ValueError, SyntaxError):
                        clean_item["conversation"] = []
                if not isinstance(clean_item.get("conversation", []), list):
                    clean_item["conversation"] = []
                cleaned.append(clean_item)
        return cleaned

    def split_data(self, data, train_ratio=0.8):
        train, val = train_test_split(data, test_size=1 - train_ratio, random_state=42)
        return train, val

    def save_data(self, data, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif ext == ".jsonl":
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        elif ext == ".csv":
            pd.DataFrame(data).to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            raise ValueError(f"不支持的文件格式: {ext}")


# ---------- 模块独立演示 ----------
if __name__ == "__main__":
    # 创建一些示例数据
    demo_data = [
        {"system": "你是个助手", "conversation": [{"human": "你好", "assistant": "你好！有什么可以帮你的吗？"}]},
        {"system": "", "conversation": [{"human": "天气如何", "assistant": "今天天气晴，适合出行。"}]},
        {"system": None, "conversation": ""},  # 会被清洗掉
    ]
    # 保存为 JSON 文件
    os.makedirs("data", exist_ok=True)
    with open("data/demo.json", "w", encoding="utf-8") as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=2)

    processor = DataProcessor()
    data = processor.load_data("data/demo.json")
    print("加载数据:", data)

    cleaned = processor.clean_data(data)
    print("清洗后:", cleaned)

    train, val = processor.split_data(cleaned, train_ratio=0.5)
    print("训练集:", train)
    print("验证集:", val)

    processor.save_data(train, "data/demo_train.json")
    processor.save_data(val, "data/demo_val.json")
    print("数据模块演示完成，文件已保存至 data 目录")