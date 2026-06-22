from modules.data.data_processor import DataProcessor

processor = DataProcessor()

# 加载 JSON
data_json = processor.load_data("data/demo.json")
print("JSON 加载结果：")
print(data_json)

# 加载 CSV
data_csv = processor.load_data("data/沐雪AI数据集.csv")
print("\nCSV 加载结果：")
print(data_csv[0:2])


# 构造脏数据
dirty_data = [
    {"system": "你是助手", "conversation": [{"human": "你好", "assistant": "你好！"}]},
    {"system": None, "conversation": ""},  # system 为 None，conversation 为空字符串
    {"invalid": "data"}  # 没有 conversation 字段
]

cleaned = processor.clean_data(dirty_data)
print("清洗后的数据：")
print(cleaned)


# 测试划分
test_data = [{"id": i} for i in range(10)]
train, val = processor.split_data(test_data)
print(f"训练集数量：{len(train)}")
print(f"验证集数量：{len(val)}")
print("训练集：", [item["id"] for item in train])
print("验证集：", [item["id"] for item in val])


# 保存数据
processor.save_data(train, "data/test_train.json")
processor.save_data(val, "data/test_val.json")

print("\n保存后的文件：")
import os
print(os.listdir("data/"))