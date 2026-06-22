import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

class Config:
    # 数据配置
    DATA_DIR = os.environ.get("DATA_DIR", "data")

    # 模型配置 - 优先使用环境变量，其次使用已微调的本地模型
    DEFAULT_MODEL = os.environ.get(
        "DEFAULT_MODEL",
        os.path.join(os.path.dirname(__file__), "models", "Qwen2___5-0___5B-Instruct")
    )

    # 微调配置
    DEFAULT_LORA_RANK = int(os.environ.get("DEFAULT_LORA_RANK", 8))
    DEFAULT_EPOCHS = int(os.environ.get("DEFAULT_EPOCHS", 3))
    DEFAULT_LEARNING_RATE = float(os.environ.get("DEFAULT_LEARNING_RATE", 1e-4))

    # 量化配置
    DEFAULT_PRECISION = os.environ.get("DEFAULT_PRECISION", "int4")

    # 部署配置
    DEFAULT_IMAGE_NAME = os.environ.get("DEFAULT_IMAGE_NAME", "lora-finetuned-model")

    # API配置
    API_HOST = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT = int(os.environ.get("API_PORT", 8000))

    # UI配置
    UI_HOST = os.environ.get("UI_HOST", "127.0.0.1")
    UI_PORT = int(os.environ.get("UI_PORT", 7860))