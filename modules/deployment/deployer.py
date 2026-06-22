"""
模型部署模块
生成 Docker 镜像和 FastAPI 服务模板
"""
import os
import docker


class Deployer:
    def __init__(self):
        self.client = docker.from_env()

    def create_docker_image(self, model_path, image_name):
        abs_model_path = os.path.abspath(model_path)
        model_dir = os.path.dirname(abs_model_path)
        model_name_in_container = os.path.basename(abs_model_path)
        dockerfile_content = f"""
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY {model_dir} /app/model
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
        self.client.images.build(path=".", tag=image_name)

    def generate_api_template(self, output_path):
        api_template = """
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained("/app/model")
    tokenizer = AutoTokenizer.from_pretrained("/app/model")

@app.post("/generate")
async def generate(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return {"generated": tokenizer.decode(outputs[0])}
"""
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "main.py"), "w") as f:
            f.write(api_template)


# ---------- 模块独立演示 ----------
if __name__ == "__main__":
    print("部署模块演示：需要安装 Docker 且有 Docker 环境。")
    d = Deployer()
    # d.create_docker_image("models/some_model", "test_image")
    d.generate_api_template("api_template")
    print("已生成 API 模板文件到 api_template/main.py")