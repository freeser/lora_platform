"""
大模型LoRA微调与量化部署平台 - 用户界面
基于 Gradio 的可视化操作界面
"""
import gradio as gr
import requests
import os
import sys

# 添加根路径到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config



API_URL = "http://localhost:8000"


def upload_and_process(file):
    if file is None:
        return "请上传文件"
    with open(file.name, "rb") as f:
        files = {"file": (os.path.basename(file.name), f)}
        resp = requests.post(f"{API_URL}/upload-data", files=files)
    if resp.status_code != 200:
        return f"上传失败: {resp.text}"
    file_path = resp.json()["file_path"]
    resp2 = requests.post(f"{API_URL}/process-data", data={"file_path": file_path})
    if resp2.status_code != 200:
        return f"处理失败: {resp2.text}"
    result = resp2.json()
    return f"数据处理成功！\n训练集: {result['train_path']}\n验证集: {result['val_path']}"


def finetune_model(model_name, train_path, lora_rank, epochs, lr):
    resp = requests.post(f"{API_URL}/finetune", data={
        "model_name": model_name,
        "train_path": train_path,
        "lora_rank": lora_rank,
        "epochs": epochs,
        "learning_rate": lr
    })
    if resp.status_code != 200:
        return f"微调失败: {resp.text}"
    return f"微调成功！模型路径: {resp.json()['model_path']}"


def quantize_model(model_path, precision):
    resp = requests.post(f"{API_URL}/quantize", data={
        "model_path": model_path,
        "precision": precision
    })
    if resp.status_code != 200:
        return f"量化失败: {resp.text}"
    return f"量化成功！输出路径: {resp.json()['quantized_path']}"


def deploy_model(model_path, image_name):
    resp = requests.post(f"{API_URL}/deploy", data={
        "model_path": model_path,
        "image_name": image_name
    })
    if resp.status_code != 200:
        return f"部署失败: {resp.text}"
    return f"部署成功！镜像: {resp.json()['image_name']}"


def get_experiments():
    resp = requests.get(f"{API_URL}/experiments")
    if resp.status_code != 200:
        return "获取失败"
    exps = resp.json()
    if not exps:
        return "暂无实验记录"
    output = ""
    for e in exps:
        output += f"时间: {e.get('timestamp')}\n模型: {e.get('model')}\n状态: {e.get('status')}\n---\n"
    return output


with gr.Blocks(title="LoRA微调与量化部署平台") as demo:
    gr.Markdown("# 大模型LoRA微调与量化部署平台")

    with gr.Tab("数据处理"):
        file_inp = gr.File(label="上传数据文件 (JSON/CSV)")
        btn_process = gr.Button("处理数据")
        out_process = gr.Textbox(label="处理结果")
        btn_process.click(upload_and_process, inputs=[file_inp], outputs=[out_process])

    with gr.Tab("模型微调"):
        model_name = gr.Textbox(label="模型名称", value=Config.DEFAULT_MODEL)
        train_path = gr.Textbox(label="训练数据路径")
        lora_rank = gr.Slider(1, 64, value=8, label="LoRA秩")
        epochs = gr.Slider(1, 10, value=3, label="训练轮数")
        lr = gr.Number(value=1e-4, label="学习率")
        btn_finetune = gr.Button("开始微调")
        out_finetune = gr.Textbox(label="微调结果")
        btn_finetune.click(finetune_model, [model_name, train_path, lora_rank, epochs, lr], [out_finetune])

    with gr.Tab("模型量化"):
        model_path = gr.Textbox(label="模型路径")
        precision = gr.Dropdown(["int4", "int8", "fp16"], value="int4", label="量化精度")
        btn_quant = gr.Button("开始量化")
        out_quant = gr.Textbox(label="量化结果")
        btn_quant.click(quantize_model, [model_path, precision], [out_quant])

    with gr.Tab("模型部署"):
        dep_path = gr.Textbox(label="模型路径")
        img_name = gr.Textbox(label="Docker镜像名称")
        btn_deploy = gr.Button("开始部署")
        out_deploy = gr.Textbox(label="部署结果")
        btn_deploy.click(deploy_model, [dep_path, img_name], [out_deploy])

    with gr.Tab("实验记录"):
        btn_exp = gr.Button("获取实验记录")
        out_exp = gr.Textbox(label="记录")
        btn_exp.click(get_experiments, outputs=[out_exp])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7864, share=False)