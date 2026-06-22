"""
大模型LoRA微调与量化部署平台 - API服务
整合所有模块，对外提供 RESTful 接口
"""
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from config import Config
from modules.data.data_processor import DataProcessor
from modules.finetune.finetuner import Finetuner
from modules.quantization.quantizer import Quantizer
from modules.deployment.deployer import Deployer
from modules.management.manager import Manager

app = FastAPI(title="大模型LoRA微调与量化部署平台", version="1.0.0")

processor = DataProcessor(Config.DATA_DIR)
manager = Manager()


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    file_path = os.path.join(Config.DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return JSONResponse({"message": "文件上传成功", "file_path": file_path})


@app.post("/process-data")
async def process_data(file_path: str = Form(...)):
    data = processor.load_data(file_path)
    cleaned = processor.clean_data(data)
    train, val = processor.split_data(cleaned)

    base, ext = os.path.splitext(file_path)
    train_path = f"{base}_train{ext}"
    val_path = f"{base}_val{ext}"
    processor.save_data(train, train_path)
    processor.save_data(val, val_path)

    return JSONResponse({
        "message": "数据处理成功",
        "train_path": train_path,
        "val_path": val_path
    })


@app.post("/finetune")
async def finetune(
    model_name: str = Form(Config.DEFAULT_MODEL),
    train_path: str = Form(...),
    lora_rank: int = Form(Config.DEFAULT_LORA_RANK),
    epochs: int = Form(Config.DEFAULT_EPOCHS),
    learning_rate: float = Form(Config.DEFAULT_LEARNING_RATE)
):
    finetuner = Finetuner(model_name)
    model, tokenizer = finetuner.load_model()
    model = finetuner.setup_lora(model, lora_rank)

    train_data = processor.load_data(train_path)
    train_data = processor.clean_data(train_data)

    finetuner.train(
        model, tokenizer, train_data,
        val_data=[],
        epochs=epochs,
        learning_rate=learning_rate
    )

    save_path = f"models/{os.path.basename(model_name)}_finetuned"
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    finetuner.save_model(model, save_path)

    manager.log_experiment({
        "model": model_name,
        "lora_rank": lora_rank,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "status": "completed"
    })

    return JSONResponse({"message": "模型微调成功", "model_path": save_path})


@app.post("/quantize")
async def quantize(
    model_path: str = Form(...),
    precision: str = Form(Config.DEFAULT_PRECISION)
):
    quantizer = Quantizer()
    output_path = f"{model_path}_{precision}"
    quantizer.quantize_model(model_path, output_path, precision)
    return JSONResponse({"message": "模型量化成功", "quantized_path": output_path})


@app.post("/deploy")
async def deploy(
    model_path: str = Form(...),
    image_name: str = Form(Config.DEFAULT_IMAGE_NAME)
):
    deployer = Deployer()
    deployer.create_docker_image(model_path, image_name)
    deployer.generate_api_template("api")
    return JSONResponse({"message": "模型部署成功", "image_name": image_name})


@app.get("/experiments")
async def get_experiments():
    return JSONResponse(manager.get_experiments())


@app.get("/")
async def root():
    return {
        "message": "大模型LoRA微调与量化部署平台API",
        "version": "1.0.0",
        "endpoints": [
            "/upload-data",
            "/process-data",
            "/finetune",
            "/quantize",
            "/deploy",
            "/experiments"
        ]
    }