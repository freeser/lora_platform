# 其中api服务
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 启动UI服务
python ui/app.py
