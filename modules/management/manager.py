"""
管理模块
提供实验记录、模型版本管理
"""
import os
import json
import shutil
from datetime import datetime


class Manager:
    def __init__(self, storage_dir="./storage"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.experiments_file = os.path.join(self.storage_dir, "experiments.json")
        self._init_experiments()

    def _init_experiments(self):
        if not os.path.exists(self.experiments_file):
            with open(self.experiments_file, 'w') as f:
                json.dump([], f)

    def log_experiment(self, experiment_data):
        experiments = self._load_experiments()
        experiment_data["timestamp"] = datetime.now().isoformat()
        experiments.append(experiment_data)
        self._save_experiments(experiments)

    def get_experiments(self):
        return self._load_experiments()

    def _load_experiments(self):
        with open(self.experiments_file, 'r') as f:
            return json.load(f)

    def _save_experiments(self, experiments):
        with open(self.experiments_file, 'w') as f:
            json.dump(experiments, f, indent=2)

    def version_model(self, model_path, version):
        version_dir = os.path.join(self.storage_dir, "models", version)
        os.makedirs(version_dir, exist_ok=True)
        if os.path.isdir(model_path):
            for item in os.listdir(model_path):
                src = os.path.join(model_path, item)
                dst = os.path.join(version_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        elif os.path.isfile(model_path):
            shutil.copy2(model_path, version_dir)

        version_info = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "source_path": model_path,
            "storage_path": version_dir
        }
        versions_file = os.path.join(self.storage_dir, "model_versions.json")
        versions = []
        if os.path.exists(versions_file):
            with open(versions_file, 'r') as f:
                versions = json.load(f)
        versions.append(version_info)
        with open(versions_file, 'w') as f:
            json.dump(versions, f, indent=2)
        return version_dir

    def get_model_versions(self):
        versions_file = os.path.join(self.storage_dir, "model_versions.json")
        if not os.path.exists(versions_file):
            return []
        with open(versions_file, 'r') as f:
            return json.load(f)

    def delete_model_version(self, version):
        versions = self.get_model_versions()
        updated = []
        for v in versions:
            if v["version"] == version:
                if os.path.exists(v["storage_path"]):
                    shutil.rmtree(v["storage_path"])
            else:
                updated.append(v)
        versions_file = os.path.join(self.storage_dir, "model_versions.json")
        with open(versions_file, 'w') as f:
            json.dump(updated, f, indent=2)


# ---------- 模块独立演示 ----------
if __name__ == "__main__":
    mgr = Manager()
    mgr.log_experiment({"model": "Qwen-0.5B", "lora_rank": 8, "status": "demo"})
    print("实验记录:", mgr.get_experiments())
    print("管理模块演示完成。")