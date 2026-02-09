import yaml
import os
import subprocess
import time
import json
import logging
from datetime import datetime
from threading import Thread, Lock


class BatchRunner:
    """批量任务执行器 - 多设备并行执行任务"""
    
    def __init__(self, config_path="config/config_multi.yaml", purpose_file="data/purpose.txt"):
        """初始化批量执行器
        
        Args:
            config_path: 配置文件路径
            purpose_file: 任务列表文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.config = self.load_config(config_path)
        self.purpose_file = purpose_file
        self.devices = self.config.get("devices", [])
        self.device_status = [True] * len(self.devices)
        self.tasks = {}
        self.tasks_lock = Lock()
        self.total_tasks = 0
        self.processed = 0
        self.start_time = None
        self.purposes = self.load_purposes()

    def load_config(self, config_path: str) -> dict:
        """加载YAML配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证必要字段
            if not config.get('devices'):
                raise ValueError("配置文件缺少 devices 字段")
            if not config.get('policy'):
                raise ValueError("配置文件缺少 policy 字段")
            
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")

    def load_purposes(self) -> list:
        """加载任务列表"""
        if not os.path.exists(self.purpose_file):
            raise FileNotFoundError(f"任务文件不存在: {self.purpose_file}")
        with open(self.purpose_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
        
    def find_idle_device(self) -> int:
        """查找空闲设备索引
        
        Returns:
            空闲设备索引，无空闲设备返回-1
        """
        with self.tasks_lock:
            return self.device_status.index(True) if True in self.device_status else -1

    def execute_task(self, task_id: int, purpose: str, device_index: int):
        """执行单个任务
        
        Args:
            task_id: 任务ID
            purpose: 任务描述
            device_index: 设备索引
        """
        device_address = self.devices[device_index]
        config = self.config
        single_task_config = config.get("single_task_config", "config/ui_venus_single.yaml")
        
        start_time = time.time()
        
        # 任务保存目录和日志文件
        task_save_dir = os.path.join(config["record_config"]["save_dir"], f'task_{task_id}')
        task_log_file = os.path.join(task_save_dir, "task.log")
        os.makedirs(task_save_dir, exist_ok=True)

        cmd = [
            "python", "main.py",
            "--config", single_task_config,
            "--purpose", purpose,
            "--device-id", device_address,
            "--step-limit", str(config["ep_config"]["step_limit"]),
            "--model-host", config["policy"]["params"]["model_host"],
            "--model-name", config["policy"]["params"]["model_name"],
            "--save-dir", task_save_dir,
            "--trace-dir", config["trace_dir"],
            "--log-file", task_log_file
        ]

        try:
            self.logger.info(f"任务 {task_id} 开始执行 (设备: {device_address})")
            self.logger.info(f"任务 {task_id} 目的: {purpose}")
            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            duration = time.time() - start_time
            
            # 读取任务日志文件并输出到 batch_runner 日志
            if os.path.exists(task_log_file):
                with open(task_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.logger.info(f"[任务 {task_id}] {line}")

            with self.tasks_lock:
                self.tasks[task_id]["status"] = "completed" if process.returncode == 0 else "failed"
                self.tasks[task_id]["exit_code"] = process.returncode
                self.tasks[task_id]["duration"] = duration
                self.tasks[task_id]["end_time"] = datetime.now().isoformat()
                if process.returncode != 0:
                    self.tasks[task_id]["error"] = stderr[-500:] if stderr else "未知错误"
                    self.logger.error(f"任务 {task_id} 失败: {self.tasks[task_id]['error']}")
                else:
                    self.logger.info(f"任务 {task_id} 完成 (耗时: {duration:.1f}秒)")
                self.device_status[device_index] = True

        except Exception as e:
            duration = time.time() - start_time
            with self.tasks_lock:
                self.tasks[task_id]["status"] = "error"
                self.tasks[task_id]["error"] = str(e)
                self.tasks[task_id]["duration"] = duration
                self.tasks[task_id]["end_time"] = datetime.now().isoformat()
                self.device_status[device_index] = True
                self.logger.error(f"任务 {task_id} 异常: {str(e)}")

        with self.tasks_lock:
            self.processed += 1
            self.print_progress()

    def print_progress(self):
        """打印进度条"""
        progress = (self.processed / self.total_tasks) * 100
        print(f"\r进度: [{int(progress):3}%] 已完成 {self.processed}/{self.total_tasks} 任务", end="")

    def generate_report(self, report_file: str = "batch_report.json"):
        """生成执行报告
        
        Args:
            report_file: 报告文件路径
        """
        completed = sum(1 for t in self.tasks.values() if t["status"] == "completed")
        failed = sum(1 for t in self.tasks.values() if t["status"] == "failed")
        error = sum(1 for t in self.tasks.values() if t["status"] == "error")
        
        total_duration = time.time() - self.start_time
        
        report = {
            "summary": {
                "total_tasks": self.total_tasks,
                "completed": completed,
                "failed": failed,
                "error": error,
                "success_rate": f"{completed / self.total_tasks * 100:.1f}%",
                "total_duration": f"{total_duration:.1f}秒",
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat()
            },
            "tasks": self.tasks
        }
        
        # 保存JSON报告
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("批量任务执行报告")
        print("=" * 60)
        print(f"总任务数: {self.total_tasks}")
        print(f"✅ 成功: {completed}")
        print(f"❌ 失败: {failed}")
        print(f"⚠️  错误: {error}")
        print(f"成功率: {completed / self.total_tasks * 100:.1f}%")
        print(f"总耗时: {total_duration:.1f}秒")
        print(f"\n详细报告已保存至: {report_file}")
        print("=" * 60)

    def run_all_tasks(self):
        """启动所有任务并等待完成"""
        self.total_tasks = len(self.purposes)
        if self.total_tasks == 0:
            print("未发现待处理任务")
            return
        
        self.start_time = time.time()
        print(f"发现 {self.total_tasks} 个待处理任务，使用 {len(self.devices)} 个设备并行执行...")
        
        for task_id, purpose in enumerate(self.purposes):
            with self.tasks_lock:
                self.tasks[task_id] = {
                    "task_id": task_id,
                    "status": "pending",
                    "device": -1,
                    "purpose": purpose,
                    "start_time": None,
                    "end_time": None,
                    "duration": 0,
                    "exit_code": None,
                    "error": None
                }

        for task_id, purpose in enumerate(self.purposes):
            while True:
                device_index = self.find_idle_device()
                if device_index != -1:
                    with self.tasks_lock:
                        self.tasks[task_id]["status"] = "processing"
                        self.tasks[task_id]["device"] = device_index
                        self.tasks[task_id]["start_time"] = datetime.now().isoformat()
                        self.device_status[device_index] = False
                    Thread(target=self.execute_task, args=(task_id, purpose, device_index), daemon=True).start()
                    break
                time.sleep(1)

        while self.processed < self.total_tasks:
            time.sleep(1)

        print("\n所有任务执行完成！")
        
        # 生成报告
        report_dir = self.config.get("record_config", {}).get("save_dir", "record/batch/")
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.generate_report(report_file)


def setup_logging(log_file: str = "logs/batch_runner.log"):
    """配置日志，同时输出到控制台和文件"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handler，避免重复
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


if __name__ == "__main__":
    setup_logging("logs/batch_runner.log")
    
    try:
        executor = BatchRunner(config_path="config/config_multi.yaml", purpose_file="data/purpose.txt")
        executor.run_all_tasks()
    except Exception as e:
        logging.error(f"批量任务执行失败: {e}", exc_info=True)
