import argparse
import yaml
import os
import subprocess
import time
import json
import logging
import sys
from datetime import datetime
from threading import Thread, Lock


class BatchRunner:
    """Run tasks concurrently across multiple devices."""
    
    def __init__(self, config_path="config/config_multi.yaml", purpose_file="data/purpose.txt",
                 trace_dir=None, reflection_enabled=False,
                 reflection_config="config/reflection.yaml"):
        """Initialize the batch runner.
        
        Args:
            config_path: Configuration file path.
            purpose_file: Task list file path.
        """
        self.logger = logging.getLogger(__name__)
        self.config = self.load_config(config_path)
        if trace_dir:
            self.config["trace_dir"] = trace_dir
        self.purpose_file = purpose_file
        self.devices = self.config.get("devices", [])
        self.device_status = [True] * len(self.devices)
        self.tasks = {}
        self.tasks_lock = Lock()
        self.total_tasks = 0
        self.processed = 0
        self.start_time = None
        self.reflection_enabled = reflection_enabled
        self.reflection_config = reflection_config
        self.purposes = self.load_purposes()

    def load_config(self, config_path: str) -> dict:
        """Load a YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields.
            if not config.get('devices'):
                raise ValueError("配置文件缺少 devices 字段")
            if not config.get('policy'):
                raise ValueError("配置文件缺少 policy 字段")
            
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")

    def load_purposes(self) -> list:
        """Load the task list."""
        if not os.path.exists(self.purpose_file):
            raise FileNotFoundError(f"任务文件不存在: {self.purpose_file}")
        with open(self.purpose_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
        
    def find_idle_device(self) -> int:
        """Find an idle device index.
        
        Returns:
            An idle device index, or -1 if all devices are busy.
        """
        with self.tasks_lock:
            return self.device_status.index(True) if True in self.device_status else -1

    def execute_task(self, task_id: int, purpose: str, device_index: int):
        """Execute one task.
        
        Args:
            task_id: Task ID.
            purpose: Task description.
            device_index: Device index.
        """
        device_address = self.devices[device_index]
        config = self.config
        single_task_config = config.get("single_task_config", "config/ui_venus_2_single.yaml")
        
        start_time = time.time()
        
        # Task output directory and log file.
        task_save_dir = os.path.join(config["record_config"]["save_dir"], f'task_{task_id}')
        task_log_file = os.path.join(task_save_dir, "task.log")
        os.makedirs(task_save_dir, exist_ok=True)

        cmd = [
            sys.executable, "main.py",
            "--config", single_task_config,
            "--purpose", purpose,
            "--device-id", device_address,
            "--step-limit", str(config["ep_config"]["step_limit"]),
            "--trace-dir", config["trace_dir"],
            "--log-file", task_log_file
        ]
        policy_params = config["policy"]["params"]
        if policy_params.get("model_host"):
            cmd += ["--model-host", policy_params["model_host"]]
        if policy_params.get("model_url"):
            cmd += ["--model-url", policy_params["model_url"]]
        if policy_params.get("model_name"):
            cmd += ["--model-name", policy_params["model_name"]]
        if self.reflection_enabled:
            cmd += ["--reflection", "--reflection-config", self.reflection_config]

        try:
            self.logger.info(f"任务 {task_id} 开始执行 (设备: {device_address})")
            self.logger.info(f"任务 {task_id} 目的: {purpose}")
            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            duration = time.time() - start_time
            
            # Copy the task log into the batch runner log.
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
        """Print the progress bar."""
        progress = (self.processed / self.total_tasks) * 100
        print(f"\r进度: [{int(progress):3}%] 已完成 {self.processed}/{self.total_tasks} 任务", end="")

    def generate_report(self, report_file: str = "batch_report.json"):
        """Generate an execution report.
        
        Args:
            report_file: Report file path.
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
        
        # Save the JSON report.
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print the summary.
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
        return failed == 0 and error == 0

    def run_all_tasks(self):
        """Start all tasks and wait for completion."""
        self.total_tasks = len(self.purposes)
        if self.total_tasks == 0:
            print("未发现待处理任务")
            return False
        
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
        
        # Generate the report.
        report_dir = self.config.get("record_config", {}).get("save_dir", "record/batch/")
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        return self.generate_report(report_file)


def setup_logging(log_file: str = "logs/batch_runner.log"):
    """Configure logging for both the console and a file."""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to prevent duplicate output.
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console output.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File output.
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量任务执行器")
    parser.add_argument("--config", default="config/config_multi.yaml", help="配置文件路径")
    parser.add_argument("--purpose-file", default="data/purpose.txt", help="任务文件路径")
    parser.add_argument("--trace-dir", help="轨迹保存目录")
    parser.add_argument("--reflection", action="store_true", help="启用实时反思监督")
    parser.add_argument("--reflection-config", default="config/reflection.yaml", help="反思监督配置文件路径")
    args = parser.parse_args()

    setup_logging("logs/batch_runner.log")
    
    try:
        executor = BatchRunner(
            config_path=args.config,
            purpose_file=args.purpose_file,
            trace_dir=args.trace_dir,
            reflection_enabled=args.reflection,
            reflection_config=args.reflection_config,
        )
        success = executor.run_all_tasks()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"批量任务执行失败: {e}", exc_info=True)
        sys.exit(1)
