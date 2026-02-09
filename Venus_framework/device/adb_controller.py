from typing import Optional
import base64
import subprocess
import logging
import time


class ADBController:
    """ADB设备控制器，封装Android设备的各种操作"""
    
    def __init__(self, device_id: str = None):
        """初始化 ADB 控制器
        
        Args:
            device_id: 设备ID，格式为IP:端口或adb序列号
        """
        self.device_id = device_id
        self.logger = logging.getLogger(__name__)
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """检查设备连接状态"""
        try:
            cmd = ['adb', 'connect', self.device_id]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
            if "failed to connect" in result.stdout or "Network is unreachable" in result.stdout:
                self.logger.error(f"设备连接失败: {result.stdout}")
                raise ConnectionError(f"设备连接失败: {result.stdout}")
            self.logger.info(f"设备连接成功: {self.device_id}")
            return True
        except subprocess.TimeoutExpired:
            raise ConnectionError(f"设备连接超时: {self.device_id}")
        except Exception as e:
            raise ConnectionError(f"ADB 连接失败: {str(e)}") from e

    def _execute_adb_command(self, cmd: list, operation: str, retry: int = 2) -> bool:
        """执行ADB命令（带重试）
        
        Args:
            cmd: ADB命令列表
            operation: 操作描述
            retry: 重试次数
            
        Returns:
            操作是否成功
        """
        for attempt in range(retry + 1):
            try:
                self.logger.info(f"输入指令: {cmd}")
                result = subprocess.run(cmd, check=True, capture_output=True, timeout=10)
                return True
            except subprocess.TimeoutExpired:
                self.logger.warning(f"{operation} 超时 (尝试 {attempt + 1}/{retry + 1})")
                if attempt < retry:
                    time.sleep(1)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"{operation} 失败: {e.stderr.decode() if e.stderr else str(e)}")
                if attempt < retry:
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"{operation} 异常: {str(e)}")
                break
        return False

    def tap(self, x: int, y: int) -> bool:
        """点击屏幕指定位置
        
        Args:
            x: 横坐标
            y: 纵坐标
            
        Returns:
            操作是否成功
        """
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'tap', str(x), str(y)])
        return self._execute_adb_command(cmd, f"点击({x},{y})")

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int = 1000) -> bool:
        """滑动操作
        
        Args:
            start_x: 起始点横坐标
            start_y: 起始点纵坐标
            end_x: 终点横坐标
            end_y: 终点纵坐标
            duration: 滑动持续时间(ms)
            
        Returns:
            操作是否成功
        """
        dist_sq = (start_x - end_x) ** 2 + (start_y - end_y) ** 2
        duration_ms = int(dist_sq / 1000)
        duration = max(1000, min(duration_ms, 2000))
        
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'swipe', 
                   str(start_x), str(start_y), 
                   str(end_x), str(end_y), 
                   str(duration)])
        return self._execute_adb_command(cmd, f"滑动({start_x},{start_y})->({end_x},{end_y})")

    def clear_input_field(self) -> None:
        """清空当前输入框内容"""
        cmd = ['adb', '-s', self.device_id, 'shell', 'am', 'broadcast', '-a', 'ADB_CLEAR_TEXT']
        subprocess.run(cmd, capture_output=True, text=True)

    def input_text(self, text: str, clear_first: bool = True) -> bool:
        """输入文本
        
        Args:
            text: 要输入的文本
            clear_first: 是否先清空输入框（默认为 True）
            
        Returns:
            操作是否成功
        """
        try:
            if clear_first:
                self.clear_input_field()
            
            text_escaped = text.replace(' ', '\\ ')
            cmd = f'adb -s "{self.device_id}" shell am broadcast -a ADB_INPUT_TEXT --es msg "{text_escaped}"'
            subprocess.check_output(cmd, shell=True, timeout=5)
            self.logger.info(f"输入文本: {text}")
            return True
        except subprocess.TimeoutExpired:
            self.logger.error(f"输入文本超时: {text}")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"输入文本失败: {e}")
            return False

    def screenshot(self, retry: int = 2) -> Optional[str]:
        """获取屏幕截图并转换为base64编码
        
        Args:
            retry: 重试次数
            
        Returns:
            base64编码的图片字符串，失败返回None
        """
        for attempt in range(retry + 1):
            try:
                cmd = ['adb']
                if self.device_id:
                    cmd.extend(['-s', self.device_id])
                cmd.extend(['shell', 'screencap', '-p'])
                screenshot_bytes = subprocess.run(cmd, check=False, capture_output=True, timeout=30)
                if screenshot_bytes.returncode == 0 and screenshot_bytes.stdout:
                    return base64.b64encode(screenshot_bytes.stdout).decode('utf-8')
                self.logger.warning(f"截图失败 (尝试 {attempt + 1}/{retry + 1})")
                if attempt < retry:
                    time.sleep(1)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"截图超时 (尝试 {attempt + 1}/{retry + 1})")
                if attempt < retry:
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"截图异常: {str(e)}")
                break
        return None
    
    def open_url(self, url: str) -> bool:
        """打开指定URL
        
        Args:
            url: 要打开的URL
        """
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'am', 'start', '-S', '-d', url])
        return self._execute_adb_command(cmd, f"打开URL({url})")
        
    def presshome(self) -> bool:
        """返回主屏幕"""
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'keyevent', '3'])
        return self._execute_adb_command(cmd, "按Home键")

    def pressback(self) -> bool:
        """按下返回键"""
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'keyevent', 'KEYCODE_BACK'])
        return self._execute_adb_command(cmd, "按返回键")
        
    def longpress(self, x: int, y: int, duration: int = 500) -> bool:
        """长按屏幕指定位置
        
        Args:
            x: 横坐标
            y: 纵坐标
            duration: 长按持续时间(ms)
        """
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'swipe', str(x), str(y), str(x), str(y), str(duration)])
        return self._execute_adb_command(cmd, f"长按({x},{y})")
            
    def pressmenu(self) -> bool:
        """按下最近应用键"""
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'keyevent', 'KEYCODE_APP_SWITCH'])
        return self._execute_adb_command(cmd, "按最近应用键")

    def pressenter(self) -> bool:
        """按下回车键"""
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'keyevent', 'KEYCODE_ENTER'])
        return self._execute_adb_command(cmd, "按回车键")

    def launch_app(self, package_or_activity: str) -> bool:
        """启动指定应用
        
        Args:
            package_or_activity: 应用包名或包名/Activity名
                - 先尝试用monkey方式启动（只用包名）
                - 如果失败且有Activity信息，再用am start方式启动
        """
        # 提取包名（取 '/' 前面的部分）
        if '/' in package_or_activity:
            package_name = package_or_activity.split('/')[0]
        else:
            package_name = package_or_activity
        
        # 先尝试 monkey 方式启动
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend([
            "shell", "monkey", "-p", package_name,
            "-c", "android.intent.category.LAUNCHER", "1"
        ])
        
        if self._execute_adb_command(cmd, f"启动应用(monkey: {package_name})"):
            return True
        
        # monkey 方式失败，如果有完整的 Activity 信息，尝试 am start 方式
        if '/' in package_or_activity:
            self.logger.info(f"monkey启动失败，尝试am start方式启动")
            cmd = ['adb']
            if self.device_id:
                cmd.extend(['-s', self.device_id])
            cmd.extend(['shell', 'am', 'start', '-n', package_or_activity])
            return self._execute_adb_command(cmd, f"启动应用(am start: {package_or_activity})")
        
        return False