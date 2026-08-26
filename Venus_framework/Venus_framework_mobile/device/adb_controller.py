from typing import Optional
import base64
import subprocess
import logging
import time


class ADBController:
    """Wrap Android device operations exposed through ADB."""
    
    def __init__(self, device_id: str = None):
        """Initialize the ADB controller.
        
        Args:
            device_id: Device ID as an IP:port address or ADB serial number.
        """
        self.device_id = device_id
        self.logger = logging.getLogger(__name__)
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check the device connection."""
        try:
            if ':' in self.device_id:
                result = subprocess.run(
                    ['adb', 'connect', self.device_id],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    message = result.stderr.strip() or result.stdout.strip()
                    raise ConnectionError(f"设备连接失败: {message}")
            state = subprocess.run(
                ['adb', '-s', self.device_id, 'get-state'],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if state.returncode != 0 or state.stdout.strip() != 'device':
                message = state.stderr.strip() or state.stdout.strip()
                raise ConnectionError(f"设备不可用: {message}")
            self.logger.info(f"设备连接成功: {self.device_id}")
            return True
        except subprocess.TimeoutExpired:
            raise ConnectionError(f"设备连接超时: {self.device_id}")
        except Exception as e:
            raise ConnectionError(f"ADB 连接失败: {str(e)}") from e

    def _execute_adb_command(self, cmd: list, operation: str, retry: int = 2) -> bool:
        """Run an ADB command with retries.
        
        Args:
            cmd: ADB command arguments.
            operation: Operation description.
            retry: Number of retries.
            
        Returns:
            Whether the operation succeeded.
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
        """Tap a screen position.
        
        Args:
            x: Horizontal coordinate.
            y: Vertical coordinate.
            
        Returns:
            Whether the operation succeeded.
        """
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'tap', str(x), str(y)])
        return self._execute_adb_command(cmd, f"点击({x},{y})")

    def double_tap(self, x: int, y: int) -> bool:
        if not self.tap(x, y):
            return False
        time.sleep(0.1)
        return self.tap(x, y)

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int = 1000) -> bool:
        """Perform a swipe gesture.
        
        Args:
            start_x: Start horizontal coordinate.
            start_y: Start vertical coordinate.
            end_x: End horizontal coordinate.
            end_y: End vertical coordinate.
            duration: Swipe duration in milliseconds.
            
        Returns:
            Whether the operation succeeded.
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
        """Clear the active input field."""
        cmd = ['adb', '-s', self.device_id, 'shell', 'am', 'broadcast', '-a', 'ADB_CLEAR_TEXT']
        subprocess.run(cmd, capture_output=True, text=True)

    def input_text(self, text: str, clear_first: bool = True) -> bool:
        """Enter text.
        
        Args:
            text: Text to enter.
            clear_first: Whether to clear the input field first. Defaults to True.
            
        Returns:
            Whether the operation succeeded.
        """
        try:
            if clear_first:
                self.clear_input_field()
            
            cmd = [
                'adb', '-s', self.device_id, 'shell', 'am', 'broadcast',
                '-a', 'ADB_INPUT_TEXT', '--es', 'msg', text,
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=5)
            self.logger.info(f"输入文本: {text}")
            return True
        except subprocess.TimeoutExpired:
            self.logger.error(f"输入文本超时: {text}")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"输入文本失败: {e}")
            return False

    def screenshot(self, retry: int = 2) -> Optional[str]:
        """Capture the screen and return a Base64-encoded image.
        
        Args:
            retry: Number of retries.
            
        Returns:
            A Base64-encoded image string, or None on failure.
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

    def get_screenshot_to_album(self) -> bool:
        directory_command = ['adb']
        if self.device_id:
            directory_command.extend(['-s', self.device_id])
        directory_command.extend(['shell', 'mkdir', '-p', '/sdcard/DCIM/Camera'])
        if not self._execute_adb_command(directory_command, '创建相册目录'):
            return False
        screenshot_path = f'/sdcard/DCIM/Camera/screenshot_{int(time.time())}.png'
        screenshot_command = ['adb']
        if self.device_id:
            screenshot_command.extend(['-s', self.device_id])
        screenshot_command.extend(['shell', 'screencap', '-p', screenshot_path])
        return self._execute_adb_command(screenshot_command, f'保存截图({screenshot_path})')
    
    def open_url(self, url: str) -> bool:
        """Open a URL.
        
        Args:
            url: URL to open.
        """
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'am', 'start', '-S', '-d', url])
        return self._execute_adb_command(cmd, f"打开URL({url})")
        
    def presshome(self) -> bool:
        """Return to the home screen."""
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'keyevent', '3'])
        return self._execute_adb_command(cmd, "按Home键")

    def pressback(self) -> bool:
        """Press the Back key."""
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'keyevent', 'KEYCODE_BACK'])
        return self._execute_adb_command(cmd, "按返回键")
        
    def longpress(self, x: int, y: int, duration: int = 500) -> bool:
        """Long-press a screen position.
        
        Args:
            x: Horizontal coordinate.
            y: Vertical coordinate.
            duration: Long-press duration in milliseconds.
        """
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'swipe', str(x), str(y), str(x), str(y), str(duration)])
        return self._execute_adb_command(cmd, f"长按({x},{y})")
            
    def pressmenu(self) -> bool:
        """Press the recent-apps key."""
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'keyevent', 'KEYCODE_APP_SWITCH'])
        return self._execute_adb_command(cmd, "按最近应用键")

    def pressenter(self) -> bool:
        """Press the Enter key."""
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'keyevent', 'KEYCODE_ENTER'])
        return self._execute_adb_command(cmd, "按回车键")

    def launch_app(self, package_or_activity: str) -> bool:
        """Launch an application.
        
        Args:
            package_or_activity: Package name or package/activity name.
                - First try Monkey with only the package name.
                - If that fails and activity data is available, try ``am start``.
        """
        # Extract the package name before the slash.
        if '/' in package_or_activity:
            package_name = package_or_activity.split('/')[0]
        else:
            package_name = package_or_activity
        
        # Try to launch the package with Monkey first.
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend([
            "shell", "monkey", "-p", package_name,
            "-c", "android.intent.category.LAUNCHER", "1"
        ])
        
        if self._execute_adb_command(cmd, f"启动应用(monkey: {package_name})"):
            return True
        
        # If Monkey fails and a full activity is available, try am start.
        if '/' in package_or_activity:
            self.logger.info(f"monkey启动失败，尝试am start方式启动")
            cmd = ['adb']
            if self.device_id:
                cmd.extend(['-s', self.device_id])
            cmd.extend(['shell', 'am', 'start', '-n', package_or_activity])
            return self._execute_adb_command(cmd, f"启动应用(am start: {package_or_activity})")
        
        return False
