from typing import Dict, Optional
from .adb_controller import ADBController

class DeviceManager:
    def __init__(self):
        self.devices: Dict[str, ADBController] = {}
        
    def connect_device(self, device_id: str) -> bool:
        """连接设备
        
        Args:
            device_id: 设备ID (IP:端口)
            
        Returns:
            连接是否成功
        """
        try:
            controller = ADBController(device_id)
            self.devices[device_id] = controller
            return True
        except ConnectionError:
            return False
            
    def get_device(self, device_id: str) -> Optional[ADBController]:
        """获取设备控制器
        
        Args:
            device_id: 设备ID
            
        Returns:
            设备控制器实例，不存在返回None
        """
        return self.devices.get(device_id)
        
    def disconnect_device(self, device_id: str) -> bool:
        """断开设备连接
        
        Args:
            device_id: 设备ID
            
        Returns:
            操作是否成功
        """
        if device_id in self.devices:
            del self.devices[device_id]
            return True
        return False 