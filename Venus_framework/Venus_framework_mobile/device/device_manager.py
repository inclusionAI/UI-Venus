from typing import Dict, Optional
from .adb_controller import ADBController

class DeviceManager:
    def __init__(self):
        self.devices: Dict[str, ADBController] = {}
        
    def connect_device(self, device_id: str) -> bool:
        """Connect a device.
        
        Args:
            device_id: Device ID (IP:port).
            
        Returns:
            Whether the connection succeeded.
        """
        try:
            controller = ADBController(device_id)
            self.devices[device_id] = controller
            return True
        except ConnectionError:
            return False
            
    def get_device(self, device_id: str) -> Optional[ADBController]:
        """Get a device controller.
        
        Args:
            device_id: Device ID.
            
        Returns:
            A device controller instance, or None if it does not exist.
        """
        return self.devices.get(device_id)
        
    def disconnect_device(self, device_id: str) -> bool:
        """Disconnect a device.
        
        Args:
            device_id: Device ID.
            
        Returns:
            Whether the operation succeeded.
        """
        if device_id in self.devices:
            del self.devices[device_id]
            return True
        return False
