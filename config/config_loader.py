# config/config_loader.py
from pathlib import Path
import yaml
from typing import Any, List

class ConfigLoader:
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_file: str = 'venus_benchmark_settings.yaml'):
        if self._config is not None:
            return self._config
        
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not exist: {config_path}")
        
        with config_path.open('r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        return self._config
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"No attribute '{name}'")
        
        if self._config is None:
            self.load()
        
        return self._config.get(name)
    
    def __getitem__(self, key: str) -> Any:
        if self._config is None:
            self.load()
        return self._config.get(key)
    
    def get(self, key: str, default=None) -> Any:
        if self._config is None:
            self.load()
        
        if '.' not in key:
            return self._config.get(key, default)
        
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def get_apk_paths(self) -> List[str]:
        apk_root = Path(self.apk_root or '')
        return [str(apk_root / apk) for apk in (self.apk_files or [])]


config = ConfigLoader()
