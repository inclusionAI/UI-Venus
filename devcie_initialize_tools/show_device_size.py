
from parallel_eval_config import AVD_IP_PORT_LIST
import subprocess

def show_device_size():
    for device_name in AVD_IP_PORT_LIST:
        print(f'{device_name}:')
        subprocess.run(['adb', '-s', device_name, 'connect', device_name])
        subprocess.run(['adb', '-s', device_name, 'shell', 'wm', 'size'])
        print('--------------------------------')

if __name__ == '__main__':
    show_device_size()