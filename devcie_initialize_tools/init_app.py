import subprocess
from parallel_eval_config import AVD_IP_PORT_LIST, AVD_IP_PORT_LIST_2, AVD_IP_PORT_LIST_k8s
from android_world.env.setup_device import apps
import os

_APPS = (
    # keep-sorted start
    apps.AndroidWorldApp,
    apps.AudioRecorder,
    apps.CameraApp,
    apps.ChromeApp,
    apps.ClipperApp,
    apps.ClockApp,
    apps.ContactsApp,
    apps.DialerApp,
    apps.ExpenseApp,
    apps.FilesApp,
    apps.JoplinApp,
    apps.MarkorApp,
    apps.MiniWobApp,
    apps.OpenTracksApp,
    apps.OsmAndApp,
    apps.RecipeApp,
    apps.RetroMusicApp,
    apps.SettingsApp,
    apps.SimpleCalendarProApp,
    apps.SimpleDrawProApp,
    apps.SimpleGalleryProApp,
    apps.SimpleSMSMessengerApp,
    apps.TasksApp,
    apps.VlcApp,
    apps.Calcyou,
    apps.KeePassLibre,
    apps.Pomodoro,
    apps.GymRoutines,
    apps.LibreOfficeViewer,
    apps.Medilog,
    apps.LibreOffice,
    apps.FossifyGallery,
    apps.Flexify,
    apps.NewPipe,
    apps.AmazeFileManager,
    apps.Fitbook,
    apps.Zipxtract,
    apps.Calculator,
    # keep-sorted end
)

def install_app(device_name: str, app_path: str, adb_server_port: str):
    subprocess.run(['adb', '-P', adb_server_port, 'connect', device_name])
    subprocess.run(['adb', '-P', adb_server_port, '-s', device_name, 'install', app_path])
    subprocess.run(['adb', '-P', adb_server_port, 'disconnect', device_name])

if __name__ == '__main__':
    ip_list = AVD_IP_PORT_LIST_k8s
    apk_folder = '/Users/zengzhengwen/Documents/code/app_data'
    adb_server_port = '5038'
    for device_name in ip_list:
        print(f'process {device_name}:...')
        for app in _APPS:
            for apk_name in app.apk_names:
                print(f'{device_name} {app.app_name} {apk_name}')
                apk_path = os.path.join(apk_folder, apk_name)
                install_app(device_name, apk_path, adb_server_port)