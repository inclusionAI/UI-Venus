from flask import Flask, render_template, send_from_directory, request
import os
import argparse
from pathlib import Path
from PIL import Image  

app = Flask(__name__)


parser = argparse.ArgumentParser(description='Android World visualization')
parser.add_argument('--path', type=str, default="", 
                    help='path of visualization files')
parser.add_argument('--port', type=int, default=5050, 
                    help='port')

args = parser.parse_args()

BASE_DIR = Path(args.path)

@app.route('/')
def index():
    tasks = []
    for entry in os.scandir(BASE_DIR):
        if entry.is_dir():
            task_dir = BASE_DIR / entry.name
            status_value, full_status = get_task_status(task_dir)

            emoji = "✅" if status_value == "1.0" or status_value == "1" else "❌"
            display_name = f"{entry.name} - {status_value} {emoji}"

            tasks.append({
                "original_name": entry.name,
                "display_name": display_name,
                "status_value": status_value,
                "full_status": full_status
            })

    tasks.sort(key=lambda x: x['original_name'])

    selected_original_name = request.args.get('task', tasks[0]['original_name'] if tasks else None)

    task_data = None
    if selected_original_name:
        task_data = prepare_task_data(selected_original_name)

    return render_template(
        'index.html',
        tasks=tasks,
        selected_original_name=selected_original_name,
        task_data=task_data
    )

def get_task_status(task_dir):
    """Retrieve the task's status value and return the entire content of the status file."""
    status_file = task_dir / "000000status.txt"
    full_status = "Status file not found."
    status_value = "?"

    try:
        with open(status_file, 'r') as f:
            full_status = f.read().strip()

            lines = full_status.split('\n')
            if lines:
                status_value = lines[0].strip()

    except FileNotFoundError:
        pass
    except Exception as e:
        full_status = f"Failed to read status file:{str(e)}"

    return status_value, full_status

@app.route('/images/<task>/<filename>')
def serve_image(task, filename):
    """extract images"""
    task_dir = BASE_DIR / task
    return send_from_directory(task_dir, filename)

def prepare_task_data(task_name):
    """prepare data"""
    task_dir = BASE_DIR / task_name

    goal = read_file(task_dir / "000000goal.txt")

    status_value, full_status = get_task_status(task_dir)

    steps = []
    step_files = []

    for entry in os.scandir(task_dir):
        if entry.name.startswith("000000"):
            continue
        step_files.append(entry.name)
    step_files.sort()

    step_groups = {}
    for filename in step_files:
        prefix = filename.split('_')[0]
        step_groups.setdefault(prefix, []).append(filename)

    for prefix, files in sorted(step_groups.items()):
        step = {
            'prefix': prefix,
            'image': next((f for f in files if f.endswith('_raw.jpg')), None),
            'thinking': next((f for f in files if f.endswith('_thinking.txt')), None),
            'tool_call': next((f for f in files if f.endswith('_tool_call.txt')), None),
            'conclusion': next((f for f in files if f.endswith('_conclusion.txt')), None),
        }

        if step['image']:
            image_path = task_dir / step['image']
            try:
                with Image.open(image_path) as img:
                    step['image_width'] = img.width  
                    step['image_height'] = img.height  
            except Exception as e:
                step['image_width'] = None
                step['image_height'] = None
        else:
            step['image_width'] = None
            step['image_height'] = None

        for file_type in ['thinking', 'tool_call', 'conclusion']:
            if step[file_type]:
                content = read_file(task_dir / step[file_type])
                step[file_type + '_content'] = content
                step['x'] = None
                step['y'] = None 
            else:
                step[file_type + '_content'] = "file not found"
                step['x'] = None
                step['y'] = None 

        steps.append(step)

    return {
        'name': task_name,
        'goal': goal,
        'status_value': status_value,
        'full_status': full_status,  
        'steps': steps
    }

def read_file(file_path, default="file not found"):
    """read file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except:
        return default

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=True)