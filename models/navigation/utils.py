import re
from typing import Optional, Tuple


USER_PROMPT = """
**You are a GUI Agent.**  
Your task is to analyze a given user task, review current screenshot and previous actions, and determine the next action to complete the task.

### User Task
{user_task}

### Previous Actions
{previous_actions}

### Available Actions
You may execute one of the following functions:
Click(box=(x1, y1))
Drag(start=(x1, y1), end=(x2, y2))
Scroll(start=(x1, y1), end=(x2, y2), direction='down/up/right/left')
Type(content='')
Launch(app='')
Wait()
Finished(content='')
CallUser(content='')
LongPress(box=(x1, y1))
PressBack()
PressHome()
PressEnter()
PressRecent()

### Instruction
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- For requests that are questions (or chat messages), remember to use the `CallUser` action to reply to user explicitly before finishing! Then, after you have replied, use the Finished action if the goal is achieved.
- Consider exploring the screen by using the `scroll` action with different directions to reveal additional content.
- To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.
- You first thinks about the reasoning process in the mind, then provide the action. The reasoning and action are enclosed in <think></think> and <action></action> tags respectively. After providing action, summarize your action in <conclusion></conclusion> tags
"""


def parse_coordinates(coord_str: str) -> Optional[Tuple[float, float]]:
    if not coord_str:
        return None, None

    coord_str_clean = coord_str.replace(" ", "")
    match = re.match(r"\(([\d.]+),([\d.]+)\)", coord_str_clean)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    match = re.match(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", coord_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    return None, None

def _split_parameters(params_str: str) -> list:
    param_parts = []
    current_part = ""
    
    in_quotes = False
    quote_char = None
    bracket_level = 0
    
    for char in params_str:
        if char in ['"', "'"] and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        
        elif not in_quotes:
            if char == '(':
                bracket_level += 1
            elif char == ')':
                bracket_level -= 1
            elif char == ',' and bracket_level == 0:
                param_parts.append(current_part.strip())
                current_part = ""
                continue
        
        current_part += char
    
    if current_part.strip():
        param_parts.append(current_part.strip())
    
    return param_parts

def parse_answer(action_str: str):
    pattern = r"^(\w+)\((.*)\)$"
    match = re.match(pattern, action_str.strip(), re.DOTALL)
    if not match:
        raise ValueError(f"Invalid action_str format: {action_str}")
    
    action_type = match.group(1)
    params_str = match.group(2).strip()
    params = {}
    
    if params_str:
        try:
            param_pairs = _split_parameters(params_str)
            
            for pair in param_pairs:
                if '=' in pair:
                    key, value = pair.split("=", 1)
                    value = value.strip("'").strip()
                    params[key.strip()] = value
                else:
                    params[pair.strip()] = None
        except Exception as e:
            print(f"Answer parse error: {e}")
    
    if action_type == 'Click':
        p_x, p_y = parse_coordinates(params.get("box", ""))
        if p_x is not None and p_y is not None:
            return 'Click', {'box': (p_x, p_y)}
        else:
            raise ValueError(f"action {action_type} Unknown click params: {repr(params)}")
    elif action_type == 'LongPress':
        p_x, p_y = parse_coordinates(params.get("box", ""))
        if p_x is not None and p_y is not None:
            return 'LongPress', {'box': (p_x, p_y)}
        else:
            raise ValueError(f"action {action_type} Unknown long press params: {repr(params)}")
    elif action_type == 'Drag':
        p_x, p_y = parse_coordinates(params.get("start", ""))
        e_x, e_y = parse_coordinates(params.get("end", ""))
        if p_x is not None and p_y is not None and e_x is not None and e_y is not None:
            return 'Drag', {'start': (p_x, p_y), 'end': (e_x, e_y)}
        else:
            raise ValueError(f"action {action_type} Unknown drag params: {repr(params)}")
    elif action_type == 'Scroll':
        p_x, p_y = parse_coordinates(params.get("start", ""))
        e_x, e_y = parse_coordinates(params.get("end", ""))
        if p_x is not None and p_y is not None and e_x is not None and e_y is not None:
            return 'Scroll', {'start': (p_x, p_y), 'end': (e_x, e_y), 'direction': ''}
        elif "direction" in params:
            direction = params.get("direction")
            return 'Scroll', {'start': (), 'end': (), 'direction': direction}
        else:
            raise ValueError(f"action {action_type} Unknown scroll params: {repr(params)}")
    elif action_type == 'Type':
        key = 'content'
        type_text = params.get(key)
        if type_text is not None:
            return 'Type', {'content': type_text}
        else:
            raise ValueError(f"action {action_type} Unknown type params: {repr(params)}")
    elif action_type == 'CallUser':
        key = 'content'
        call_text = params.get(key)
        if call_text is not None:
            return 'CallUser', {'content': call_text}
        else:
            raise ValueError(f"action {action_type} Unknown call user params: {repr(params)}")
    elif action_type == 'Launch':
        app = params.get("app", "")
        url = params.get("url", "")
        if app is not None:
            return 'Launch', {'app': app, 'url': url}
        else:
            raise ValueError(f"action {action_type} Unknown launch params: {repr(params)}")
    elif action_type == 'Finished':
        key = 'content'
        finished_text = params.get(key, "")
        return 'Finished', {'content': finished_text}
    elif action_type in ['Wait', 'PressBack', 'PressHome', 'PressEnter', 'PressRecent']:
        return action_type, {}
    else:
        raise ValueError(f"action {action_type} Unknown action: {repr(params)}")
    