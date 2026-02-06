import random
import textwrap
import pprint
import string

def generate_random_html_for_agent_v6(button_type=None):
    """
    Generates an HTML file for a UI Agent with 10 random functions (v6 - All English).
    - Task prompts (for the agent) are not displayed on the webpage, only output in Python.
    - The webpage contains validation logic to verify the agent's actions.
    - Calculator, Notepad, and Search functions require specific tasks to be completed.
    - The Search function is now a two-step task: type then click.

    Returns:
        tuple: (html_content, icon_feature_mapping)
            - html_content (str): The complete HTML code as a string.
            - icon_feature_mapping (list): A list of dictionaries detailing each button's
              icon, feature, and the 'prompt'/'target_value' for the agent.
    """

    # 1. Define optional icons
    icons = [
        "fa-calculator", "fa-search", "fa-file-alt", "fa-image", "fa-map-marker-alt",
        "fa-palette", "fa-clock", "fa-vial", "fa-quote-right", "fa-random",
        "fa-camera", "fa-music", "fa-gamepad", "fa-lightbulb", "fa-moon",
        "fa-sun", "fa-wifi", "fa-battery-full", "fa-car", "fa-bicycle"
    ]

    # 2. Define feature templates (All English)
    features = [
        {
            'id': 'calculator',
            'name': 'Simple Calculator',
            'html_template': textwrap.dedent("""
                <div id="calc-display" style="background:#333; color:white; padding:10px; text-align:right; font-size:1.5em; border-radius:5px; margin-bottom:10px;">0</div>
                <div id="calc-buttons" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                    {buttons}
                </div>
                <p id="calculator-success-msg" class="success-message">Success!</p>
            """),
            'js_init_template': 'document.getElementById("calc-display").innerText = "0"; document.getElementById("calculator-success-msg").style.display="none";',
            'js_logic_template': """
                case '=':
                    try {{
                        const display = document.getElementById('calc-display');
                        const result = eval(display.innerText.replace('×', '*').replace('÷', '/'));
                        display.innerText = result;
                        if (result == {target_number}) {{
                            document.getElementById('calculator-success-msg').style.display = 'block';
                        }}
                    }} catch {{ document.getElementById('calc-display').innerText = 'Error'; }}
                    break;
            """
        },
        {
            'id': 'search',
            'name': 'Web Search',
            'html_template': textwrap.dedent("""
                <input type="text" id="search-input" placeholder="Enter search query..." 
                       style="width:100%; padding:10px; font-size:1em; margin-bottom:10px; box-sizing: border-box;">
                <button onclick="validate_search_input()">Perform Search</button>
                <p id="search-success-msg" class="success-message">Success!</p>
            """),
            'js_init_template': "document.getElementById('search-input').value = ''; document.getElementById('search-success-msg').style.display='none';",
            'js_logic_template': """
            function validate_search_input() {{
                const target = '{target_str}';
                const current = document.getElementById('search-input').value;
                if (current === target) {{
                    document.getElementById('search-success-msg').style.display = 'block';
                }}
            }}
            """
        },
        {
            'id': 'notepad',
            'name': 'Temporary Notepad',
            'html_template': textwrap.dedent("""
                <textarea id="notepad-area" rows="8" style="width:100%; font-size:1em; box-sizing: border-box;" 
                          placeholder="Type here..." oninput="check_notepad_input()"></textarea>
                <p id="notepad-success-msg" class="success-message">Success!</p>
            """),
            'js_init_template': 'document.getElementById("notepad-area").value = ""; document.getElementById("notepad-success-msg").style.display="none";',
            'js_logic_template': """
            function check_notepad_input() {{
                const target = '{target_str}';
                const current = document.getElementById('notepad-area').value;
                if (current === target) {{
                    document.getElementById('notepad-success-msg').style.display = 'block';
                }} else {{
                    document.getElementById('notepad-success-msg').style.display = 'none';
                }}
            }}
            """
        },
        {'id': 'image_picker', 'name': 'Image Picker', 'html': '<p>Click the button below to select an image from your gallery.</p><input type="file" id="image-input" accept="image/*" style="display:none;"><button onclick="document.getElementById(\'image-input\').click();">Select Image</button><img id="image-preview" style="max-width:100%; margin-top:15px; border-radius:5px;">', 'js_init': "document.getElementById('image-preview').src = ''; document.getElementById('image-input').onchange = preview_image;"},
        {'id': 'location', 'name': 'Get My Location', 'html': '<p id="location-status">Click the button to get your current coordinates.</p><button onclick="get_location()">Get Location</button><div id="location-result" style="margin-top:10px; font-weight:bold; white-space: pre-wrap;"></div><p id="location-success-msg" class="success-message">Success!</p>', 'js_init': 'document.getElementById("location-status").innerText = "Click the button to get your current coordinates."; document.getElementById("location-result").innerText = ""; document.getElementById("location-success-msg").style.display="none";'},
        {'id': 'color_changer', 'name': 'Background Changer', 'html': '<p>Click the button to change the background color.</p><button onclick="change_bg_color()">Random Color</button><p id="color-changer-success-msg" class="success-message">Success!</p>', 'js_init': 'document.getElementById("color-changer-success-msg").style.display="none";'},
        {'id': 'current_time', 'name': 'Current Time', 'html': '<p style="font-size: 2em; text-align:center;" id="time-display"></p>', 'js_init': 'document.getElementById("time-display").innerText = new Date().toLocaleTimeString("en-US");'},
        {'id': 'vibrator', 'name': 'Phone Vibrator', 'html': '<p>Click the button to make the phone vibrate (if supported).</p><button onclick="vibrate_phone()">Vibrate for 200ms</button>'},
        {'id': 'random_quote', 'name': 'Random Quote', 'html': '<p id="quote-display" style="font-style:italic; text-align:center; padding: 10px 0;"></p>', 'js_init': 'show_random_quote();'},
        {'id': 'random_number', 'name': 'Random Number Generator', 'html': '<p>Generate a random number between 1 and 100.</p><button onclick="generate_random_number()">Generate</button><p id="random-number-display" style="font-size: 2em; text-align:center; font-weight:bold; margin-top:10px;"></p><p id="random-number-success-msg" class="success-message">Success!</p>', 'js_init': 'document.getElementById("random-number-display").innerText = ""; document.getElementById("random-number-success-msg").style.display="none";'}
    ]

    # 3. Randomly select 10 unique icons and features
    if len(icons) < 10 or len(features) < 10:
        raise ValueError("At least 10 icons and 10 features are required.")
        
    selected_icons = random.sample(icons, 10)
    selected_features_definitions = random.sample(features, 10)

# ----- 2. 根据 button_type 决定是否启用 Success! -----
    def should_enable_success(feature_id):
        if button_type is None:
            return True   # v6 行为
        return feature_id == button_type



    # 4. Generate HTML content and mappings
    buttons_html, features_html, js_init_cases = "", "", ""
    icon_feature_mapping = []
    extra_js_logic, calculator_js_logic = "", ""
    feature_name_map_str = ", ".join([f"'{f['id']}': '{f['name']}'" for f in features])

    for i in range(10):
        icon_class = selected_icons[i]
        feature_info = selected_features_definitions[i].copy()
        feature_id = feature_info['id']
        
        mapping_entry = {
            'button_index': i,
            'icon': icon_class,
            'feature_id': feature_id,
            'feature_name': feature_info['name']
        }

        if feature_id == 'notepad':
            target_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            prompt = f"Please type the following text accurately into the Temporary Notepad: {target_str}"
            feature_info['html'] = feature_info['html_template']
            extra_js_logic += feature_info['js_logic_template'].format(target_str=target_str)
            feature_info['js_init'] = feature_info['js_init_template']
            mapping_entry['prompt'] = prompt
            mapping_entry['target_value'] = target_str

        elif feature_id == 'search':
            target_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            prompt = f"Please type '{target_str}' into the search box and click the 'Perform Search' button."
            feature_info['html'] = feature_info['html_template']
            extra_js_logic += feature_info['js_logic_template'].format(target_str=target_str)
            feature_info['js_init'] = feature_info['js_init_template']
            mapping_entry['prompt'] = prompt
            mapping_entry['target_value'] = target_str

        elif feature_id == 'calculator':
            num1 = random.randint(100, 999)
            num2 = random.randint(10, 99)
            target_number = num1 * num2
            prompt = f"Please use the calculator to compute {num1} × {num2}."

            calc_buttons_html = textwrap.dedent("""
                <button onclick="calc_press('7')">7</button><button onclick="calc_press('8')">8</button><button onclick="calc_press('9')">9</button><button onclick="calc_press('/')">÷</button>
                <button onclick="calc_press('4')">4</button><button onclick="calc_press('5')">5</button><button onclick="calc_press('6')">6</button><button onclick="calc_press('*')">×</button>
                <button onclick="calc_press('1')">1</button><button onclick="calc_press('2')">2</button><button onclick="calc_press('3')">3</button><button onclick="calc_press('-')">-</button>
                <button onclick="calc_press('0')">0</button><button onclick="calc_press('.')">.</button><button onclick="calc_press('C')">C</button><button onclick="calc_press('+')">+</button>
                <button style="grid-column: span 4;" onclick="calc_press('=')">=</button>
            """)
            feature_info['html'] = feature_info['html_template'].format(buttons=calc_buttons_html)
            calculator_js_logic = feature_info['js_logic_template'].format(target_number=target_number)
            feature_info['js_init'] = feature_info['js_init_template']
            mapping_entry['prompt'] = prompt
            mapping_entry['target_value'] = target_number
        
        buttons_html += f'<button class="grid-button" onclick="show_modal(\'{feature_id}\')"><i class="fas {icon_class}"></i></button>\n'
        features_html += f'<div id="{feature_id}-content" class="feature-content" style="display:none;">{feature_info["html"]}</div>\n'
        
        if 'js_init' in feature_info:
            js_init_cases += f"case '{feature_id}':\n    {feature_info['js_init']}\n    break;\n"
            
        icon_feature_mapping.append(mapping_entry)

    # 5. Assemble the final HTML using an f-string, escaping all non-Python braces with {{}}
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Agent Task Board v6</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        :root {{ --primary-color: #007bff; --dark-bg: #f4f4f9; --light-bg: #ffffff; --shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 15px; background-color: var(--dark-bg); -webkit-tap-highlight-color: transparent; transition: background-color 0.5s; }}
        #button-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 15px; }}
        .grid-button {{ aspect-ratio: 1 / 1; background-color: var(--light-bg); border: none; border-radius: 15px; box-shadow: var(--shadow); cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; font-size: 2.5em; color: var(--primary-color); }}
        .grid-button:active {{ transform: scale(0.95); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        #modal-overlay {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.6); display: none; justify-content: center; align-items: center; z-index: 1000; }}
        #modal-container {{ background-color: var(--light-bg); padding: 20px; border-radius: 15px; width: 90%; max-width: 500px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); display: flex; flex-direction: column; max-height: 80vh; }}
        #modal-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }}
        #modal-title {{ font-size: 1.2em; font-weight: bold; }}
        #modal-close-btn {{ background: none; border: none; font-size: 1.5em; cursor: pointer; color: #888; }}
        #modal-content-wrapper {{ overflow-y: auto; }}
        #modal-content-wrapper button {{ width: 100%; padding: 12px; background-color: var(--primary-color); color: white; border: none; border-radius: 8px; font-size: 1em; cursor: pointer; margin-top: 10px; }}
        #modal-content-wrapper button:active {{ opacity: 0.8; }}
        .success-message {{ color: #28a745; font-weight: bold; text-align: center; margin-top: 15px; font-size: 1.1em; display: none; }}
    </style>
</head>
<body>
    <h1 style="text-align:center; color:#333; font-weight:300;">Agent Task Board v6</h1>
    <div id="button-grid">{buttons_html}</div>
    <div id="modal-overlay" onclick="if(event.target === this) hide_modal()">
        <div id="modal-container">
            <div id="modal-header">
                <span id="modal-title">Feature</span>
                <button id="modal-close-btn" onclick="hide_modal()">&times;</button>
            </div>
            <div id="modal-content-wrapper">{features_html}</div>
        </div>
    </div>
<script>
    const modal_overlay = document.getElementById('modal-overlay');
    const modal_title = document.getElementById('modal-title');
    const feature_contents = document.querySelectorAll('.feature-content');
    const feature_names = {{ {feature_name_map_str} }};

    function show_modal(feature_id) {{
        document.querySelectorAll('.success-message').forEach(el => el.style.display = 'none');
        feature_contents.forEach(content => content.style.display = 'none');
        document.getElementById(feature_id + '-content').style.display = 'block';
        modal_title.innerText = feature_names[feature_id] || 'Feature';
        switch (feature_id) {{
            {js_init_cases}
            default: break;
        }}
        modal_overlay.style.display = 'flex';
    }}
    function hide_modal() {{ modal_overlay.style.display = 'none'; }}
    function calc_press(key) {{
        const display = document.getElementById('calc-display');
        let current = display.innerText;
        switch (key) {{
            case 'C': display.innerText = '0'; break;
            {calculator_js_logic}
            default:
                if (current === '0' && !'+-*/×÷.'.includes(key)) {{ display.innerText = key; }}
                else {{ display.innerText += key; }}
                break;
        }}
    }}
    function preview_image(event) {{
        const reader = new FileReader();
        reader.onload = () => {{ document.getElementById('image-preview').src = reader.result; }};
        if (event.target.files[0]) {{
            reader.readAsDataURL(event.target.files[0]);
        }}
    }}
    function get_location() {{
        const status = document.getElementById('location-status'), result = document.getElementById('location-result');
        if (navigator.geolocation) {{
            status.innerText = 'Getting location...';
            navigator.geolocation.getCurrentPosition(
                p => {{
                    status.innerText = 'Location acquired successfully!';
                    result.innerText = `Latitude: ${{p.coords.latitude.toFixed(4)}}\\nLongitude: ${{p.coords.longitude.toFixed(4)}}`;
                    document.getElementById('location-success-msg').style.display = 'block';
                }},
                () => {{ status.innerText = 'Unable to get location. Please check location permissions for the app or browser.'; }}
            );
        }} else {{ status.innerText = 'Geolocation is not supported by this browser.'; }}
    }}
    function change_bg_color() {{
        document.body.style.backgroundColor = '#' + Math.floor(Math.random()*16777215).toString(16).padStart(6, '0');
        document.getElementById('color-changer-success-msg').style.display = 'block';
    }}
    function vibrate_phone() {{ if ('vibrate' in navigator) navigator.vibrate(200); else alert('Your browser does not support the Vibration API.'); }}
    const quotes = [
        "The only way to do great work is to love what you do.", "Stay hungry, stay foolish.",
        "The best time to plant a tree was 20 years ago. The second best time is now."
    ];
    function show_random_quote() {{ document.getElementById('quote-display').innerText = quotes[Math.floor(Math.random() * quotes.length)]; }}
    function generate_random_number() {{
        document.getElementById('random-number-display').innerText = Math.floor(Math.random() * 100) + 1;
        document.getElementById('random-number-success-msg').style.display = 'block';
    }}
    {extra_js_logic}
</script>
</body>
</html>
    """
    
    return html_content.strip(), icon_feature_mapping


# --- Main execution block ---
if __name__ == "__main__":
    html_content, icon_to_feature_map = generate_random_html_for_agent_v6()

    file_name = "agent_tasks_v6_en.html"
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ Success! Task file '{file_name}' generated for the agent.")
        print("\nNext steps:")
        print(f"1. Open '{file_name}' in a browser within your test environment (e.g., Android emulator).")
        print("2. Feed the mapping and task prompts printed below to your UI agent.")
    except Exception as e:
        print(f"❌ Error generating file: {e}")

    print("\n" + "="*60)
    print("Icon, Feature, and Task (Prompt) Mapping for the Agent:")
    print("="*60)
    pprint.pprint(icon_to_feature_map)
    print(icon_to_feature_map)
    print("="*60)
    print("\nNote: The 'prompt' field exists only in this Python output to instruct the agent.")
    print("The web UI itself does not display these task instructions.")

