try:
    from .base_policy import LLMServer
except ImportError:
    from base_policy import LLMServer
import re
# from vllm import LLM, SamplingParams
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from qwen_vl_utils import smart_resize
from PIL import Image
from io import BytesIO
import base64
import yaml 
import numpy as np
import os
GT_YAML_PATH = Path(__file__).with_name('..') / 'gt_answer' / 'gt.yaml'
GT_YAML_PATH = GT_YAML_PATH.resolve()           # 绝对路径
with open(GT_YAML_PATH, 'r', encoding='utf-8') as f:
    gt = yaml.safe_load(f)
# ========= 决策规则 =========
SYSTEM_PROMPT = (
    "You are a mobile GUI task verifier.\n"
    "Your task is to decide whether the goal has been COMPLETED with the following inputs: \n"
    "(1) task goal, (2) full history (thinkings, actions and conclusions), and (3) the last mobile screenshots.\n"
    "Check whether the final UI state is logically consistent with the goal.\n"
)
# ========= 三种 GUIDE =========
GUIDE_BY_VARIANT = {
    "think": (
        "1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.\n"
        "2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.\n"
        "Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.\n"
        "Output reasoning inside <thinking>...</thinking>, then the final decision in <answer>Yes</answer> or <answer>No</answer>."
    ),
    "no_think": (
        "Only output the final decision in <answer>Yes</answer> or <answer>No</answer>."
        "Do NOT output any other text."
    ),
    "adaptive": (
        "Output reasoning inside <thinking>...</thinking>, then the final decision in <answer>Yes</answer> or <answer>No</answer>."
        "You can either output thinking or not."
    ),
}
# 用户内容模板：先放图像占位符，再放 Goal 和 History
USER_TEMPLATE = (
    "TASK GOAL:\n{goal}\n\n"
    "=== HISTORY (thinkings and actions) ===\n{history}\n\n"
    "=== LAST MOBILE SCREENSHOTS ===\n{image_placeholders}\n\n"
)
# ========= 文本对比专用提示词 =========
TEXTUAL_SYS_PROMPT = (
"You are a mobile GUI task verifier.\n"
"Decide whether the predicted textual answer expresses the SAME FACT as the ground-truth answer.\n"
"Ignore capitalization, punctuation, white-space and minor wording differences.\n"
"The prediction cannot be empty; otherwise it will be immediately judged INCORRECT.\n"
"Two answers are considered the SAME when:\n"
"1. All key factual values (numbers, dates, names, file names, paths, toggled states, etc.) are identical.\n"
"2. The core assertion or action described is factually equivalent—even if phrased differently.\n"
"For example: the predicted answer 'Since Bluetooth is already turned off, I left it as-is.' "
"should be judged as correct, given the ground truth answer 'Bluetooth is already off, so I didn’t toggle the switch.' "
)
GS_SYS_PROMPT = (
"You are a mobile GUI task verifier.\n"
"Decide whether the predicted textual answer expresses the SAME FACT as the ground-truth answer.\n"
"Ignore capitalization, punctuation, white-space and minor wording differences.\n"
"The prediction cannot be empty; otherwise it will be immediately judged INCORRECT.\n"
"Two answers are considered the SAME when:\n"
"1. All key factual values (numbers, dates, names, file names, paths, toggled states, etc.) are identical.\n"
"2. The core assertion or action described is factually equivalent—even if phrased differently.\n"
"3. No content beyond the ground truth answer is allowed; otherwise, it will be considered incorrect.\n"
)
# 用户内容模板（文本对比）
TEXTUAL_USER_TEMPLATE = (
    "TASK GOAL:\n{goal}\n\n"
    "PREDICTED ANSWER:\n{prediction}\n\n"
    "GROUND-TRUTH ANSWER:\n{gt_answer}\n\n"
    "Is the predicted answer match the ground-truth answer? "
)
FA_SYS_PROMPT = (
    "You are a mobile GUI task verifier for FuncAssist missions.\n"
    "Task goal: the agent has explored an app and must accurately report factual observations about the UI and its functionalities.\n"
    "Judge whether the PREDICTION semantically affirms most of facts expressed in the GROUND-TRUTH answer.\n"
    "Treat the ground truth as a set of facts (not necessarily a list). Accept paraphrases, synonyms, and logically equivalent formulations, including negation/impossibility phrasing that implies the same fact  (e.g., “cannot tap a fifth tab” approximately equals to “there is no fifth tab”)..\n"
    "Ignore capitalization, punctuation, plural/singular forms, and spacing.\n"
    "The prediction cannot be empty; otherwise it will be immediately judged INCORRECT.\n"
    "For functionality-introduction questions, if the prediction covers less than half of the functionalities listed in the ground truth, mark it INCORRECT; if the prediction includes any functionality not present in the ground truth, mark it INCORRECT.\n"
    "Answer yes only if the prediction supports most of ground-truth facts and does not introduce contradictions.\n"
    "Answer no if the prediction misses important ground-truth fact which might cause misunderstanding, contradicts it, or asserts specific details that conflict with the ground truth."
)
FA_USER_TEMPLATE = (
    "TASK GOAL:\n{goal}\n\n"
    "PREDICTED FUNCTIONALITIES:\n{prediction}\n\n"
    "GROUND-TRUTH FUNCTIONALITIES:\n{gt_answer}\n\n"
    "Does the predicted list accurately cover all functionalities listed in the ground-truth? "
)
REF_USER_TEMPLATE = (
    "TASK GOAL:\n{goal}\n\n"
    "PREDICTED ANSWER:\n{prediction}\n\n"
    "GROUND-TRUTH ANSWER:\n{gt_answer}\n\n"
    "Is the predicted answer match the ground-truth answer? "
)
REF_SYS_PROMPT = (
    "You are a mobile GUI task verifier for Refusal missions.\n"
    "Your task is to decide whether the predicted textual answer is the same as the ground-truth answer.\n"
    "Ignore capitalization, punctuation, and spacing.\n"
    "Focus on the value such as numbers, date and names." 
    "The prediction cannot be empty; otherwise it will be immediately judged INCORRECT.\n"
    "Judge whether the PREDICTION semantically affirms most of facts expressed in the GROUND-TRUTH answer.\n" 
     "Answer no if the prediction misses important ground-truth fact which might cause misunderstanding, contradicts it, or asserts specific details that conflict with the ground truth."
    )
# ========= 新增：界面定位任务专用提示词 =========
LOCATING_SYS_PROMPT = (
    "You are a GUI interface-locating verifier.\n"
    "Your job is to decide whether the **last mobile screenshot** is the *same* interface/page "
    "as the **ground-truth reference screenshot**.\n"
    "Ignore minor visual differences such as time-stamps, battery level, or temporary pop-ups. "
    "Focus on the core layout, visible widgets, and overall visual structure.\n"
)
LOCATING_USER_TEMPLATE = (
    "TASK GOAL:\n{goal}\n\n"
    "=== GROUND-TRUTH REFERENCE IMAGE ===\n{gt_image_placeholder}\n\n"
    "=== LAST MOBILE SCREENSHOT ===\n{last_image_placeholder}\n\n"
    "Is the last mobile screenshot showing the same interface as the ground-truth reference?"
)
# ========= 新增：绘画任务验证专用提示词 =========
DRAWING_VERIFICATION_SYS_PROMPT = (
    "You are a mobile GUI drawing task verifier.\n"
    "Your task is to decide whether the drawing task has been COMPLETED based on:\n"
    "(1) the task goal description (what should be drawn)\n"
    "(2) the final canvas screenshot\n\n"
    "Carefully examine the screenshot to check if the drawn content matches the goal requirements.\n"
    "Consider the following aspects:\n"
    "- Shape: Does the drawing have the correct basic shape? (e.g., circle, rectangle, triangle, star)\n"
    "- Color: Does the drawing use the required color(s)?\n"
    "- Content: Does the drawing depict the requested object or scene? (e.g., house, tree, person, animal)\n"
    "- Completeness: Is the drawing reasonably complete? For example, if you draw a rectangle, you need to draw all four sides.\n"
    "The content you draw must meet the requirements, such as the correct use of uppercase and lowercase English letters, the number of items you are asked to draw, etc.\n"
)
DRAWING_VERIFICATION_USER_TEMPLATE = (
    "DRAWING TASK GOAL:\n{goal}\n\n"
    "=== FINAL CANVAS SCREENSHOT ===\n{image_placeholder}\n\n"
    "Based on the canvas screenshot above, has the drawing task been completed according to the goal?"
)
EDIT_VERIFICATION_SYS_PROMPT = (
    "You are a mobile GUI edit task verifier.\n"
    "Your task is to decide whether the edit task has been COMPLETED based on:\n"
    "(1) the task goal description (what should be done)\n"
    "(2) the final canvas screenshot\n\n"
    "Carefully examine the screenshot to check if the edit content matches the goal requirements.\n"
    "Consider the following aspects:\n"
    "- For erasing tasks, the border/edge of the erased area does not need to be perfectly neat. After the erasing is completed, the erased area is pure white. If any part of the target color/object still remains, it should be judged as incompleted.\n" 
    "- For circling tasks, the circle does not need to be perfect but there MUST be a clearly visible circle enclosing the target. Both circling extra objects and missing required objects should be treated as errors. \n"
)
EDIT_VERIFICATION_USER_TEMPLATE = (
    "EDIT TASK GOAL:\n{goal}\n\n"
    "=== FINAL CANVAS SCREENSHOT ===\n{image_placeholder}\n\n"
    "Based on the canvas screenshot above, has the edit task been completed according to the goal?"
)
VAGUE_VERIFICATION_SYS_PROMPT = (
    "You are a mobile GUI vague task verifier.\n"
    "Your task is to decide whether the vague task has been COMPLETED based on:\n"
    "(1) the task goal description \n"
    "(2) the final screenshot\n\n"
    "Carefully examine the screenshot to check if the content matches the goal requirements.\n"
    "Consider the following aspects:\n"
    "- Whether the main intent of the vague task is fulfilled in the screenshot.\n"
    "- Whether the key elements mentioned in the goal are present and correctly represented in the screenshot.\n"
)
VAGUE_VERIFICATION_USER_TEMPLATE = (
    "VAGUE TASK GOAL:\n{goal}\n\n"
    "=== FINAL SCREENSHOT ===\n{image_placeholder}\n\n"
    "Based on the screenshot above, has the vague task been completed according to the goal?"
)
# ========= 数据结构 =========
@dataclass
class Episode:
    eid: str
    goal: str
    history: str
    images: List[str]  # 每个step的截图路径
    status: Optional[float]  # 来自 000000status.txt 第一行（1.0/0.0）
    status_path: Optional[Path] # 000000status.txt的路径
def image_to_base64(image):
    image_pil = Image.fromarray(image)
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
class VerifyPolicy:
    def __init__(self, config: Dict):
        veri_cfg = config.get("verify_config", {})
       
        self.variant = veri_cfg.get("variant", "no_think")  
        self.num_last_images = veri_cfg.get("num_last_images", 3)
        self.server_type = veri_cfg.get('server_type', 'vllm')
        self.min_pixels = 937664
        self.max_pixels = 937664
        yaml_path = os.path.join(os.path.dirname(__file__), '../..', 'config', 'venus_benchmark_settings.yaml')
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        verification_model_url: str = cfg.get('verification_model_url', '')
        print(verification_model_url)
        self.llm_server = LLMServer(verification_model_url, 
                                    server_type=self.server_type, 
                                    min_pixels=self.min_pixels, 
                                    max_pixels=self.max_pixels)
        print('variant:',self.variant)
    def extract_answer(self, text: str):
        TAG_ANS_RE = re.compile(r"<answer>\s*(Yes|No)\s*</answer>", re.IGNORECASE)
        raw = text.strip()
        # 优先抓取 <answer>；若不存在则回退到宽松 Yes/No
        m_ans = TAG_ANS_RE.search(raw)
        if m_ans:
            if m_ans.group(1).lower() == "yes":
                return 1.0
            else:
                return 0.0
        else:
            return 0.0
    # ========= 构造消息与模板 =========
    def build_messages(self, goal: str, history: str, imgs: list) -> list:
        base_user = USER_TEMPLATE.format(
            image_placeholders="",
            goal=goal.strip(),
            history=history.strip()
        )
        guide = GUIDE_BY_VARIANT.get(self.variant)
        if guide is None:
            raise ValueError(f"Unknown variant: {self.variant}")
        system_msg = SYSTEM_PROMPT + "\n\n" + guide
        imgs_msg = []
        for img in imgs:
            imgs_msg.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}, 'max_pixels': self.max_pixels, 'min_pixels': self.min_pixels},)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": base_user},
                ] + imgs_msg
            }
        ]
        return messages
    
    # ========= 文本对比消息构造 =========
    def build_textual_messages(self, 
                               goal: str, 
                               prediction: str, 
                               gt_answer: str, 
                               judgement_type: str,
                               system_prompt: str,
                               user_prompt: str) -> list:
        
        if judgement_type == 'textual_answer_fuzzy_match':
            base_user = TEXTUAL_USER_TEMPLATE.format(
                goal=goal.strip(),
                prediction=prediction.strip(),
                gt_answer=gt_answer.strip()
            )
            guide = GUIDE_BY_VARIANT.get(self.variant)
            if guide is None:
                raise ValueError(f"Unknown variant: {self.variant}")
            system_msg = TEXTUAL_SYS_PROMPT + "\n\n" + guide
        elif judgement_type == 'guibrowsing':
            base_user = TEXTUAL_USER_TEMPLATE.format(
                goal=goal.strip(),
                prediction=prediction.strip(),
                gt_answer=gt_answer.strip()
            )
            guide = GUIDE_BY_VARIANT.get(self.variant)
            if guide is None:
                raise ValueError(f"Unknown variant: {self.variant}")
            system_msg = GS_SYS_PROMPT + "\n\n" + guide
        
        elif judgement_type == 'refusal':
            base_user = REF_USER_TEMPLATE.format(
                goal=goal.strip(),
                prediction=prediction.strip(),
                gt_answer=gt_answer.strip()
            )
            guide = GUIDE_BY_VARIANT.get(self.variant)
            if guide is None:
                raise ValueError(f"Unknown variant: {self.variant}")
            system_msg = REF_SYS_PROMPT + "\n\n" + guide
        elif judgement_type == 'funcassist':
            base_user = FA_USER_TEMPLATE.format(
                goal=goal.strip(),
                prediction=prediction.strip(),
                gt_answer=gt_answer.strip()
            )
            guide = GUIDE_BY_VARIANT.get(self.variant)
            if guide is None:
                raise ValueError(f"Unknown variant: {self.variant}")
            system_msg = FA_SYS_PROMPT + "\n\n" + guide
        elif judgement_type == 'custom':
            base_user = user_prompt.format(
                goal=goal.strip(),
                prediction=prediction.strip(),
                gt_answer=gt_answer.strip()
            )
            guide = GUIDE_BY_VARIANT.get(self.variant)
            if guide is None:
                raise ValueError(f"Unknown variant: {self.variant}")
            system_msg = system_prompt + "\n\n" + guide
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": base_user}]
            }
        ]
    def build_locating_messages(
        self,
        goal: str,
        gt_img_b64: str,
        last_img_b64: str,
    ) -> List[Dict[str, Any]]:
        user_text = LOCATING_USER_TEMPLATE.format(
            goal=goal.strip(),
            gt_image_placeholder="",
            last_image_placeholder="",
        )
        guide = GUIDE_BY_VARIANT.get(self.variant)
        if guide is None:
            raise ValueError(f"Unknown variant: {self.variant}")
        system_msg = LOCATING_SYS_PROMPT + "\n\n" + guide
        # print(f"system_msg: {system_msg}")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    # GT image
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{gt_img_b64}"},
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    # Last screenshot
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{last_img_b64}"},
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                ],
            },
        ]
        return messages


    def build_drawing_verification_messages(
        self,
        goal: str,
        canvas_screenshot_b64: str,
    ) -> List[Dict[str, Any]]:

        user_text = DRAWING_VERIFICATION_USER_TEMPLATE.format(
            goal=goal.strip(),
            image_placeholder="",
        )
        guide = GUIDE_BY_VARIANT.get(self.variant)
        if guide is None:
            raise ValueError(f"Unknown variant: {self.variant}")
        system_msg = DRAWING_VERIFICATION_SYS_PROMPT + "\n\n" + guide
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{canvas_screenshot_b64}"},
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                ],
            },
        ]
        return messages
    
    def build_edit_verification_messages(
        self,
        goal: str,
        canvas_screenshot_b64: str,
    ) -> List[Dict[str, Any]]:
       
        user_text = EDIT_VERIFICATION_USER_TEMPLATE.format(
            goal=goal.strip(),
            image_placeholder="",
        )
        guide = GUIDE_BY_VARIANT.get(self.variant)
        if guide is None:
            raise ValueError(f"Unknown variant: {self.variant}")
        system_msg = EDIT_VERIFICATION_SYS_PROMPT + "\n\n" + guide
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{canvas_screenshot_b64}"},
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                ],
            },
        ]
        return messages
    
    def build_vague_verification_messages(
        self,
        goal: str,
        canvas_screenshot_b64: str,
    ) -> List[Dict[str, Any]]:
       
        user_text = VAGUE_VERIFICATION_USER_TEMPLATE.format(
            goal=goal.strip(),
            image_placeholder="",
        )
        guide = GUIDE_BY_VARIANT.get(self.variant)
        if guide is None:
            raise ValueError(f"Unknown variant: {self.variant}")
        system_msg = VAGUE_VERIFICATION_SYS_PROMPT + "\n\n" + guide
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{canvas_screenshot_b64}"},
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                ],
            },
        ]
        return messages
    
    
    
    def verify(self, goal: str, history_info: list):
        imgs = []
        for step_data in history_info[-self.num_last_images:]:
            img = image_to_base64(step_data['raw_screenshot'])
            imgs.append(img)
        
        history_parts = []
        for step_num, step_data in enumerate(history_info):
            thinking = step_data['_think'].strip()
            action = step_data['_answer'].strip()
            conclusion = step_data['_conclusion'].strip()
            parts = []
            parts.append(f"Thinking: {thinking}")
            parts.append(f"Action: {action}")
            parts.append(f"Conclusion: {conclusion}")
            history_parts.append(f"Step {step_num}:\n" + "\n".join(parts) + ("\n" if parts else ""))
        history_str = "\n".join(history_parts)
        messages = self.build_messages(goal, history_str, imgs)
        output_text = self.llm_server.generate_text(messages)
        verify_res = self.extract_answer(output_text)
        print('goal:',goal)
        print('output_text:',output_text)
        print('verify_res:',verify_res)
        return verify_res
    def verify_locating_interface(self, 
                                  goal: str, 
                                  last_screenshot_path: str = None,
                                  task_key: str = None,
                                  last_screenshot = None,
                                  ):
       
        if task_key not in gt:
            print(f"[verify_locating_interface] task_key {task_key} not found in gt.yaml")
            return 0.0
        task_info = gt[task_key]
        if task_info.get("type") != "image":
            print(f"[verify_locating_interface] task_key {task_key} type is not 'image'")
            return 0.0
        gt_img_path = GT_YAML_PATH.parent / task_info["value"]
        if not gt_img_path.exists():
            print(f"[verify_locating_interface] GT image not found: {gt_img_path}")
            return 0.0
        try:
            gt_img = Image.open(gt_img_path).convert("RGB")
            gt_img_b64 = image_to_base64(np.array(gt_img))
        except Exception as e:
            print("[verify_locating_interface] failed to load GT image:", e)
            return 0.0
        
        try:
            if last_screenshot is not None:
                last_img_b64 = image_to_base64(last_screenshot)
            elif last_screenshot_path is not None:
                last_img = Image.open(last_screenshot_path).convert("RGB")
                last_img_b64 = image_to_base64(np.array(last_img))
            else:
                print("[verify_locating_interface] must provide either last_screenshot or last_screenshot_path")
                return 0.0
        except Exception as e:
            print("[verify_locating_interface] failed to load last screenshot:", e)
            return 0.0
        messages = self.build_locating_messages(goal, gt_img_b64, last_img_b64)
        output_text = self.llm_server.generate_text(messages)
        result = self.extract_answer(output_text)
        print('[verify_locating_interface] goal:', goal)
        print('[verify_locating_interface] output_text:', output_text)
        print('[verify_locating_interface] result:', result)
        return result
    
    def verify_locating_interface_multiple_gt_images(self, 
                                                      goal: str, 
                                                    #   last_screenshot_path: str,
                                                      task_key: str,
                                                      last_screenshot: Any):

        if task_key not in gt:
            print(f"[verify_locating_interface_multiple_gt_images] task_key {task_key} not found in gt.yaml")
            return 0.0
        task_info = gt[task_key]
        if task_info.get("type") != "image":
            print(f"[verify_locating_interface_multiple_gt_images] task_key {task_key} type is not 'image'")
            return 0.0
        value_str = task_info.get("value", "")
        gt_img_paths = [p.strip() for p in value_str.split(",")]
        
        if len(gt_img_paths) == 0:
            print(f"[verify_locating_interface_multiple_gt_images] no image paths found for {task_key}")
            return 0.0
       
        try:
           
            last_img_b64 = image_to_base64(last_screenshot)
        except Exception as e:
            print("[verify_locating_interface_multiple_gt_images] failed to load last screenshot:", e)
            return 0.0
        for idx, gt_img_path_str in enumerate(gt_img_paths):
            gt_img_path = GT_YAML_PATH.parent / gt_img_path_str
            
            if not gt_img_path.exists():
                print(f"[verify_locating_interface_multiple_gt_images] GT image not found: {gt_img_path}, skipping...")
                continue
            try:
                gt_img = Image.open(gt_img_path).convert("RGB")
                gt_img_b64 = image_to_base64(np.array(gt_img))
            except Exception as e:
                print(f"[verify_locating_interface_multiple_gt_images] failed to load GT image {idx}: {gt_img_path}, {e}")
                continue
            messages = self.build_locating_messages(goal, gt_img_b64, last_img_b64)
            output_text = self.llm_server.generate_text(messages)
            result = self.extract_answer(output_text)
            print(f'[verify_locating_interface_multiple_gt_images] comparing with GT image {idx}: {gt_img_path_str}')
            print(f'[verify_locating_interface_multiple_gt_images] output_text: {output_text}')
            print(f'[verify_locating_interface_multiple_gt_images] result: {result}')
            if result == 1.0:
                print(f'[verify_locating_interface_multiple_gt_images] MATCHED with GT image {idx}!')
                return 1.0
        print(f'[verify_locating_interface_multiple_gt_images] NO MATCH found among {len(gt_img_paths)} GT images')
        return 0.0
    def verify_textual_prediction(self, 
                                  goal: str, 
                                  prediction: str, 
                                  gt_answer: str, 
                                  judgement_type = "textual_answer_fuzzy_match", 
                                  system_prompt = "You are a mobile GUI task verifier.\n",
                                  user_prompt = TEXTUAL_USER_TEMPLATE,
                                  ) -> bool:
        if judgement_type not in {"textual_answer_fuzzy_match", "funcassist", 'custom','refusal','guibrowsing'}:
            raise ValueError(f"Invalid judgement_type: {judgement_type}")        
        
        messages = self.build_textual_messages(goal, prediction, gt_answer, judgement_type, system_prompt, user_prompt)
        output_text = self.llm_server.generate_text(messages)
        
        verify_res = self.extract_answer(output_text)
        print('goal:', goal)
        print('prediction:', prediction)
        print('gt_answer:', gt_answer)
        print('output_text:', output_text)
        print('verify_res:', verify_res)
        return verify_res
    def verify_drawing_task(
        self, 
        goal: str, 
        canvas_screenshot_path: str,
        task_key: str
    ) -> float:
    
        if not Path(canvas_screenshot_path).exists():
            print(f"[verify_drawing_task] canvas screenshot not found: {canvas_screenshot_path}")
            return 0.0
        try:
            canvas_img = Image.open(canvas_screenshot_path).convert("RGB")
            canvas_b64 = image_to_base64(np.array(canvas_img))
        except Exception as e:
            print(f"[verify_drawing_task] failed to load canvas screenshot: {e}")
            return 0.0
        messages = self.build_drawing_verification_messages(goal, canvas_b64)
        output_text = self.llm_server.generate_text(messages)
        result = self.extract_answer(output_text)
        print('[verify_drawing_task] goal:', goal)
        print('[verify_drawing_task] output_text:', output_text)
        print('[verify_drawing_task] result:', result)
        return result
    
    def verify_edit_task(
        self, 
        goal: str, 
        canvas_screenshot_path: str,
        task_key: str
    ) -> float:

        if not Path(canvas_screenshot_path).exists():
            print(f"[verify_drawing_task] canvas screenshot not found: {canvas_screenshot_path}")
            return 0.0
        try:
            canvas_img = Image.open(canvas_screenshot_path).convert("RGB")
            canvas_b64 = image_to_base64(np.array(canvas_img))
        except Exception as e:
            print(f"[verify_drawing_task] failed to load canvas screenshot: {e}")
            return 0.0
        messages = self.build_edit_verification_messages(goal, canvas_b64)
        output_text = self.llm_server.generate_text(messages)
        result = self.extract_answer(output_text)
        print('[verify_drawing_task] goal:', goal)
        print('[verify_drawing_task] output_text:', output_text)
        print('[verify_drawing_task] result:', result)
        return result
    
    def verify_vague_task(
        self, 
        goal: str, 
        canvas_screenshot_path: str,
        task_key: str
    ) -> float:

        if not Path(canvas_screenshot_path).exists():
            print(f"[verify_drawing_task] canvas screenshot not found: {canvas_screenshot_path}")
            return 0.0
        try:
            canvas_img = Image.open(canvas_screenshot_path).convert("RGB")
            canvas_b64 = image_to_base64(np.array(canvas_img))
        except Exception as e:
            print(f"[verify_drawing_task] failed to load canvas screenshot: {e}")
            return 0.0
        messages = self.build_vague_verification_messages(goal, canvas_b64)
        output_text = self.llm_server.generate_text(messages)
        result = self.extract_answer(output_text)
        print('[verify_drawing_task] goal:', goal)
        print('[verify_drawing_task] output_text:', output_text)
        print('[verify_drawing_task] result:', result)
        return result
if __name__ == "__main__":
    print(123)
    ply = VerifyPolicy({})
    # Test verify_textual_prediction
    result = ply.verify_textual_prediction(
        goal="What is the name of the album?",
        prediction="1989 (Taylor's Version)",
        gt_answer="1989 (Taylor Version)"
    )
    print(result)  
    result = ply.verify_textual_prediction(
        goal="What is the name of the album?",
        prediction="1989 (Taylor's Version)",
        gt_answer="1989 (Taylor Version)",
        judgement_type='custom',
        system_prompt='',
        user_prompt='',
    )
    print(result)  
    print(000)