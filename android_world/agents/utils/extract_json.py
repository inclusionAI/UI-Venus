import re
import json

def extract_json(text):
    def fix_missing_brackets(s):
        stack = []
        for c in s:
            if c in ('{', '['):
                stack.append(c)
            elif c == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif c == ']' and stack and stack[-1] == '[':
                stack.pop()
        missing = []
        while stack:
            c = stack.pop()
            if c == '{':
                missing.append('}')
            elif c == '[':
                missing.append(']')
        return s + ''.join(missing)

    # 尝试直接解析整个字符串
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 提取所有候选JSON片段（含嵌套结构）
    candidates = []
    for match in re.finditer(r'\{[\s\S]*', text):
        start = match.start()
        substr = text[start:]
        max_length = min(len(substr), 2000)  # 防止处理过长文本
        candidates.append(substr[:max_length])

    # 按长度排序（优先处理长文本）
    candidates.sort(key=len, reverse=True)

    # 尝试解析候选片段
    for candidate in candidates:
        # 先尝试修复括号
        fixed_candidate = fix_missing_brackets(candidate)
        try:
            return json.loads(fixed_candidate)
        except json.JSONDecodeError:
            pass

        # 尝试逐字符截断
        for end in range(len(fixed_candidate), 0, -1):
            try:
                return json.loads(fixed_candidate[:end])
            except:
                continue

    # 最终回退：提取第一个{开始的有效内容
    start_idx = text.find('{')
    if start_idx != -1:
        for end in range(len(text), start_idx, -1):
            try:
                return json.loads(text[start_idx:end])
            except:
                continue

    return None

if __name__ == '__main__':
    examples = [
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [948, 1507]}}\n</tool_call>\n📐\n⚗️',
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [560, 1716]}}\n⚗\n',
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "swipe", "coordinate": [567, 1480], "coordinate2": [567, 555]}}\n📐\n\n⚗️',
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [948, 1506]}}\n⚗\n',
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [948, 1507]}}\n⚗\n',
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [948, 1506]}}\n⚗\n',
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [957, 1508]}}\n📐\n\nuser\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [957, 1508]}}\n📐\n⚗️',
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [968, 1507]}}\n⚗\n',
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [967, 1508]}}\n⚗\n',
    ]

    for text in examples:
        result = extract_json(text)
        print(f"{json.dumps(result,)}")