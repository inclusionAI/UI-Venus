from processor.rpa_utils.postproc_utils import action2des_qwen25vl_strict


class RuntimeContext:
    def __init__(self):
        self.step = 0
        self.history = []
        self.pred_action = []
        self.action_description = []
    
    def reset(self):
        self.step = 0
        self.history = []
        self.pred_action = []
        self.action_description = []
    
    def update_action_description(self, state, action, result):
        _ = state  # Keep the state parameter for future extensions; it is not used yet.
        _, description = action2des_qwen25vl_strict(action, result, fail_reason='')
        self.action_description.append(description)
    
    def update_history(self, key, value):
        self.history[-1][key] = value
    
    def close(self):
        pass
