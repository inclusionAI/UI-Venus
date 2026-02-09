def action2des_qwen25vl_strict(step_action, result, fail_reason):
    target_text = ''
    input_text = ''
    history =''
    
    # Assuming step_action is already the processed action dictionary
    pred_action = step_action 
    
    if pred_action is None:
        return None, result
    action_type = pred_action.get('action', None)

    if action_type in ['CLK']:
        box = pred_action.get('box')
        history = f'点击位置为"{box}"的地方'

    elif action_type in ['INPUT']:
        input_text =  pred_action['input_text']
        history = f'在输入框中输入"{input_text}"'

    elif action_type in ['WAIT']:
        history = '等待页面加载'
    elif action_type in ['BACK']:
        history = '回退上一步'
    elif action_type in ['SWIPE_UP']:
        history = "向上滑动屏幕"
    elif action_type in ['SWIPE_DOWN']:
        history="向下滑动屏幕"
    elif action_type in ['SWIPE_LEFT']:
        history = "向左滑动屏幕"
    elif action_type in ['SWIPE_RIGHT']:
        history = "向右滑动屏幕"
    elif action_type in ['REOPEN']:
        history = '重新进入小程序'
    elif action_type in ['FAIL']:
        history = '失败'
    elif action_type in ['SUCCESS']:
        history = '成功'
    elif action_type in ['STOP']:
        if fail_reason == '':
            history = ''
            pred_action['action'] = 'SUCCESS'
            pred_action['memo'] = '成功'            
        else:
            history = fail_reason
            pred_action['action'] = 'FAIL'
            pred_action['memo'] = fail_reason
    return pred_action, result