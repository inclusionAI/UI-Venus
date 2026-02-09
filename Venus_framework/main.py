import logging
import argparse
import yaml
import os
import sys
from app.run_handler import RunHandler


def validate_config(config: dict, args) -> bool:
    """验证配置是否完整
    
    Args:
        config: 配置字典
        args: 命令行参数
        
    Returns:
        配置是否有效
    """
    logger = logging.getLogger(__name__)
    
    # 检查必要配置项
    required_keys = {
        'device.id': config.get('device', {}).get('id'),
        'policy.type': config.get('policy', {}).get('type'),
        'policy.params.model_host': config.get('policy', {}).get('params', {}).get('model_host'),
        'ep_config.step_limit': config.get('ep_config', {}).get('step_limit'),
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    if missing_keys:
        logger.error(f"配置缺失必要字段: {', '.join(missing_keys)}")
        return False
    
    # 检查 trace_dir
    if not args.trace_dir:
        logger.error("必须指定 --trace-dir 参数")
        return False
    
    # 检查 purpose
    if not args.purpose or args.purpose.strip() == '':
        logger.error("必须指定 --purpose 参数（任务描述）")
        return False
    
    return True


def setup_logging(log_file: str = None):
    """配置日志系统
    
    Args:
        log_file: 日志文件路径（可选）
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    parser = argparse.ArgumentParser(
        description='Android自动化任务执行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python main.py --config config/ui_venus_single.yaml \\
                 --device-id "192.168.1.100:5555" \\
                 --purpose "打开微博，搜索杭州天气" \\
                 --trace-dir "record/traces/"
        '''
    )
    
    parser.add_argument('--config', default='config/ui_venus_single.yaml', 
                       help='配置文件路径 (默认: config/ui_venus_single.yaml)')
    parser.add_argument('--purpose', required=True, help='任务描述（必填）')
    parser.add_argument('--device-id', type=str, help='设备ID')
    parser.add_argument('--trace-dir', required=True, help='轨迹保存目录（必填）')
    parser.add_argument('--step-limit', type=int, help='最大步数限制')
    parser.add_argument('--model-host', type=str, help='模型服务地址')
    parser.add_argument('--model-name', type=str, help='模型名称')
    parser.add_argument('--save-dir', type=str, help='截图保存目录')
    parser.add_argument('--log-file', type=str, help='日志文件路径')
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加载外部 app_mapping 配置文件（如果指定了）
        app_mapping_config = config.get('app_mapping', {})
        if 'config_file' in app_mapping_config:
            app_mapping_file = app_mapping_config['config_file']
            # 支持相对路径（相对于主配置文件目录）
            if not os.path.isabs(app_mapping_file):
                config_dir = os.path.dirname(args.config)
                app_mapping_file = os.path.join(config_dir, os.path.basename(app_mapping_file))
            
            if os.path.exists(app_mapping_file):
                with open(app_mapping_file, 'r', encoding='utf-8') as f:
                    app_mapping_data = yaml.safe_load(f)
                    config['app_mapping'] = app_mapping_data.get('app_mapping', {})
                    logger.info(f"已加载应用映射配置: {app_mapping_file}")
            else:
                logger.warning(f"应用映射配置文件不存在: {app_mapping_file}")
                config['app_mapping'] = {}
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        sys.exit(1)
    
    # 命令行参数覆盖配置
    if args.device_id:
        config.setdefault('device', {})['id'] = args.device_id
    if args.step_limit is not None:
        config.setdefault('ep_config', {})['step_limit'] = args.step_limit
    if args.model_host:
        config.setdefault('policy', {}).setdefault('params', {})['model_host'] = args.model_host
    if args.model_name:
        config.setdefault('policy', {}).setdefault('params', {})['model_name'] = args.model_name
    if args.save_dir:
        config.setdefault('record_config', {})['save_dir'] = args.save_dir
    
    # 验证配置
    if not validate_config(config, args):
        sys.exit(1)
    
    # 打印配置信息
    logger.info("=" * 60)
    logger.info("任务配置:")
    logger.info(f"  设备ID: {config['device']['id']}")
    logger.info(f"  任务描述: {args.purpose}")
    logger.info(f"  步数限制: {config['ep_config']['step_limit']}")
    logger.info(f"  模型地址: {config['policy']['params']['model_host']}")
    logger.info(f"  模型名称: {config['policy']['params'].get('model_name', 'model')}")
    logger.info(f"  轨迹目录: {args.trace_dir}")
    logger.info("=" * 60)
    
    try:
        # 创建并运行处理器
        kwargs = config['policy'].get('params', {})
        
        handler = RunHandler(
            device_id=config['device']['id'],
            trace_dir=args.trace_dir,
            policy_type=config['policy']['type'],
            ep_config=config.get('ep_config'),
            app_mapping=config.get('app_mapping', {}),
            **kwargs
        )
        
        result = handler.run(purpose=args.purpose)
        
        # 解析返回结果 (is_successful, termination_reason, call_user_content)
        is_successful, termination_reason, call_user_content = result
        
        # 根据不同的终止原因打印不同的信息
        if termination_reason == 'success':
            logger.info("✅ 任务执行成功")
            sys.exit(0)
        elif termination_reason == 'call_user':
            logger.info("📢 任务需要用户接管或反馈")
            if call_user_content:
                logger.info(">> 反馈内容: %s", call_user_content)
            sys.exit(0)
        elif termination_reason == 'max_steps':
            logger.warning("⚠️ 任务达到最大步数限制，但尚未完成")
            sys.exit(1)
        elif termination_reason == 'repeat_loop':
            logger.warning("⚠️ 任务陷入重复循环，已自动终止")
            sys.exit(1)
        elif termination_reason == 'screenshot_failed':
            logger.error("❌ 截图获取失败，任务终止")
            sys.exit(1)
        else:
            logger.warning("⚠️ 任务以其他原因终止: %s", termination_reason)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ 任务执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()