import logging
import argparse
import yaml
import os
import sys
from app.run_handler import RunHandler


def validate_config(config: dict, args) -> bool:
    """Validate that the configuration is complete.
    
    Args:
        config: Configuration dictionary.
        args: Command-line arguments.
        
    Returns:
        Whether the configuration is valid.
    """
    logger = logging.getLogger(__name__)
    
    # Check required configuration values.
    required_keys = {
        'device.id': config.get('device', {}).get('id'),
        'policy.type': config.get('policy', {}).get('type'),
        'policy.params.model_host/model_url': (
            config.get('policy', {}).get('params', {}).get('model_host')
            or config.get('policy', {}).get('params', {}).get('model_url')
        ),
        'ep_config.step_limit': config.get('ep_config', {}).get('step_limit'),
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    if missing_keys:
        logger.error(f"配置缺失必要字段: {', '.join(missing_keys)}")
        return False
    
    # Check trace_dir.
    if not args.trace_dir:
        logger.error("必须指定 --trace-dir 参数")
        return False
    
    # Check the task purpose.
    if not args.purpose or args.purpose.strip() == '':
        logger.error("必须指定 --purpose 参数（任务描述）")
        return False
    
    return True


def setup_logging(log_file: str = None):
    """Configure logging.
    
    Args:
        log_file: Optional log file path.
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
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
  python main.py --config config/ui_venus_2_single.yaml \\
                 --device-id "192.168.1.100:5555" \\
                 --purpose "打开微博，搜索杭州天气" \\
                 --trace-dir "record/traces/"
        '''
    )
    
    parser.add_argument('--config', default='config/ui_venus_2_single.yaml',
                       help='配置文件路径 (默认: config/ui_venus_2_single.yaml)')
    parser.add_argument('--purpose', required=True, help='任务描述（必填）')
    parser.add_argument('--device-id', type=str, help='设备ID')
    parser.add_argument('--trace-dir', required=True, help='轨迹保存目录（必填）')
    parser.add_argument('--step-limit', type=int, help='最大步数限制')
    parser.add_argument('--model-host', type=str, help='模型服务地址')
    parser.add_argument('--model-url', type=str, help='模型 API 地址')
    parser.add_argument('--model-name', type=str, help='模型名称')
    parser.add_argument('--log-file', type=str, help='日志文件路径')
    parser.add_argument('--reflection', action='store_true', help='启用实时反思监督')
    parser.add_argument('--reflection-config', default='config/reflection.yaml', help='反思监督配置文件路径')
    
    args = parser.parse_args()
    
    # Configure logging.
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Check that the configuration file exists.
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load an external app_mapping configuration when specified.
        app_mapping_config = config.get('app_mapping', {})
        if 'config_file' in app_mapping_config:
            app_mapping_file = app_mapping_config['config_file']
            # Resolve relative paths from the main configuration directory.
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
    
    # Override configuration values with command-line arguments.
    if args.device_id:
        config.setdefault('device', {})['id'] = args.device_id
    if args.step_limit is not None:
        config.setdefault('ep_config', {})['step_limit'] = args.step_limit
    if args.model_host:
        config.setdefault('policy', {}).setdefault('params', {})['model_host'] = args.model_host
    if args.model_url:
        config.setdefault('policy', {}).setdefault('params', {})['model_url'] = args.model_url
    if args.model_name:
        config.setdefault('policy', {}).setdefault('params', {})['model_name'] = args.model_name
    if args.reflection:
        if not os.path.exists(args.reflection_config):
            logger.error(f"反思配置文件不存在: {args.reflection_config}")
            sys.exit(1)
        with open(args.reflection_config, 'r', encoding='utf-8') as f:
            reflection_config = yaml.safe_load(f)
        reflection_params = reflection_config['reflection'].setdefault('params', {})
        if args.model_url:
            reflection_params['model_url'] = args.model_url
        if args.model_name:
            reflection_params['model_name'] = args.model_name
        config.setdefault('ep_config', {})['reflection'] = reflection_config['reflection']
    
    # Validate the configuration.
    if not validate_config(config, args):
        sys.exit(1)
    
    # Print configuration details.
    logger.info("=" * 60)
    logger.info("任务配置:")
    logger.info(f"  设备ID: {config['device']['id']}")
    logger.info(f"  任务描述: {args.purpose}")
    logger.info(f"  步数限制: {config['ep_config']['step_limit']}")
    model_endpoint = config['policy']['params'].get('model_url') or config['policy']['params'].get('model_host')
    logger.info(f"  模型地址: {model_endpoint}")
    logger.info(f"  模型名称: {config['policy']['params'].get('model_name', 'model')}")
    logger.info(f"  轨迹目录: {args.trace_dir}")
    if args.reflection:
        reflection_params = config['ep_config']['reflection'].get('params', {})
        logger.info(
            "  反思监督: 已启用 (模型: %s, 最大重试: %s)",
            reflection_params.get('model_name', 'model'),
            reflection_params.get('max_retries', 3),
        )
    logger.info("=" * 60)
    
    try:
        # Create and run the handler.
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
        
        # Unpack (is_successful, termination_reason, call_user_content).
        is_successful, termination_reason, call_user_content = result
        
        # Report the result for each termination reason.
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
