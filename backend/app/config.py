"""
配置管理
统一从项目根目录的 .env 文件加载配置
"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False
    
    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # Graph memory配置。生产默认使用本地 Graphiti；Zep Cloud 保留为回退。
    MEMORY_PROVIDER = os.environ.get('MEMORY_PROVIDER', 'graphiti').lower()

    # Zep配置
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')

    # Graphiti配置
    GRAPHITI_BASE_URL = os.environ.get('GRAPHITI_BASE_URL', 'http://127.0.0.1:8010')
    GRAPHITI_NEO4J_URI = os.environ.get('GRAPHITI_NEO4J_URI', 'bolt://127.0.0.1:17687')
    GRAPHITI_NEO4J_USER = os.environ.get('GRAPHITI_NEO4J_USER', 'neo4j')
    GRAPHITI_NEO4J_PASSWORD = os.environ.get('GRAPHITI_NEO4J_PASSWORD')
    GRAPHITI_API_KEY = os.environ.get('GRAPHITI_API_KEY')
    GRAPHITI_LLM_BASE_URL = os.environ.get('GRAPHITI_LLM_BASE_URL', LLM_BASE_URL)
    GRAPHITI_LLM_MODEL_NAME = os.environ.get('GRAPHITI_LLM_MODEL_NAME', 'gpt-4.1-mini')
    GRAPHITI_SMALL_MODEL_NAME = os.environ.get('GRAPHITI_SMALL_MODEL_NAME', 'gpt-4.1-nano')
    GRAPHITI_EMBEDDING_BASE_URL = os.environ.get('GRAPHITI_EMBEDDING_BASE_URL', LLM_BASE_URL)
    GRAPHITI_EMBEDDING_MODEL = os.environ.get('GRAPHITI_EMBEDDING_MODEL', 'text-embedding-3-small')
    GRAPHITI_EMBEDDING_DIM = int(os.environ.get('GRAPHITI_EMBEDDING_DIM', '1024'))
    GRAPHITI_EPISODE_TIMEOUT_SECONDS = int(os.environ.get('GRAPHITI_EPISODE_TIMEOUT_SECONDS', '90'))
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls):
        """验证必要配置"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")
        errors.extend(cls.validate_graph_memory())
        return errors

    @classmethod
    def validate_graph_memory(cls):
        """按当前 memory provider 验证图谱记忆配置。"""
        errors = []
        if cls.MEMORY_PROVIDER == 'zep':
            if not cls.ZEP_API_KEY:
                errors.append("ZEP_API_KEY 未配置")
        elif cls.MEMORY_PROVIDER == 'graphiti':
            if not cls.GRAPHITI_NEO4J_URI:
                errors.append("GRAPHITI_NEO4J_URI 未配置")
            if not cls.GRAPHITI_NEO4J_USER:
                errors.append("GRAPHITI_NEO4J_USER 未配置")
            if not cls.GRAPHITI_NEO4J_PASSWORD:
                errors.append("GRAPHITI_NEO4J_PASSWORD 未配置")
            if not (cls.GRAPHITI_API_KEY or cls.LLM_API_KEY):
                errors.append("GRAPHITI_API_KEY 或 LLM_API_KEY 未配置")
        else:
            errors.append(f"MEMORY_PROVIDER 不支持: {cls.MEMORY_PROVIDER}")
        return errors
