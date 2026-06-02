class AIError(Exception):
    """AI 模块基础异常"""
    pass

class APIAuthError(AIError):
    """API 认证失败"""
    pass

class APIRateLimitError(AIError):
    """API 限流"""
    pass

class APITimeoutError(AIError):
    """API 超时"""
    pass

class APIServerError(AIError):
    """API 服务端错误"""
    pass

class ContextTooLongError(AIError):
    """上下文过长"""
    pass

class DataInsufficientError(AIError):
    """数据不足"""
    pass


class AIUnavailableError(AIError):
    """AI 服务熔断降级 — 快速失败"""
    pass
