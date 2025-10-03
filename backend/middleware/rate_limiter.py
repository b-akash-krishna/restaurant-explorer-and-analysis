from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure the rate limiter
# We use `get_remote_address` to identify the client, which is suitable for basic use cases.
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

def rate_limit_exceeded_handler(request, exc):
    """
    Custom handler for when a rate limit is exceeded.
    """
    return _rate_limit_exceeded_handler(request, exc)