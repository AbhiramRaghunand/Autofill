from slowapi import Limiter
from slowapi.util import get_remote_address

# One shared limiter instance for the entire application and all routers.
limiter = Limiter(key_func=get_remote_address)
