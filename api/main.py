try:
    from index import app  # noqa: F401
except ImportError:
    from api.index import app  # noqa: F401
