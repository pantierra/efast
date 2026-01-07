def __getattr__(name):
    if name == "fusion":
        from .efast import fusion

        return fusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["fusion"]
