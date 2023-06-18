class cached_property:
    'Read-only and cached property'

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, objtype):
        try:
            return self._cache
        except AttributeError:
            self._cache = self.fget(obj)
            return self._cache

    def __set__(self, obj, value):
        self._cache = value
