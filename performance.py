import time

global_start_at = time.time()
_tm_stack = []
class TimeMeasure:
    enabled = True

    def __init__(self, message: str = '', stack=False):
        self._start_at = None
        self._message = message
        self._stack = stack

    def __enter__(self):
        if not self.enabled: return
        self._start_at = time.time()
        if self._stack:
            _tm_stack.append(self._message)

    def __exit__(self, exception_type, exception_value, traceback):
        if not self.enabled: return
        if self._stack:
            _tm_stack.pop()
        message = '->'.join(_tm_stack + [self._message])
        if self.enabled:
            print(
                f'[TIME][{self._start_at - global_start_at:#.3f}-{time.time() - global_start_at:#.3f}s:{time.time() - self._start_at:#.5f}s]{message}')
