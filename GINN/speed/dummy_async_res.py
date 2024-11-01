class DummyAsyncResult():
    def __init__(self, result):
        self.result = result
    def ready(self):
        return True
    def get(self):
        return self.result