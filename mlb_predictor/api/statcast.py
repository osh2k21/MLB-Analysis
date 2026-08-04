from .structured import StructuredFeedClient


class StatcastClient(StructuredFeedClient):
    def __init__(self, url, http):
        super().__init__("Baseball Savant / Statcast", url, http)

