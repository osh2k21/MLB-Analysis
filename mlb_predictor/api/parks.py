from .structured import StructuredFeedClient


class ParkFactorClient(StructuredFeedClient):
    def __init__(self, url, http):
        super().__init__("Park factors", url, http)

