from .structured import StructuredFeedClient


class RetrosheetClient(StructuredFeedClient):
    def __init__(self, url, http):
        super().__init__("Retrosheet", url, http)

