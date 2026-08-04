from .structured import StructuredFeedClient


class UmpireClient(StructuredFeedClient):
    def __init__(self, url, http):
        super().__init__("Umpire assignments", url, http)

