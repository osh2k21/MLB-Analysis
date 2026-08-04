from .structured import StructuredFeedClient


class InjuryClient(StructuredFeedClient):
    def __init__(self, url, http):
        super().__init__("Injury feed", url, http)

