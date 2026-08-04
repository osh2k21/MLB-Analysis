from .structured import StructuredFeedClient


class FanGraphsClient(StructuredFeedClient):
    def __init__(self, url, http):
        super().__init__("FanGraphs", url, http)

