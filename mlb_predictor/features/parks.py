"""Maps live MLB venue names to the Retrosheet park (site) codes used by
``bundle.metadata["park_factors"]`` (computed during training from the same
Retrosheet game logs, see ``training/train.py``).

Venue names are looked up case-insensitively and stripped of sponsor-name
churn (e.g. "UNIQLO Field at Dodger Stadium" still resolves to Dodger
Stadium's park code) so a mid-season naming-rights change doesn't silently
drop the feature back to neutral.
"""

VENUE_TO_SITE = {
    "angel stadium": "ANA01",
    "globe life field": "ARL03",
    "truist park": "ATL03",
    "oriole park at camden yards": "BAL12",
    "fenway park": "BOS07",
    "sahlen field": "BUF05",
    "wrigley field": "CHI11",
    "rate field": "CHI12",
    "guaranteed rate field": "CHI12",
    "great american ball park": "CIN09",
    "progressive field": "CLE08",
    "coors field": "DEN02",
    "comerica park": "DET05",
    "td ballpark": "DUN01",
    "daikin park": "HOU03",
    "minute maid park": "HOU03",
    "kauffman stadium": "KAN06",
    "dodger stadium": "LOS03",
    "loandepot park": "MIA02",
    "american family field": "MIL06",
    "target field": "MIN04",
    "citi field": "NYC20",
    "yankee stadium": "NYC21",
    "oakland coliseum": "OAK01",
    "citizens bank park": "PHI13",
    "chase field": "PHO01",
    "pnc park": "PIT08",
    "sutter health park": "SAC01",
    "petco park": "SAN02",
    "t-mobile park": "SEA03",
    "oracle park": "SFO03",
    "busch stadium": "STL10",
    "tropicana field": "STP01",
    "george m. steinbrenner field": "TAM02",
    "steinbrenner field": "TAM02",
    "rogers centre": "TOR02",
    "nationals park": "WAS11",
}


def _normalize(venue_name: str) -> str:
    lowered = venue_name.strip().lower()
    for prefix in ("uniqlo field at ", "at&t "):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
    return lowered


def park_run_factor(venue_name: str, park_factors: dict) -> float:
    """Trailing park run factor for a venue, or 1.0 (neutral) if unmapped or unseen in training."""
    if not venue_name or not park_factors:
        return 1.0
    site = VENUE_TO_SITE.get(_normalize(venue_name))
    if site is None:
        return 1.0
    return float(park_factors.get(site, 1.0))
