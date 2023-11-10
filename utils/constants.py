"""Project's constant variables and API keys.

Heavily inspired by
https://github.com/allenai/real-toxicity-prompts/blob/master/utils/constants.py
"""

import os

PERSPECTIVE_API_KEY = os.environ.get("PERSPECTIVE_API_KEY", None)

# All attributes can be found here:
# https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md
PERSPECTIVE_API_ATTRIBUTES = (
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "THREAT",
    "PROFANITY",
    "SEXUALLY_EXPLICIT",
    "FLIRTATION",
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
