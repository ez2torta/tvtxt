import os
import time
from uuid import uuid4
from typing import List
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
from pydantic import BaseModel, Field

# Shared utility to get image from URL

def get_image_from_url(image_url):
    img_byte_stream = BytesIO(urlopen(image_url).read())
    return Image.open(img_byte_stream).convert("RGB")

# Shared Pydantic models

class PlayerHUD(BaseModel):
    character: str = Field(..., description="Name of the character for this player.")
    health: str = Field(..., description="Health bar for the player, usually in a solid color which can be yellow (full or almost full) to red (dangerously low).")
    ratio: int = Field(..., ge=1, le=4, description="Ratio number for the current character (1-4).")
    guard_gauge: str = Field(..., description="Small bar below the health bar that shows the guard gauge status for the player.")
    position: List[int] = Field(..., description="Bounding box of the player in the format [x_min, y_min, x_max, y_max] in image coordinates.")


class FightingGameHUD(BaseModel):
    player1: PlayerHUD = Field(..., description="HUD information for the player on the left (Player 1).")
    player2: PlayerHUD = Field(..., description="HUD information for the player on the right (Player 2).")
    round_timer: int = Field(..., ge=0, le=999, description="This timer is in the upper middle of the screen and shows how much time is left in the round.")
    combo_counter_messages: str = Field(..., description="The numbers show how many hits you have done in a combo. Below it, there can also appear messages like 'Guard Crush', 'Reversal', 'Counter', and the type of K.O. in the end of a round.")
