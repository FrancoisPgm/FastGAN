# panel exec DAT

"""
Panel Execute DAT

me - this DAT

panelValue - the PanelValue object that changed
prev - the previous value of the PanelValue object that changed

Make sure the corresponding toggle is enabled in the Panel Execute DAT.
"""

import os
from typing import Any

import zmq_manager


def onOffToOn(panelValue: PanelValue):
    """
    Called when a panel value changes from 0 to non-zero.
    """
    sleep = float(op("delay_entre_patchs")["sleep"])
    biome = os.path.basename(str(op("movie1").par.file)).split("_rush")[0]
    zmq_manager.send_image("movie1", is_blast=True, biome=biome, sleep=sleep)
