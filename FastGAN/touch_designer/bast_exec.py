# panel exec DAT

"""
Panel Execute DAT

me - this DAT

panelValue - the PanelValue object that changed
prev - the previous value of the PanelValue object that changed

Make sure the corresponding toggle is enabled in the Panel Execute DAT.
"""

from typing import Any

import zmq_manager


def onOffToOn(panelValue: PanelValue):
    """
    Called when a panel value changes from 0 to non-zero.
    """
    zmq_manager.trigger_blast("movie1")
