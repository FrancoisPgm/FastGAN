# DAT execute (frame start on)

import zmq_manager
import numpy as np

def onFrameStart(frame):
    # 1. THE DISPLAY: If Python has pushed a frame to the queue, trigger the cook
    if zmq_manager.new_frame_event.is_set():
        op("zmq_receiver_script").par.Cooktrigger += 1
        op("zmq_receiver_box").par.Cooktrigger += 1
        # Note: we don't return early here because we aren't waiting for a reply anymore
        zmq_manager.new_frame_event.clear()

