# DAT execute (frame start on)

import zmq_manager
import numpy as np


def onFrameStart(frame):
    # --- PART 1: Pushing the image to Python ---
    top = op("movie1")  # NOM DE L'INPUT
    if top:
        img = (top.numpyArray() * 255).astype(np.uint8)

        try:
            if zmq_manager.input_q.full():
                zmq_manager.input_q.get()
            zmq_manager.input_q.put_nowait(img)
        except:
            pass
    # --- PART 2: Triggering the Cook ---
    if zmq_manager.new_frame_event.is_set():
        op("zmq_receiver_script").par.Cooktrigger = (
            op("zmq_receiver_script").par.Cooktrigger + 1
        ) % 1000
        zmq_manager.new_frame_event.clear()
