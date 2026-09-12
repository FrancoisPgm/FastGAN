# TOP script

import zmq_manager
import numpy as np


def onCook(scriptOp):
    q = zmq_manager.received_box_q

    if not q.empty():
        # Pull the processed frame from the global queue
        processed_frame = q.get()

        # Convert uint8 -> float32 for TD
        img_float = processed_frame.astype(np.float32) / 255.0
        scriptOp.copyNumpyArray(img_float)
    else:
        # Keep current image or scriptOp.clear()
        pass
