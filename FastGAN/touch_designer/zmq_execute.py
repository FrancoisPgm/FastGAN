# DAT execute (frame start on)

import zmq_manager
import numpy as np

# Local variable to track if we just triggered a cook
is_cooking_current_frame = False

def onFrameStart(frame):
    global is_cooking_current_frame
    top = op("switch1")

    # 1. THE TRIGGER: If Python has sent a new image, trigger the cook
    if zmq_manager.new_frame_event.is_set():
        op("zmq_receiver_script").par.Cooktrigger += 1
        top.par.index = 1
        zmq_manager.new_frame_event.clear()
        is_cooking_current_frame = True  # Mark that we are updating the image
        return  # Exit early to let TD actually perform the cook

    # 2. THE RELEASE: If we were cooking, wait one frame for TD to finish rendering
    if is_cooking_current_frame:
        is_cooking_current_frame = False
        return  # Exit early to ensure the image is on screen before sending back

    # 3. THE SEND: Only send if Python is NOT busy and we aren't cooking
    if not zmq_manager.is_waiting_for_reply:
        # Get the image from the FINAL TOP in your chain (including TD edits)
        if top:
            img = (top.numpyArray() * 255).astype(np.uint8)
            try:
                if zmq_manager.input_q.full():
                    zmq_manager.input_q.get()
                zmq_manager.input_q.put_nowait(img)
            except:
                pass
