# DAT execute (activate onFrameEnd)

import zmq_manager


def onFrameEnd(frame: int):
    last_frame = me.storage.get("last_frame", -1)
    current_frame = op("movie1").index
    if current_frame - last_frame > 30:
        zmq_manager.send_image("movie1", is_blast=False)
        me.storage["last_frame"] = current_frame
    return
