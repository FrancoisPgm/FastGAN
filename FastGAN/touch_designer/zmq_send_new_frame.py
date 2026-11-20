# DAT execute (activate onFrameEnd)

import os
import zmq_manager

def onFrameEnd(frame: int):
    last_frame = me.storage.get("last_frame", -1)
    current_frame = op("movie1").index
    sleep = float(op("delay_entre_patchs")["sleep"])
    biome = os.path.basename(str(op("movie1").par.file)).split("_rush")[0]
    if current_frame - last_frame > 30 or last_frame > current_frame:
        zmq_manager.send_image("movie1", is_blast=False, biome=biome, sleep=sleep)
        me.storage["last_frame"] = current_frame
    return
