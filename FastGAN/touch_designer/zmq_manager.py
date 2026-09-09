# DAT text

import zmq
import numpy as np
import threading
import queue

# These are now global and can be imported by any other script
input_q = queue.Queue(maxsize=1)
output_q = queue.Queue(maxsize=1)
thread = None
new_frame_event = threading.Event()


def zmq_loop():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    print("ZMQ Thread: Connected to Server")

    while True:
        try:
            # 1. Wait for a frame from the main thread
            img = input_q.get()

            # 2. Send to Python Server
            socket.send(img.tobytes())

            # 3. Wait for response
            reply = socket.recv()

            # 4. Process and put into output queue
            processed_frame = np.frombuffer(reply, dtype=np.uint8).reshape(img.shape)

            if output_q.full():
                output_q.get()
            output_q.put(processed_frame)
            new_frame_event.set()

        except Exception as e:
            print(f"ZMQ Thread Error: {e}")


def start_thread():
    global thread
    if thread is None or not thread.is_alive():
        thread = threading.Thread(target=zmq_loop, daemon=True)
        thread.start()
        print("ZMQ Thread: Started")
