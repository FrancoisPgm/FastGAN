import queue
import threading

import numpy as np
import zmq

# output_q size increased slightly to buffer the stream
input_q = queue.Queue(maxsize=1)
output_q = queue.Queue(maxsize=10)
thread = None
new_frame_event = threading.Event()
stop_event = threading.Event()

def zmq_loop():
    global new_frame_event
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)  # Changed from REQ to PAIR
    socket.connect("tcp://localhost:5555")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print("ZMQ Thread: Connected to Server (Stream Mode)")

    while not stop_event.is_set():
        try:
            # 1. NON-BLOCKING SEND: If a "Blast" image is in the queue, send it immediately
            try:
                img = input_q.get_nowait()
                socket.send(img.tobytes())
                print("ZMQ Thread: Blast image sent!")
            except queue.Empty:
                pass

            # 2. POLL FOR RECEIVE: Check if the server has pushed a new frame
            socks = dict(poller.poll(10))  # Short poll to keep loop responsive
            if socket in socks and socks[socket] == zmq.POLLIN:
                reply = socket.recv()

                # Process and push to output queue
                processed_frame = np.frombuffer(reply, dtype=np.uint8).reshape(
                    img.shape
                )

                # To handle reshaping, we'll assume standard size or store last known shape
                # For safety, we use a simple approach here:
                if output_q.full():
                    output_q.get()  # Drop oldest frame to keep stream real-time

                output_q.put(processed_frame)
                new_frame_event.set()

        except Exception as e:
            print(f"ZMQ Thread Error: {e}")
            break


def start_thread():
    global thread
    stop_event.clear()
    new_frame_event.clear()
    thread = threading.Thread(target=zmq_loop, daemon=True)
    thread.start()
    print("ZMQ Thread: Started")


def stop_thread():
    global thread
    stop_event.set()
    if thread:
        print("ZMQ Thread: Stopping...")


def trigger_blast(source_op_name="switch1"):
    """
    Starts a new cycle by sending a frame to the Python server.
    """
    top = op(source_op_name)
    if top:
        # 1. Capture the current frame
        img = (top.numpyArray() * 255).astype(np.uint8)

        # 2. Clear queues to ensure the new cycle starts fresh (no old frames)
        while not input_q.empty():
            input_q.get()
        while not output_q.empty():
            output_q.get()

        # 3. Push the initial image to the background thread
        input_q.put(img)
        print(f"Blast Triggered: Image from {source_op_name} in send queue.")
    else:
        print(f"Error: Could not find TOP named {source_op_name}")
