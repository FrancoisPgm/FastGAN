# text DAT

import zmq
import numpy as np
import threading
import queue
import pickle

# receive queues size increased slightly to buffer the stream
send_q = queue.Queue(maxsize=1)
received_image_q = queue.Queue(maxsize=10)
received_box_q = queue.Queue(maxsize=10)
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
                send_packet = send_q.get_nowait()
                socket.send(pickle.dumps(send_packet))
                print("ZMQ Thread: Blast image sent!")
            except queue.Empty:
                pass

            # 2. POLL FOR RECEIVE: Check if the server has pushed a new frame
            socks = dict(poller.poll(10))  # Short poll to keep loop responsive
            if socket in socks and socks[socket] == zmq.POLLIN:
                reply = pickle.loads(socket.recv())

                # To handle reshaping, we'll assume standard size or store last known shape
                # For safety, we use a simple approach here:
                if received_image_q.full():
                    received_image_q.get()  # Drop oldest frame to keep stream real-time
                if received_box_q.full():
                    received_box_q.get()

                received_image_q.put(reply["image"].copy())
                received_box_q.put(reply["box"].copy())
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


def send_image(source_op_name="switch1", is_blast=True):
    """
    Starts a new cycle by sending a frame to the Python server.
    """
    top = op(source_op_name)
    if top:
        # 1. Capture the current frame
        img = (top.numpyArray() * 255).astype(np.uint8)
        packet = {"image": img, "is_blast": is_blast}

        # 2. Clear queues to ensure the new cycle starts fresh (no old frames)
        while not send_q.empty():
            send_q.get()
        while not received_image_q.empty():
            received_image_q.get()
        while not received_box_q.empty():
            received_box_q.get()

        # 3. Push the initial image to the background thread
        send_q.put(packet)
        print(f"Blast Triggered: Image from {source_op_name} in send queue.")
    else:
        print(f"Error: Could not find TOP named {source_op_name}")
