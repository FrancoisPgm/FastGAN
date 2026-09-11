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
stop_event = threading.Event()
is_waiting_for_reply = False


def zmq_loop():
    global is_waiting_for_reply, new_frame_event
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    # Create a Poller to monitor the socket
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print("ZMQ Thread: Connected to Server")

    while not stop_event.is_set():
        try:
            # 1. Wait for a frame from the main thread
            img = input_q.get(timeout=0.1)

            # 2. Send to Python Server
            socket.send(img.tobytes())
            is_waiting_for_reply = True

            # 3. POLL for the response (The "Smart Wait")
            # We loop here until the server responds OR the stop_event is triggered
            received = False
            while not stop_event.is_set():
                # poll() waits for 100ms to see if data is available
                socks = dict(poller.poll(100))

                if socket in socks and socks[socket] == zmq.POLLIN:
                    # Data is finally here!
                    reply = socket.recv()
                    received = True
                    break

            # 4. Process and put into output queue
            if received:
                processed_frame = np.frombuffer(reply, dtype=np.uint8).reshape(
                    img.shape
                )
                if output_q.full():
                    output_q.get()
                output_q.put(processed_frame)
                new_frame_event.set()
            is_waiting_for_reply = False

        except queue.Empty:
            continue
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
