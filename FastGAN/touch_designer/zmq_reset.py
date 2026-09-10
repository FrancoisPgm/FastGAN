# DAT text

import zmq_manager
import time

# 1. Signal the thread to stop
zmq_manager.stop_thread()

# 2. Clear the queues
while not zmq_manager.input_q.empty():
    zmq_manager.input_q.get()
while not zmq_manager.output_q.empty():
    zmq_manager.output_q.get()

# 3. Give the thread a tiny moment to actually exit
time.sleep(0.2)

# 4. Start a fresh connection
zmq_manager.start_thread()

print("ZMQ Connection Reset Successfully.")
