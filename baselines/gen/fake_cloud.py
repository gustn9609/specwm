"""[10/19 light device, scratch; 10/19 thesis: copied to baselines/gen/ for the queue memory check gen/queue/8/311] CPU-only stand-in for `two_machine --role cloud` to measure the DEVICE process's memory on one GPU.
Speaks the same protocol: READY -> (CLIP task N) -> PREPARED -> per DRAFT a CORR carrying the draft itself with rel 1.0 (> --thr 0, so the device
re-commits it whenever age < slots, as with the real cloud) after --lag-s seconds -> CLIP_DONE + READY after N drafts; END -> exit.
usage: python fake_cloud.py PORT [LAG_S]"""
import socket, struct, pickle, sys, time, threading, queue
import torch  # noqa: F401  (unpickles the fp16 CPU tensors)
port = int(sys.argv[1]); lag = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
def send(sock, obj):
    b = pickle.dumps(obj, protocol=4); sock.sendall(struct.pack("!I", len(b)) + b)
def recv_all(sock, n):
    buf = b""
    while len(buf) < n:
        p = sock.recv(n - len(buf))
        if not p: raise EOFError
        buf += p
    return buf
def recv(sock): return pickle.loads(recv_all(sock, struct.unpack("!I", recv_all(sock, 4))[0]))
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(("127.0.0.1", port)); srv.listen(1)
print("[fake cloud] listening on", port, flush=True)
conn, _ = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); send(conn, ("READY", ""))
lock = threading.Lock(); outq = queue.Queue()
def sender():   # delayed CORR messages in order
    while True:
        t_due, msg = outq.get()
        if msg is None: return
        d = t_due - time.time()
        if d > 0: time.sleep(d)
        with lock: send(conn, msg)
threading.Thread(target=sender, daemon=True).start()
task, N, n = None, 0, 0
while True:
    try: m = recv(conn)
    except EOFError: print("[fake cloud] device gone", flush=True); break
    if m[0] == "END": print("[fake cloud] END", flush=True); break
    if m[0] == "CLIP":
        task, N, n = m[1], m[2], 0
        with lock: send(conn, ("PREPARED", task))
        print("[fake cloud] clip", task, N, flush=True); continue
    if m[0] == "DRAFT":
        _, c, xd = m[:3]; n += 1
        outq.put((time.time() + lag, ("CORR", c, xd, 1.0, time.time() + lag, task)))
        if n == N:
            outq.put((time.time() + lag + 0.05, ("CLIP_DONE", task))); outq.put((time.time() + lag + 0.1, ("READY", task)))
outq.put((0, None)); time.sleep(0.5); conn.close(); srv.close()
