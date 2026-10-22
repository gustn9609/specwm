"""[10/15 two-machine] transparent TCP relay: listen on --port, forward every connection to --to host:port (both directions). Run on a third
host (lamarr) so the device<->cloud traffic of specwm.two_machine crosses the real network twice instead of the loopback.
usage: python3 tcp_relay.py --port 31501 --to spi0.ece.utexas.edu:31500"""
import argparse, socket, threading
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, required=True); ap.add_argument("--to", required=True); a = ap.parse_args()
host, port = a.to.rsplit(":", 1)
def pipe(src, dst):
    try:
        while True:
            b = src.recv(1 << 16)
            if not b: break
            dst.sendall(b)
    except Exception: pass
    finally:
        try: dst.shutdown(socket.SHUT_WR)
        except Exception: pass
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(("0.0.0.0", a.port)); srv.listen(4); print("relay %d -> %s" % (a.port, a.to), flush=True)
while True:
    c, addr = srv.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    u = socket.create_connection((host, int(port))); u.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); print("conn from", addr, flush=True)
    threading.Thread(target=pipe, args=(c, u), daemon=True).start(); threading.Thread(target=pipe, args=(u, c), daemon=True).start()
