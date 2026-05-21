import socket


def get_lan_ip() -> str:
    """Infer local LAN IP without external network calls."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 80))
        lan_ip = sock.getsockname()[0]
        return lan_ip or "127.0.0.1"
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()
