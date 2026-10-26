"""Execute a materialized method with process-local outbound network denial.

Local IPC remains available to numerical/data-loading libraries. Asset fetching
is a separate explicit command, never an incidental analysis-stage side effect.
"""

from pathlib import Path
import ipaddress
import os
import runpy
import socket
import sys


def _loopback(host):
    if host in ("localhost", b"localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def deny_outbound_network():
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def permitted(sock, address):
        return (hasattr(socket, "AF_UNIX") and sock.family == socket.AF_UNIX) or (
            isinstance(address, tuple) and _loopback(address[0])
        )

    def connect(sock, address):
        if not permitted(sock, address):
            raise OSError(
                "Outbound network is disabled during analysis; prepare pinned assets first"
            )
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if not permitted(sock, address):
            raise OSError("Outbound network is disabled during analysis")
        return original_connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if not _loopback(host):
            raise OSError("Outbound name resolution is disabled during analysis")
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo


def main():
    root = Path(os.environ["CSI_MAIZE_WORKSPACE"]).absolute()
    base = root / "prototype/research_v2"
    relative = Path(sys.argv[1])
    script = base / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not script.resolve().is_relative_to(base.resolve())
    ):
        raise ValueError("Method script must be contained in the materialized code directory")
    if script.suffix != ".py" or not script.is_file():
        raise ValueError("Method script is missing")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", YOLO_AUTOINSTALL="false")
    deny_outbound_network()
    sys.path.insert(0, str(base))
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
