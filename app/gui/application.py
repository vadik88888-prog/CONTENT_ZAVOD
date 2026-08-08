from __future__ import annotations

from hashlib import sha256
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow
from app.gui.services.desktop_services import DesktopServices
from app.gui.styles import load_theme


# Qt permits exactly one application object per process.  Keep the desktop
# shell equally singular: an accidental second invocation should focus the
# window a person is already using instead of creating another project flow
# (and another set of media players and modal dialogs).
_main_window: MainWindow | None = None
_instance_server: QLocalServer | None = None
_instance_server_key: str | None = None
_pending_instance_activation = False


def application_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _instance_server_name() -> str:
    """Return a stable, filesystem-safe endpoint for this desktop install.

    A path digest lets two unrelated developer checkouts run independently,
    while separate launches of this installation always address the same local
    endpoint.  ``QLocalServer`` scopes the transport to the local machine.
    """

    root = str(application_root().resolve()).encode("utf-8", "surrogatepass")
    return f"content-factory-desktop-{sha256(root).hexdigest()[:24]}"


def _dispose_local_socket(socket: QLocalSocket) -> None:
    """Release a one-shot activation connection without surfacing Qt teardown errors."""

    try:
        socket.disconnectFromServer()
    except RuntimeError:
        pass
    try:
        socket.deleteLater()
    except RuntimeError:
        pass


def _activate_existing_instance(app: QApplication, server: QLocalServer) -> None:
    """Accept every local activation ping and focus this process's window."""

    global _pending_instance_activation
    received = False
    while True:
        try:
            socket = server.nextPendingConnection()
        except RuntimeError:
            return
        if socket is None:
            break
        received = True
        _dispose_local_socket(socket)
    if not received:
        return
    window = _live_main_window(app)
    if window is None:
        # The listener is acquired immediately before constructing the first
        # shell.  Preserve a rare early ping until that shell exists instead
        # of silently creating a second visible launch.
        _pending_instance_activation = True
        return
    _pending_instance_activation = False
    _activate(window)


def _notify_existing_instance(server_name: str) -> bool:
    """Ask a running desktop process to focus itself, if one owns the endpoint."""

    socket = QLocalSocket()
    try:
        socket.connectToServer(server_name)
        if not socket.waitForConnected(300):
            return False
        socket.write(b"activate")
        socket.flush()
        socket.waitForBytesWritten(150)
        return True
    finally:
        _dispose_local_socket(socket)


def _release_instance_server() -> None:
    """Stop the listener when this QApplication exits.

    ``QLocalServer.close`` releases a cleanly-owned endpoint; the explicit
    stale-endpoint cleanup below is intentionally used only before claiming a
    new listener, so a shutdown cannot remove a freshly-started peer.
    """

    global _instance_server, _instance_server_key, _pending_instance_activation
    server = _instance_server
    _instance_server = None
    _instance_server_key = None
    _pending_instance_activation = False
    if server is None:
        return
    try:
        server.close()
    except RuntimeError:
        pass
    try:
        server.deleteLater()
    except RuntimeError:
        pass


def _start_instance_server(app: QApplication) -> bool:
    """Claim the process-wide desktop endpoint, or activate its current owner.

    ``False`` means a separate process is already serving the desktop shell.
    A failed listen after two failed connections is treated as a stale local
    endpoint and removed once before the final listen attempt.
    """

    global _instance_server, _instance_server_key
    if _instance_server is not None:
        return True

    server_name = _instance_server_name()
    if _notify_existing_instance(server_name):
        return False

    server = QLocalServer()
    if not server.listen(server_name):
        # A process can win the race between the initial connect and listen.
        # Prefer its window whenever it became reachable.
        if _notify_existing_instance(server_name):
            return False
        # A graceful close removes the endpoint.  This fallback only handles
        # stale names left by a crash (not a listener we successfully reached).
        QLocalServer.removeServer(server_name)
        if not server.listen(server_name):
            if _notify_existing_instance(server_name):
                return False
            try:
                server.deleteLater()
            except RuntimeError:
                pass
            raise RuntimeError("Content Factory cannot claim its single-instance endpoint.")

    server.newConnection.connect(lambda: _activate_existing_instance(app, server))
    _instance_server = server
    _instance_server_key = server_name
    try:
        app.aboutToQuit.connect(_release_instance_server)
    except (AttributeError, RuntimeError):
        # The fallback keeps embedded/test application hosts usable; ``run``
        # also releases the server after its own event loop returns.
        pass
    return True


def _desktop_application(argv: list[str] | None) -> QApplication:
    """Return the process QApplication without trying to upgrade QCoreApplication.

    Test hosts and command-line integrations occasionally already own a
    QCoreApplication.  Qt cannot safely replace that object with QApplication,
    so fail with a precise message rather than letting Qt abort the process.
    """

    existing = QCoreApplication.instance()
    if existing is None:
        return QApplication(argv if argv is not None else sys.argv)
    if not isinstance(existing, QApplication):
        raise RuntimeError(
            "Content Factory desktop requires QApplication, but this process already owns QCoreApplication."
        )
    return existing


def _live_main_window(app: QApplication) -> MainWindow | None:
    """Find the existing Content Factory shell, including one made by a host."""

    global _main_window
    candidates = [_main_window, *app.topLevelWidgets()]
    for candidate in candidates:
        if candidate is None or not isinstance(candidate, MainWindow):
            continue
        try:
            # A deleted PySide wrapper raises RuntimeError when accessed.
            candidate.windowTitle()
        except RuntimeError:
            if candidate is _main_window:
                _main_window = None
            continue
        _main_window = candidate
        return candidate
    return None


def _activate(window: MainWindow) -> None:
    """Bring the one existing shell forward without starting a nested event loop."""

    if window.isMinimized():
        window.showNormal()
    else:
        window.show()
    window.raise_()
    window.activateWindow()


def run(argv: list[str] | None = None) -> int:
    global _main_window
    app = _desktop_application(argv)
    app.setApplicationName("Content Factory")
    app.setOrganizationName("Content Factory")
    app.setStyleSheet(load_theme())

    existing = _live_main_window(app)
    if existing is not None:
        _activate(existing)
        return 0
    if not _start_instance_server(app):
        return 0

    try:
        window = MainWindow(DesktopServices.create(application_root()))
        _main_window = window
        window.show()
        if _pending_instance_activation:
            _activate(window)
        return app.exec()
    finally:
        # ``aboutToQuit`` normally does this first.  The finally clause also
        # covers a test host, a constructor failure, or an event loop that
        # returns without delivering the Qt shutdown signal.
        _release_instance_server()
