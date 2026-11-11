
import json
import os
import sys
import threading
import subprocess
import mimetypes
import webbrowser
import signal

import hashlib
import hmac
import secrets
import time

import urllib.request
import urllib.error

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

TRACKING_SCRIPT = BASE_DIR / "human_tracking_dual_cctv_cuda.py"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "tracking_engine.log"

FACE_SCRIPT = BASE_DIR / "face_recognition_only.py"
FACE_LOG_FILE = LOG_DIR / "face_recognition_engine.log"

HOST = "127.0.0.1"
PORT = 8000
VIEWER_PORT = 8080
GPU_DEVICE = "0"

FACE_VIEWER_PORT = 8081
FACE_GPU_DEVICE = "0"
FACE_MODEL = "buffalo_l"


# =====================================================
# AUTHENTICATION CONFIGURATION
# =====================================================

AUTH_CONFIG_FILE = BASE_DIR / ".env"

def load_env_file():
    """Membaca konfigurasi sederhana dari file .env."""
    if not AUTH_CONFIG_FILE.exists():
        return

    for line in AUTH_CONFIG_FILE.read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()

ADMIN_USERNAME = os.getenv(
    "GATEWATCH_ADMIN_USERNAME",
    ""
)

ADMIN_PASSWORD_HASH = os.getenv(
    "GATEWATCH_ADMIN_PASSWORD_HASH",
    ""
)

SESSION_TTL = int(
    os.getenv("GATEWATCH_SESSION_TTL", "28800")
)

COOKIE_SECURE = (
    os.getenv("GATEWATCH_COOKIE_SECURE", "0") == "1"
)

SESSION_COOKIE_NAME = "gatewatch_session"

# Penyimpanan session sementara di memory.
# Session akan berakhir jika server dimatikan.
sessions = {}
session_lock = threading.Lock()


def verify_password(password, stored_hash):
    """Memverifikasi password menggunakan PBKDF2-SHA256."""
    try:
        algorithm, iterations, salt, expected = stored_hash.split("$")

        if algorithm != "pbkdf2_sha256":
            return False

        calculated = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            int(iterations)
        ).hex()

        return hmac.compare_digest(calculated, expected)

    except (ValueError, TypeError):
        return False


def create_session():
    token = secrets.token_urlsafe(48)
    expires_at = time.time() + SESSION_TTL

    with session_lock:
        sessions[token] = expires_at

    return token


def get_valid_session(token):
    if not token:
        return False

    now = time.time()

    with session_lock:
        # Bersihkan session yang kedaluwarsa.
        expired = [
            key for key, expiry in sessions.items()
            if expiry <= now
        ]

        for key in expired:
            sessions.pop(key, None)

        expiry = sessions.get(token)

        if expiry is None or expiry <= now:
            sessions.pop(token, None)
            return False

        return True


def delete_session(token):
    if not token:
        return

    with session_lock:
        sessions.pop(token, None)

face_process = None
tracking_process = None
process_lock = threading.Lock()


def get_status():
    global tracking_process

    with process_lock:
        if tracking_process is None:
            return {
                "running": False,
                "pid": None,
                "status": "STOPPED"
            }

        exit_code = tracking_process.poll()

        if exit_code is not None:
            tracking_process = None

            return {
                "running": False,
                "pid": None,
                "status": "STOPPED",
                "exit_code": exit_code
            }

        return {
            "running": True,
            "pid": tracking_process.pid,
            "status": "RUNNING"
        }

def start_tracking():
    global tracking_process

    with process_lock:

        if tracking_process is not None:
            if tracking_process.poll() is None:
                return False, "Tracking sudah berjalan."

            tracking_process = None

        if not TRACKING_SCRIPT.exists():
            return False, "File tracking tidak ditemukan."

        LOG_DIR.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["INFER_DEVICE"] = GPU_DEVICE
        env["VIEWER_PORT"] = str(VIEWER_PORT)

        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log:

                tracking_process = subprocess.Popen(
                    [
                        sys.executable,
                        str(TRACKING_SCRIPT),
                        "--headless"
                    ],
                    cwd=str(BASE_DIR),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )

            return True, "Tracking berhasil dijalankan."

        except Exception as exc:
            tracking_process = None
            return False, str(exc)

def start_face_recognition():
    global face_process

    with process_lock:
        if face_process is not None:
            if face_process.poll() is None:
                return False, "Face Recognition sudah berjalan."

            face_process = None

        if not FACE_SCRIPT.exists():
            return False, f"File Face Recognition tidak ditemukan: {FACE_SCRIPT.name}"

        LOG_DIR.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()

        env["FACE_GPU"] = FACE_GPU_DEVICE
        env["FACE_MODEL"] = FACE_MODEL
        env["FACE_VIEWER_PORT"] = str(FACE_VIEWER_PORT)

        try:
            with open(FACE_LOG_FILE, "a", encoding="utf-8") as log:
                face_process = subprocess.Popen(
                    [
                        sys.executable,
                        str(FACE_SCRIPT)
                    ],
                    cwd=str(BASE_DIR),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )

            return True, "Face Recognition berhasil dijalankan."

        except Exception as exc:
            face_process = None
            return False, str(exc)

def stop_tracking():
    global tracking_process

    with process_lock:

        if tracking_process is None:
            return True, "Tracking sudah berhenti."

        if tracking_process.poll() is not None:
            tracking_process = None
            return True, "Tracking sudah berhenti."

        try:
            # Kirim sinyal penghentian ke proses Python.
            tracking_process.send_signal(
                signal.SIGBREAK
            )

            tracking_process.wait(timeout=10)

        except subprocess.TimeoutExpired:
            tracking_process.kill()
            tracking_process.wait(timeout=5)

        except Exception:
            tracking_process.kill()
            tracking_process.wait(timeout=5)

        tracking_process = None

        return True, "Tracking berhasil dihentikan."

def stop_face_recognition():
    global face_process

    with process_lock:
        if face_process is None:
            return True, "Face Recognition sudah berhenti."

        if face_process.poll() is not None:
            face_process = None
            return True, "Face Recognition sudah berhenti."

        try:
            face_process.send_signal(signal.SIGBREAK)
            face_process.wait(timeout=10)

        except subprocess.TimeoutExpired:
            face_process.kill()
            face_process.wait(timeout=5)

        except Exception:
            face_process.kill()
            face_process.wait(timeout=5)

        face_process = None

        return True, "Face Recognition berhasil dihentikan."

def proxy_viewer_api(path, port=None):
    if port is None:
        port = VIEWER_PORT

    url = f"http://127.0.0.1:{port}{path}"

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, response.read()

    except Exception:
        return None, None

def get_dashboard_status():
    """
    Mengambil status dashboard dari YOLO Viewer (port 8080).
    Tidak mengubah atau mengganggu endpoint Face Recognition.
    """

    code, body = proxy_viewer_api(
        "/api/status",
        VIEWER_PORT
    )

    if code == 200 and body:
        try:
            return json.loads(body.decode("utf-8"))

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[YOLO] Invalid status response: {exc}")

    process = get_status()

    return {
        "running": process["running"],
        "pid": process["pid"],
        "status": process["status"],
        "error": "YOLO Viewer belum tersedia."
    }

def get_face_dashboard_status():
    process = get_face_status()

    if not process["running"]:
        return {
            **process,
            "camera_connected": False,
            "camera_error": "",
            "fps": 0,
            "faces_detected": 0,
            "recognized_count": 0,
            "model": FACE_MODEL,
            "gpu_id": int(FACE_GPU_DEVICE),
            "det_ms": 0,
            "embedding_ms": 0,
            "faces": []
        }

    code, body = proxy_viewer_api(
        "/api/status",
        FACE_VIEWER_PORT
    )

    if code == 200:
        try:
            data = json.loads(body)
            data.update(process)
            return data
        except json.JSONDecodeError:
            pass

    return {
        **process,
        "status": "STARTING",
        "camera_connected": False,
        "camera_error": "Menunggu Face Recognition Viewer...",
        "fps": 0,
        "faces_detected": 0,
        "recognized_count": 0,
        "model": FACE_MODEL,
        "gpu_id": int(FACE_GPU_DEVICE),
        "det_ms": 0,
        "embedding_ms": 0,
        "faces": []
    }

def get_face_status():
    global face_process

    with process_lock:
        if face_process is None:
            return {
                "running": False,
                "pid": None,
                "status": "STOPPED"
            }

        exit_code = face_process.poll()

        if exit_code is not None:
            face_process = None

            return {
                "running": False,
                "pid": None,
                "status": "STOPPED",
                "exit_code": exit_code
            }

        return {
            "running": True,
            "pid": face_process.pid,
            "status": "RUNNING"
        }

class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass
    
    
    def get_session_token(self):
        cookie_header = self.headers.get("Cookie", "")

        for item in cookie_header.split(";"):
            item = item.strip()

            if item.startswith(SESSION_COOKIE_NAME + "="):
                return item.split("=", 1)[1]

        return None

    def is_authenticated(self):
        return get_valid_session(self.get_session_token())

    def send_redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_json(self, data, code=200, extra_headers=None):
        body = json.dumps(data).encode("utf-8")

        try:
            self.send_response(code)

            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.send_header("Cache-Control", "no-store")
            self.send_connection_close = True

            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)

            self.end_headers()
            self.wfile.write(body)

        except (
            ConnectionAbortedError,
            ConnectionResetError,
            BrokenPipeError
        ):
            pass

    def send_session_cookie(self, token):
        secure = "; Secure" if COOKIE_SECURE else ""

        return (
            f"{SESSION_COOKIE_NAME}={token}; "
            f"Path=/; "
            f"HttpOnly; "
            f"SameSite=Strict; "
            f"Max-Age={SESSION_TTL}"
            f"{secure}"
        )

    def clear_session_cookie(self):
        secure = "; Secure" if COOKIE_SECURE else ""

        return (
            f"{SESSION_COOKIE_NAME}=; "
            f"Path=/; "
            f"HttpOnly; "
            f"SameSite=Strict; "
            f"Max-Age=0"
            f"{secure}"
        )

    def send_json(self, data, code=200):
        body = json.dumps(data).encode("utf-8")

        try:
            self.send_response(code)

            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.send_header("Cache-Control", "no-store")
            self.send_connection_close = True

            self.end_headers()
            self.wfile.write(body)

        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # Client menutup koneksi sebelum respons selesai.
            pass
        
    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        # =============================================
        # PUBLIC: LOGIN PAGE
        # =============================================

        if path == "/login.html":
            if self.is_authenticated():
                self.send_redirect("/")
                return

        # =============================================
        # PROTECTED: API
        # =============================================

        if path.startswith("/api/"):
            if not self.require_authentication():
                return

            if path == "/api/auth/me":
                self.send_json({
                    "success": True,
                    "authenticated": True,
                    "username": ADMIN_USERNAME
                })
                return

            if path == "/api/status":
                self.send_json(get_dashboard_status())
                return

            if path == "/api/control/status":
                self.send_json(get_status())
                return

            if path == "/api/face/control/status":
                self.send_json(get_face_status())
                return

            if path == "/api/face/status":
                self.send_json(get_face_dashboard_status())
                return

            if path == "/api/logs":
                code, body = proxy_viewer_api("/api/logs")

                if code == 200:
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "application/json; charset=utf-8"
                    )
                    self.send_header(
                        "Content-Length",
                        str(len(body))
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_json([])

                return

        # =============================================
        # PROTECTED: HTML PAGES
        # =============================================

        protected_pages = {
            "/",
            "/index.html",
            "/dashboard.html",
            "/monitoring.html",
            "/face-recognition.html",
            "/logs.html"
        }

        if path in protected_pages and not self.is_authenticated():
            self.send_redirect("/login.html")
            return

        # =============================================
        # STATIC FILE SERVER
        # =============================================

        relative = "index.html" if path == "/" else path.lstrip("/")

        try:
            root = WEB_DIR.resolve()
            file_path = (root / relative).resolve()

            file_path.relative_to(root)

            if not file_path.is_file():
                raise FileNotFoundError()

            body = file_path.read_bytes()

        except (OSError, ValueError):
            self.send_json({
                "error": "File tidak ditemukan."
            }, 404)
            return

        content_type, _ = mimetypes.guess_type(str(file_path))

        self.send_response(200)
        self.send_header(
            "Content-Type",
            content_type or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    
    def do_POST(self):
        path = urlparse(self.path).path

        # =============================================
        # LOGIN
        # =============================================

        if path == "/api/auth/login":
            if not self.check_request_origin():
                self.send_json({
                    "success": False,
                    "message": "Request origin tidak valid."
                }, 403)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))

                if length <= 0 or length > 10000:
                    raise ValueError("Ukuran request tidak valid.")

                payload = json.loads(
                    self.rfile.read(length).decode("utf-8")
                )

                username = payload.get("username", "")
                password = payload.get("password", "")

                if not isinstance(username, str) or not isinstance(password, str):
                    raise ValueError("Input tidak valid.")

            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                self.send_json({
                    "success": False,
                    "message": "Data login tidak valid."
                }, 400)
                return

            if not ADMIN_USERNAME or not ADMIN_PASSWORD_HASH:
                self.send_json({
                    "success": False,
                    "message": "Konfigurasi Administrator belum tersedia."
                }, 500)
                return

            valid_username = hmac.compare_digest(
                username.encode("utf-8"),
                ADMIN_USERNAME.encode("utf-8")
            )

            valid_password = verify_password(
                password,
                ADMIN_PASSWORD_HASH
            )

            if not (valid_username and valid_password):
                self.send_json({
                    "success": False,
                    "message": "Username atau password salah."
                }, 401)
                return
            
            # Login berhasil
            token = create_session()

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header(
                "Set-Cookie",
                self.send_session_cookie(token)
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            self.wfile.write(json.dumps({
                "success": True,
                "message": "Login berhasil.",
                "redirect": "/"
            }).encode("utf-8"))

            return

        # =============================================
        # LOGOUT
        # =============================================

        if path == "/api/auth/logout":
            if not self.check_request_origin():
                self.send_json({
                    "success": False,
                    "message": "Request origin tidak valid."
                }, 403)
                return

            token = self.get_session_token()
            delete_session(token)

            self.send_json({
                "success": True,
                "message": "Logout berhasil."
            }, 200, {
                "Set-Cookie": self.clear_session_cookie()
            })

            return

        # =============================================
        # PROTECTED CONTROL ENDPOINTS
        # =============================================

        if not self.require_authentication():
            return

        if not self.check_request_origin():
            self.send_json({
                "success": False,
                "message": "Request origin tidak valid."
            }, 403)
            return

        if path == "/api/control/start":
            success, message = start_tracking()
            process = get_status()

        elif path == "/api/control/stop":
            success, message = stop_tracking()
            process = get_status()

        elif path == "/api/face/control/start":
            success, message = start_face_recognition()
            process = get_face_status()

        elif path == "/api/face/control/stop":
            success, message = stop_face_recognition()
            process = get_face_status()

        else:
            self.send_json({
                "error": "Endpoint tidak ditemukan."
            }, 404)
            return

        self.send_json({
            "success": success,
            "message": message,
            "process": process
        }, 200 if success else 409)
            
    def require_authentication(self):
        if self.is_authenticated():
            return True

        self.send_json({
            "success": False,
            "message": "Sesi login tidak valid atau sudah berakhir.",
            "authenticated": False
        }, 401)

        return False

    def check_request_origin(self):
        """
        Mencegah request POST lintas situs.
        Request tanpa Origin tetap diperbolehkan untuk
        kompatibilitas klien non-browser lokal.
        """
        origin = self.headers.get("Origin")

        if not origin:
            return True

        origin_host = urlparse(origin).netloc.lower()
        request_host = self.headers.get("Host", "").lower()

        return origin_host == request_host    


if __name__ == "__main__":

    if os.name != "nt":
        raise RuntimeError("Khusus Windows")

    import signal
    
    if not ADMIN_USERNAME or not ADMIN_PASSWORD_HASH:
        raise RuntimeError(
            "Konfigurasi Administrator belum tersedia. Periksa file .env."
        )

    server = ThreadingHTTPServer(
        (HOST, PORT),
        DashboardHandler
    )

    print("======================================")
    print(" GATEWATCH CONTROL CENTER")
    print("======================================")
    print(f"Dashboard : http://localhost:{PORT}")
    print(f"Viewer    : http://localhost:{VIEWER_PORT}")
    print("Tracking  : OFF")
    print("======================================")

    # Buka browser setelah server siap menerima koneksi
    threading.Timer(
        1.5,
        lambda: webbrowser.open(f"http://localhost:{PORT}")
    ).start()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("Control Center dihentikan.")

    finally:
        stop_tracking()
        stop_face_recognition()
        server.server_close()
