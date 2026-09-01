"""
================================================================================
 LC CHARACTERIZATION  ·  v2 (restructured UI)
 Automation of operation & characterization system for liquid-crystal devices
 Project P-2026-129  ·  Omar Wattad  ·  Supervisor: Prof. Ibrahim Abdulhalim
 Ben-Gurion University of the Negev
================================================================================

"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json, os, threading, time, re, subprocess
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ─── OPTIONAL HARDWARE IMPORTS ───────────────────────────────────────────────
try:
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration, AcquisitionType
    import nidaqmx.system
    NIDAQMX_AVAILABLE = True
except ImportError:
    NIDAQMX_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# DirectShow capture stack (see the CAMERA BACKEND section below).  OpenCV's
# DSHOW backend cannot instantiate the TUCam source filter, so all camera work
# goes through pygrabber + a hand-built DirectShow graph instead.
try:
    import queue
    import ctypes
    from ctypes import POINTER, c_long
    import comtypes                                   # noqa: F401
    from pygrabber.dshow_graph import FilterGraph     # noqa: F401
    DSHOW_BACKEND = True
    _DSHOW_IMPORT_ERROR = ""
except ImportError as _exc:
    DSHOW_BACKEND = False
    _DSHOW_IMPORT_ERROR = str(_exc)

# NOTE: seabreeze (Ocean Optics) is deliberately NOT used -- this rig has an
# AVANTES AvaSpec, which seabreeze cannot enumerate.  The Avantes AvaSpec DLL
# is loaded lazily instead; see the SPECTROMETER BACKEND section below.
import struct
_AVS_LOCK = threading.Lock()

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
CONFIG_FILE  = "daq_config_v2.json"
RESULTS_DIR  = "Results"
SAMPLE_RATE  = 25000            # NI 9264 max update rate (AO)
AI_RATE      = 100000.0         # AI faster than AO, as in the reference rig
AI_TERM_NAME = "DIFF"
AI_VMIN, AI_VMAX = -10.0, 10.0
AO_VMIN, AO_VMAX = -10.0, 10.0
MAX_AO_LINES = 16               # ao0..ao15 hard ceiling
MAX_AI_LINES = 16               # ai0..ai15 hard ceiling

# Defaults only — module names are re-detected at start-up.
AO_DEVICE = "cDAQ3Mod1"         # NI 9264
AI_DEVICE = "cDAQ5Mod1"         # NI 9205

V_MIN, V_MAX       = 0.0, 10.0
FREQ_MIN, FREQ_MAX = 1000.0, 10000.0

# Deep Teal theme
BG_DARK="#001e28"; BG_PANEL="#003747"; BG_CARD="#065465"
ACCENT_CYAN="#4fc3d8"; ACCENT_GREEN="#2ab8cc"; ACCENT_RED="#e03c3c"
ACCENT_ORANGE="#f0a500"; ACCENT_YELLOW="#7de8f5"
TEXT_WHITE="#e0f4f8"; TEXT_GRAY="#5a9aaa"
BTN_RUN="#046276"; BTN_STOP="#e03c3c"; PLOT_BG="#00131c"


# ════════════════════════════════════════════════════════════════════════════
#  LC PHYSICS MODEL (reference curve; not used for measured data)
# ════════════════════════════════════════════════════════════════════════════
class LCModel:
    V_TH=1.2; V50=2.6; P=3.2; T_MAX=1.0; T_MIN=0.05; TAU_OFF=0.012

    @classmethod
    def transmission(cls, v):
        v = max(0.0, v)
        if v <= cls.V_TH:
            return cls.T_MAX
        return cls.T_MIN + (cls.T_MAX - cls.T_MIN) / (1.0 + (v / cls.V50) ** cls.P)


def clamp(x, lo, hi): return max(lo, min(hi, x))


def ai_terminal_config():
    if not NIDAQMX_AVAILABLE:
        return None
    n = (AI_TERM_NAME or "DIFF").upper().strip()
    return {"DEFAULT": TerminalConfiguration.DEFAULT,
            "RSE":     TerminalConfiguration.RSE,
            "NRSE":    TerminalConfiguration.NRSE}.get(n, TerminalConfiguration.DIFF)


# ════════════════════════════════════════════════════════════════════════════
#  HARDWARE DETECTION
# ════════════════════════════════════════════════════════════════════════════
def detect_daq_devices():
    if not NIDAQMX_AVAILABLE:
        return [], []
    real, simulated = [], []
    try:
        for d in nidaqmx.system.System.local().devices:
            try: is_sim = d.is_simulated
            except Exception: is_sim = False
            if is_sim:
                simulated.append(d.name)
            else:
                try: d.self_test_device(); real.append(d.name)
                except Exception: pass
    except Exception:
        pass
    return real, simulated


def resolve_daq_modules():
    """Return (ao_device, ai_device, n_ao, n_ai).  Real hardware preferred over
    simulated.  n_ao / n_ai are the REAL line counts of those modules, capped
    at 16 — this is what limits how many channels the UI lets you add."""
    if not NIDAQMX_AVAILABLE:
        return None, None, MAX_AO_LINES, MAX_AI_LINES
    best = {"ao": (None, 0, True), "ai": (None, 0, True)}   # name, count, sim
    try:
        for d in nidaqmx.system.System.local().devices:
            try:
                n_ao = len(d.ao_physical_chans.channel_names)
                n_ai = len(d.ai_physical_chans.channel_names)
                sim  = bool(getattr(d, "is_simulated", False))
            except Exception:
                continue
            for kind, n in (("ao", n_ao), ("ai", n_ai)):
                if n <= 0:
                    continue
                cur_name, cur_n, cur_sim = best[kind]
                # a real device always beats a simulated one; otherwise first wins
                if cur_name is None or (cur_sim and not sim):
                    best[kind] = (d.name, n, sim)
    except Exception:
        pass
    ao_dev, n_ao, _ = best["ao"]
    ai_dev, n_ai, _ = best["ai"]
    return (ao_dev, ai_dev,
            min(n_ao or MAX_AO_LINES, MAX_AO_LINES),
            min(n_ai or MAX_AI_LINES, MAX_AI_LINES))


# ════════════════════════════════════════════════════════════════════════════
#  CAMERA BACKEND — DirectShow capture for the Tucsen ISH1000
#
#  Why this exists:  cv2.VideoCapture(index, CAP_DSHOW) cannot instantiate the
#  TUCam source filter ("backend is generally available but can't be used to
#  capture by index"), so OpenCV never sees the ISH1000 even though Windows
#  enumerates it fine.  Building a real DirectShow graph with pygrabber works.
#
#  THREAD CONFINEMENT:  DirectShow is COM, and COM objects are apartment-
#  affine.  CameraIntensity owns a private worker thread; the graph is created,
#  run, controlled and torn down entirely inside it, and every public method
#  posts a command to that thread.  This matters here because _worker_camera
#  runs on a capture thread while _apply_cam_settings is called from the
#  tkinter main thread — letting both touch one COM object produces
#  intermittent, hard-to-reproduce COMError crashes.
#
#  MONO SENSOR:  the ISH1000 has no colour filter array.  The filter still
#  delivers RGB24 with all three channels identical, so channel 0 is taken
#  directly rather than running a luma conversion — exact, and 3x less memory
#  traffic on a 10 MP frame.
# ════════════════════════════════════════════════════════════════════════════

TUCSEN_HINTS = ("tucam", "tucsen", "ish")

FLAG_AUTO = 0x0001
FLAG_MANUAL = 0x0002

CAMERA_CONTROL_PROPS = {0: "Pan", 1: "Tilt", 2: "Roll", 3: "Zoom",
                        4: "Exposure", 5: "Iris", 6: "Focus"}
PROC_AMP_PROPS = {0: "Brightness", 1: "Contrast", 2: "Hue", 3: "Saturation",
                  4: "Sharpness", 5: "Gamma", 6: "ColorEnable",
                  7: "WhiteBalance", 8: "BacklightCompensation", 9: "Gain"}

CC_EXPOSURE = 4
VPA_GAIN = 9


# ---------------------------------------------------------------------------
# DirectShow control interfaces (declared by hand; no typelib required)
# ---------------------------------------------------------------------------
def _make_interfaces():
    from comtypes import COMMETHOD, GUID, IUnknown

    # ctypes.HRESULT exists only on Windows; comtypes re-exports it.
    HRESULT = getattr(ctypes, "HRESULT", None)
    if HRESULT is None:
        from comtypes import HRESULT

    methods = [
        COMMETHOD([], HRESULT, "GetRange",
                  (["in"], c_long, "Property"),
                  (["out"], POINTER(c_long), "pMin"),
                  (["out"], POINTER(c_long), "pMax"),
                  (["out"], POINTER(c_long), "pSteppingDelta"),
                  (["out"], POINTER(c_long), "pDefault"),
                  (["out"], POINTER(c_long), "pCapsFlags")),
        COMMETHOD([], HRESULT, "Set",
                  (["in"], c_long, "Property"),
                  (["in"], c_long, "lValue"),
                  (["in"], c_long, "Flags")),
        COMMETHOD([], HRESULT, "Get",
                  (["in"], c_long, "Property"),
                  (["out"], POINTER(c_long), "lValue"),
                  (["out"], POINTER(c_long), "Flags")),
    ]

    class IAMCameraControl(IUnknown):
        _iid_ = GUID("{C6E13370-30AC-11d0-A18C-00A0C9118956}")
        _methods_ = methods

    class IAMVideoProcAmp(IUnknown):
        _iid_ = GUID("{C6E13360-30AC-11d0-A18C-00A0C9118956}")
        _methods_ = methods

    return IAMCameraControl, IAMVideoProcAmp


# ---------------------------------------------------------------------------
def camera_names():
    """DirectShow device names. Initialises COM on the calling thread."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return [], ("pygrabber is not installed. "
                    "Install it with:  pip install pygrabber comtypes")

    uninit = None
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
            uninit = pythoncom.CoUninitialize
        except ImportError:
            from comtypes import CoInitialize, CoUninitialize
            CoInitialize()
            uninit = CoUninitialize
        return FilterGraph().get_input_devices(), ""
    except Exception as exc:
        return [], f"DirectShow name enumeration failed: {exc}"
    finally:
        if uninit is not None:
            try:
                uninit()
            except Exception:
                pass


def find_tucsen_index(names=None):
    if names is None:
        names = camera_names()[0]
    for i, n in enumerate(names):
        if any(h in (n or "").lower() for h in TUCSEN_HINTS):
            return i
    return None


def probe_cameras():
    """[(index, name)] for every device a DirectShow graph can build on.

    Replaces the old cv2-based probe, which rejected the TUCam filter.
    Non-webcam devices sort first so a scientific camera is the default.
    """
    names, err = camera_names()
    if not names:
        return [], err

    from pygrabber.dshow_graph import FilterGraph
    uninit = None
    found = []
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
            uninit = pythoncom.CoUninitialize
        except ImportError:
            from comtypes import CoInitialize, CoUninitialize
            CoInitialize()
            uninit = CoUninitialize

        for i, name in enumerate(names):
            g = None
            try:
                g = FilterGraph()
                g.add_video_input_device(i)   # instantiation is the real test
                found.append((i, name))
            except Exception:
                pass
            finally:
                if g is not None:
                    try:
                        g.stop()
                    except Exception:
                        pass
    finally:
        if uninit is not None:
            try:
                uninit()
            except Exception:
                pass

    webcam_hints = ("integrated", "built-in", "builtin", "hd webcam",
                    "hd camera", "uvc webcam", "realtek", "lenovo",
                    "surface", "facetime", "ir camera")

    def is_webcam(nm):
        return any(h in (nm or "").lower() for h in webcam_hints)

    found.sort(key=lambda t: (is_webcam(t[1]), t[0]))
    return found, ""


# ---------------------------------------------------------------------------
class CameraIntensity:
    """API-compatible with the old cv2 version, plus read() and set_controls().

    Public surface used by GUIGP.py:
        open()            -> self
        read()            -> (ok, frame2d_uint8)
        mean_intensity()  -> float
        set_controls(ms, gain) -> [note strings]
        close()
        resolution        -> (w, h) or None
    """

    def __init__(self, index=0, target_fps=15.0,
                 flip_h=False, flip_v=False, rot_quarters=0):
        self.index = index
        self.target_fps = target_fps
        # Orientation, read per frame in the callback.  Plain attribute
        # writes are atomic under the GIL, so these are live-changeable.
        self.flip_h = bool(flip_h)
        self.flip_v = bool(flip_v)
        self.rot_quarters = int(rot_quarters) % 4
        self.resolution = None
        # Software gain, used when the vendor filter refuses the standard
        # DirectShow gain interface (TUCam does).  1.0 = untouched.
        self.digital_gain = 1.0
        # Self-healing: the graph is rebuilt automatically if grabbing fails
        # or frames stall, so the stream never dies permanently.
        self.reconnects = 0
        self.stall_rebuild_s = 4.0
        self.last_error = ""
        self._pending_controls = None   # re-applied after a rebuild

        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()

        self._cmd_q = queue.Queue()
        self._thread = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None

        # Kept so old code doing `cam._cap is None` still behaves sanely.
        self._cap = None

    # ------------------------------------------------------------- lifecycle

    def open(self, timeout=20.0):
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="TucsenGraph",
                                        daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise RuntimeError(f"camera did not start within {timeout:.0f} s")
        if self._error:
            err = self._error
            self.close()
            raise RuntimeError(err)
        self._cap = self            # truthy sentinel for legacy checks
        return self

    def close(self, timeout=5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            self._thread = None
        self._cap = None

    def is_open(self):
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------ frame access

    def read(self, timeout=2.0):
        """(ok, frame). frame is 2-D uint8. Waits briefly for the first frame."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                f = self._frame
            if f is not None:
                return True, f
            if not self.is_open() or time.monotonic() > deadline:
                return False, None
            time.sleep(0.005)

    def read_new(self, last_id, timeout=2.0):
        """Wait for a frame newer than last_id. Returns (ok, frame, frame_id)."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                f, fid = self._frame, self._frame_id
            if f is not None and fid != last_id:
                return True, f, fid
            if not self.is_open() or time.monotonic() > deadline:
                return False, None, last_id
            time.sleep(0.005)

    def set_orientation(self, flip_h=None, flip_v=None, rot_quarters=None):
        """Change flips/rotation of the live stream. Takes effect on the
        next frame; no restart needed."""
        if flip_h is not None:
            self.flip_h = bool(flip_h)
        if flip_v is not None:
            self.flip_v = bool(flip_v)
        if rot_quarters is not None:
            self.rot_quarters = int(rot_quarters) % 4

    def set_digital_gain(self, gain):
        """Live software gain (used when the filter refuses hardware gain)."""
        self.digital_gain = max(0.1, min(16.0, float(gain)))

    def set_fps(self, fps):
        """Change the capture rate of a running camera. Thread-safe: the
        graph thread re-reads target_fps every loop iteration, and a plain
        float assignment is atomic under the GIL."""
        self.target_fps = max(0.5, float(fps))

    def mean_intensity(self):
        ok, frame = self.read()
        if not ok or frame is None:
            raise RuntimeError("Camera grab failed.")
        return float(np.mean(self._gray_of(frame)))

    def roi_stats(self, roi=None):
        """Stats over roi=(x,y,w,h), or the whole frame if roi is None.

        `clipped` is the fraction of pixels at 255. Any non-zero value
        invalidates a transmission measurement from this ROI -- unlike the
        old all-or-nothing saturation test, this catches partial clipping,
        which is the case that silently distorts a T(V) curve.
        """
        ok, frame = self.read()
        if not ok or frame is None:
            return None
        if roi:
            x, y, w, h = roi
            H, W = frame.shape[:2]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(W, int(x + w)), min(H, int(y + h))
            if x1 <= x0 or y1 <= y0:
                return None
            frame = frame[y0:y1, x0:x1]
        g = self._gray_of(frame)
        clip = (frame >= 255)
        if clip.ndim == 3:
            clip = clip.any(axis=-1)
        return {"mean": float(g.mean()), "std": float(g.std()),
                "min": float(g.min()), "max": float(g.max()),
                "clipped": float(clip.mean()),
                "n_pixels": int(g.size)}

    # --------------------------------------------------------------- controls

    def set_controls(self, exposure_ms, gain, lock_auto=True):
        """Apply exposure/gain on the graph thread. Returns note strings."""
        if not self.is_open():
            return ["camera not running"]
        self._pending_controls = (exposure_ms, gain, lock_auto)
        reply = queue.Queue(maxsize=1)
        self._cmd_q.put(("controls", (exposure_ms, gain, lock_auto), reply))
        try:
            return reply.get(timeout=10.0)
        except queue.Empty:
            return ["control request timed out"]

    def inventory(self):
        """List every control with value and auto/manual flag."""
        if not self.is_open():
            return ["camera not running"]
        reply = queue.Queue(maxsize=1)
        self._cmd_q.put(("inventory", None, reply))
        try:
            return reply.get(timeout=10.0)
        except queue.Empty:
            return ["inventory request timed out"]

    def show_properties(self):
        """Open the filter's native property page (vendor dialog)."""
        if not self.is_open():
            return ["camera not running"]
        reply = queue.Queue(maxsize=1)
        self._cmd_q.put(("props", None, reply))
        try:
            return reply.get(timeout=60.0)
        except queue.Empty:
            return ["property page timed out"]

    # ----------------------------------------------------------- graph thread

    def _build_graph(self):
        """Create and start a fresh DirectShow graph."""
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        graph.add_video_input_device(self.index)
        try:
            self.resolution = graph.get_input_device().get_current_format()
        except Exception:
            pass
        graph.add_sample_grabber(self._on_frame)
        graph.add_null_render()
        graph.prepare_preview_graph()
        graph.run()
        # Re-apply exposure/gain after a rebuild, otherwise a reconnect
        # silently reverts the camera to its power-on defaults.
        if self._pending_controls is not None:
            try:
                self._handle_cmd(graph, "controls", self._pending_controls)
            except Exception:
                pass
        return graph

    def _stream(self, graph):
        """Grab frames until stop is requested or a rebuild is needed.

        Returns True to keep the graph, False to ask for a rebuild.
        A single grab_frame() failure used to `break` and kill the capture
        thread forever -- that is why the camera could only be revived with
        Stop/Run.  Failures are now tolerated and escalated instead.
        """
        next_grab = 0.0
        fails = 0
        seen_id = self._frame_id
        last_progress = time.monotonic()

        while not self._stop.is_set():
            # service pending control commands first
            try:
                while True:
                    kind, payload, reply = self._cmd_q.get_nowait()
                    try:
                        reply.put(self._handle_cmd(graph, kind, payload))
                    except Exception as exc:
                        reply.put([f"failed: {exc!r}"])
            except queue.Empty:
                pass

            now = time.monotonic()
            if self._frame_id != seen_id:          # real progress
                seen_id = self._frame_id
                last_progress = now
                fails = 0

            # Read target_fps fresh so the spinbox applies live.
            period = 1.0 / max(0.5, self.target_fps)
            if now >= next_grab:
                next_grab = now + period
                try:
                    graph.grab_frame()
                except Exception as exc:
                    fails += 1
                    self.last_error = f"grab_frame: {exc!r}"
                    if fails >= 25:
                        return False               # ask for a rebuild
            if now - last_progress > self.stall_rebuild_s:
                self.last_error = (f"no frame for {self.stall_rebuild_s:.0f}s")
                return False                       # ask for a rebuild
            self._stop.wait(0.002)
        return True

    def _run(self):
        """Supervisor: keeps a live graph until stop() is called.

        The camera is only ever given up when the user stops it; any other
        failure results in the graph being torn down and rebuilt.
        """
        co_init = False
        try:
            import comtypes
            try:
                comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
                co_init = True
            except Exception:
                try:
                    comtypes.CoInitialize()
                    co_init = True
                except Exception:
                    pass

            first = True
            while not self._stop.is_set():
                graph = None
                try:
                    graph = self._build_graph()
                    self._ready.set()
                    first = False
                    if self._stream(graph):
                        break                      # clean stop
                except Exception as exc:
                    self.last_error = repr(exc)
                    if first:
                        # Could not even open once: report through open().
                        self._error = repr(exc)
                        self._ready.set()
                        return
                finally:
                    if graph is not None:
                        try:
                            graph.stop()
                        except Exception:
                            pass
                if self._stop.is_set():
                    break
                self.reconnects += 1
                self._stop.wait(0.8)               # settle before retrying
        finally:
            if co_init:
                try:
                    import comtypes
                    comtypes.CoUninitialize()
                except Exception:
                    pass

    def _handle_cmd(self, graph, kind, payload):
        if kind == "props":
            dev = graph.get_input_device()
            for meth in ("show_properties", "show_property_page",
                         "show_format_dialog"):
                fn = getattr(dev, meth, None)
                if fn:
                    fn()
                    return [f"opened {meth}()"]
            return ["this filter exposes no property page"]

        filt = self._com_filter(graph)
        if filt is None:
            return ["could not reach the COM filter for control access"]

        IAMCameraControl, IAMVideoProcAmp = _make_interfaces()

        if kind == "inventory":
            out = []
            for iface, props, label in (
                    (IAMCameraControl, CAMERA_CONTROL_PROPS, "CameraControl"),
                    (IAMVideoProcAmp, PROC_AMP_PROPS, "VideoProcAmp")):
                try:
                    ctrl = filt.QueryInterface(iface)
                except Exception:
                    out.append(f"{label}: not supported")
                    continue
                for pid, pname in sorted(props.items()):
                    try:
                        val, flags = ctrl.Get(pid)
                    except Exception:
                        continue
                    mode = "AUTO" if flags & FLAG_AUTO else "manual"
                    out.append(f"{label}.{pname} = {val} [{mode}]")
            return out or ["no controls exposed"]

        if kind == "controls":
            exposure_ms, gain, lock_auto = payload
            notes = []

            # Exposure via IAMCameraControl.
            try:
                cc = filt.QueryInterface(IAMCameraControl)
                try:
                    mn, mx, step, default, caps = cc.GetRange(CC_EXPOSURE)
                except Exception:
                    mn = mx = None
                if exposure_ms and exposure_ms > 0:
                    # DirectShow convention is log2(seconds); vendor filters
                    # sometimes use raw units. Try log2 first, verify readback,
                    # fall back to a range-scaled raw value.
                    want = int(round(np.log2(max(exposure_ms, 1e-3) / 1000.0)))
                    ok = False
                    try:
                        cc.Set(CC_EXPOSURE, want, FLAG_MANUAL)
                        got, _ = cc.Get(CC_EXPOSURE)
                        ok = abs(got - want) <= 1
                    except Exception:
                        ok = False
                    if not ok and mn is not None:
                        try:
                            raw = int(np.clip(exposure_ms, mn, mx))
                            cc.Set(CC_EXPOSURE, raw, FLAG_MANUAL)
                            got, _ = cc.Get(CC_EXPOSURE)
                            ok = abs(got - raw) <= max(1, step or 1)
                        except Exception:
                            ok = False
                    got, flags = cc.Get(CC_EXPOSURE)
                    notes.append(
                        f"exposure {'OK' if ok else 'NOT accepted'} "
                        f"(readback {got}, "
                        f"{'AUTO' if flags & FLAG_AUTO else 'manual'}"
                        f"{f', range {mn}..{mx}' if mn is not None else ''})")
                if lock_auto:
                    for pid in (CC_EXPOSURE,):
                        try:
                            v, f = cc.Get(pid)
                            if f & FLAG_AUTO:
                                cc.Set(pid, v, FLAG_MANUAL)
                                notes.append(
                                    f"{CAMERA_CONTROL_PROPS[pid]} auto -> manual")
                        except Exception:
                            pass
            except Exception:
                notes.append("IAMCameraControl not supported by this filter")

            # Gain and the other auto controls via IAMVideoProcAmp.
            try:
                vp = filt.QueryInterface(IAMVideoProcAmp)
                if gain is not None:
                    hw_ok = False
                    try:
                        mn, mx, step, default, caps = vp.GetRange(VPA_GAIN)
                        raw = int(np.clip(gain, mn, mx))
                        vp.Set(VPA_GAIN, raw, FLAG_MANUAL)
                        got, flags = vp.Get(VPA_GAIN)
                        # "Accepted" only counts if the value actually stuck.
                        hw_ok = abs(int(got) - raw) <= max(1, int(step or 1))
                        notes.append(
                            f"hardware gain {got} (range {mn}..{mx})"
                            if hw_ok else
                            f"hardware gain REFUSED (asked {raw}, got {got})")
                    except Exception:
                        notes.append("hardware gain not supported")
                    if hw_ok:
                        self.digital_gain = 1.0
                    else:
                        # Fall back to software gain so the control does
                        # something.  This multiplies the captured pixels; it
                        # amplifies noise with the signal and does NOT add
                        # real sensitivity, so keep it at 1 for quantitative
                        # work and use "Driver settings..." for true gain.
                        self.digital_gain = max(0.1, min(16.0, float(gain)))
                        notes.append(
                            f"using DIGITAL gain x{self.digital_gain:g}")
                if lock_auto:
                    locked = []
                    for pid in sorted(PROC_AMP_PROPS):
                        try:
                            v, f = vp.Get(pid)
                            if f & FLAG_AUTO:
                                vp.Set(pid, v, FLAG_MANUAL)
                                v2, f2 = vp.Get(pid)
                                if not (f2 & FLAG_AUTO):
                                    locked.append(PROC_AMP_PROPS[pid])
                        except Exception:
                            pass
                    if locked:
                        notes.append("locked to manual: " + ", ".join(locked))
            except Exception:
                notes.append("IAMVideoProcAmp not supported by this filter")

            return notes or ["no controls exposed by this filter"]

        return [f"unknown command {kind!r}"]

    @staticmethod
    def _com_filter(graph):
        """Dig the COM source filter out of pygrabber's wrapper."""
        IAMCameraControl, IAMVideoProcAmp = _make_interfaces()
        cands = []
        try:
            dev = graph.get_input_device()
            cands.append(dev)
            for a in ("filter", "instance", "_filter", "source", "obj"):
                cands.append(getattr(dev, a, None))
        except Exception:
            pass
        for a in ("video_input_filter", "source_filter", "filter"):
            cands.append(getattr(graph, a, None))

        expanded = []
        for c in cands:
            if c is None:
                continue
            expanded.append(c)
            inner = getattr(c, "instance", None)
            if inner is not None:
                expanded.append(inner)

        for obj in expanded:
            for iface in (IAMCameraControl, IAMVideoProcAmp):
                try:
                    obj.QueryInterface(iface)
                    return obj
                except Exception:
                    continue
        return None

    # --------------------------------------------------------------- callback

    @staticmethod
    def _gray_of(frame):
        """Luma of an RGB frame (float32). Pass-through for 2-D frames."""
        if frame.ndim == 2:
            return frame.astype(np.float32)
        f = frame.astype(np.float32)
        return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]

    def _on_frame(self, image):
        try:
            arr = image
            if arr.ndim == 3:
                arr = arr[..., ::-1]              # DirectShow BGR -> RGB
            # User-controlled orientation.  No unconditional vertical flip:
            # this filter already delivers frames top-down, so flipping by
            # default is what made the image appear upside-down.
            if self.flip_v:
                arr = arr[::-1]
            if self.flip_h:
                arr = arr[:, ::-1]
            k = self.rot_quarters % 4
            if k:
                arr = np.rot90(arr, k)
            g = self.digital_gain
            if g and abs(g - 1.0) > 1e-3:
                arr = np.clip(arr.astype(np.float32) * g, 0, 255).astype(np.uint8)
            arr = np.ascontiguousarray(arr)
        except Exception:
            return
        with self._lock:
            self._frame = arr
            self._frame_id += 1


WEBCAM_HINTS=("integrated","built-in","builtin","hd webcam","hd camera",
              "uvc webcam","realtek","lenovo","surface","facetime","ir camera")
ISH1000_NAME_HINTS=("ish1000","ish 1000","tucsen","h series")
# Tucsen's discontinued-camera table lists both of these USB identities for
# ISH1000 units.  Detecting one proves that Windows can see the physical camera;
# it does NOT by itself prove that OpenCV/DirectShow can acquire frames from it.
ISH1000_USB_IDS=("VID_5453&PID_A803","VID_0547&PID_A003")


# ════════════════════════════════════════════════════════════════════════════
#  SPECTROMETER BACKEND — Avantes AvaSpec (AS5216 / AvaSpec DLL)
#
#  IMPORTANT:  this hardware is AVANTES, not Ocean Optics.  The previous
#  implementation used `seabreeze`, which drives Ocean Optics / Ocean Insight
#  devices only and can never enumerate an AvaSpec — it silently fell back to
#  a synthetic spectrum, which is why the tab appeared to "work" while showing
#  data that never came from the instrument.  Simulation is now removed
#  entirely: if the spectrometer is absent, you get an error, not fake data.
#
#  The DLL is plain C (not COM), so calls are serialised with a lock rather
#  than confined to one thread.
# ════════════════════════════════════════════════════════════════════════════

AVS_SERIAL_LEN   = 10
USER_ID_LEN      = 64
INVALID_AVS_HANDLE = 1000
MAX_NR_PIXELS    = 4096

AVS_ERRORS = {
    0:   "success",
   -1:   "invalid parameter",
   -2:   "invalid parameter",
   -3:   "operation not supported",
   -4:   "device not found",
   -5:   "invalid device ID",
   -6:   "operation pending",
   -7:   "timeout",
   -8:   "invalid password",
   -9:   "invalid measurement data",
   -10:  "invalid size",
   -11:  "invalid pixel range",
   -12:  "invalid integration time",
   -13:  "invalid combination of settings",
   -15:  "no measurement buffer available",
   -16:  "unknown error",
   -17:  "communication error",
   -18:  "no spectra in RAM",
   -19:  "invalid DLL version",
   -20:  "out of memory",
   -21:  "DLL initialisation failed",
   -22:  "invalid state",
   -23:  "invalid password",
   -100: "device not configured",
}


def _avs_msg(code):
    return AVS_ERRORS.get(int(code), f"error code {code}")


class SpectrometerError(RuntimeError):
    pass


class _AvsIdentity(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("m_SerialNumber",     ctypes.c_char * AVS_SERIAL_LEN),
                ("m_UserFriendlyName", ctypes.c_char * USER_ID_LEN),
                ("m_Status",           ctypes.c_ubyte)]


class _DarkCorrection(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("m_Enable", ctypes.c_ubyte),
                ("m_ForgetPercentage", ctypes.c_ubyte)]


class _Smoothing(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("m_SmoothPix", ctypes.c_ushort),
                ("m_SmoothModel", ctypes.c_ubyte)]


class _Trigger(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("m_Mode", ctypes.c_ubyte),
                ("m_Source", ctypes.c_ubyte),
                ("m_SourceType", ctypes.c_ubyte)]


class _ControlSettings(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("m_StrobeControl", ctypes.c_ushort),
                ("m_LaserDelay", ctypes.c_uint),
                ("m_LaserWidth", ctypes.c_uint),
                ("m_LaserWaveLength", ctypes.c_float),
                ("m_StoreToRam", ctypes.c_ushort)]


class _MeasConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("m_StartPixel", ctypes.c_ushort),
                ("m_StopPixel", ctypes.c_ushort),
                ("m_IntegrationTime", ctypes.c_float),
                ("m_IntegrationDelay", ctypes.c_uint),
                ("m_NrAverages", ctypes.c_uint),
                ("m_CorDynDark", _DarkCorrection),
                ("m_Smoothing", _Smoothing),
                ("m_SaturationDetection", ctypes.c_ubyte),
                ("m_Trigger", _Trigger),
                ("m_Control", _ControlSettings)]


_AVS_DLL = None
_AVS_DLL_PATH = ""
_AVS_DLL_ERROR = ""


def _avs_dll_candidates():
    """Every plausible location for the AvaSpec DLL, 64-bit names first."""
    import glob as _glob
    names64 = ("avaspecx64.dll", "AS5216x64.dll", "avaspec64.dll")
    names32 = ("avaspec.dll", "AS5216.dll")
    names = names64 + names32 if struct.calcsize("P") == 8 else names32 + names64

    roots = [r"C:\Windows\System32", r"C:\Windows\SysWOW64",
             r"C:\Program Files\Avantes", r"C:\Program Files (x86)\Avantes",
             r"C:\AvaSpecx64-DLL", r"C:\AvaSpec-DLL",
             os.getcwd()]
    out = []
    for n in names:
        for r in roots:
            p = os.path.join(r, n)
            if os.path.isfile(p):
                out.append(p)
    # AvaSoft / SDK installs live under versioned folders
    for pat in (r"C:\Program Files*\Avantes\**\*.dll",
                r"C:\AvaSpec*\**\*.dll",
                r"C:\Program Files*\AvaSoft*\**\*.dll"):
        try:
            for p in _glob.glob(pat, recursive=True):
                if os.path.basename(p).lower() in {n.lower() for n in names}:
                    out.append(p)
        except Exception:
            pass
    seen, uniq = set(), []
    for p in out:
        k = p.lower()
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq


def _pe_bitness(path):
    try:
        with open(path, "rb") as fh:
            if fh.read(2) != b"MZ":
                return "?"
            fh.seek(0x3C)
            off = struct.unpack("<I", fh.read(4))[0]
            fh.seek(off)
            if fh.read(4) != b"PE\0\0":
                return "?"
            machine = struct.unpack("<H", fh.read(2))[0]
        return {0x014C: "32-bit", 0x8664: "64-bit"}.get(machine, "?")
    except Exception:
        return "?"


def load_avs_dll():
    """Load the AvaSpec DLL once and bind the functions we use."""
    global _AVS_DLL, _AVS_DLL_PATH, _AVS_DLL_ERROR
    if _AVS_DLL is not None or _AVS_DLL_ERROR:
        return _AVS_DLL

    cands = _avs_dll_candidates()
    if not cands:
        _AVS_DLL_ERROR = (
            "The Avantes AvaSpec DLL was not found.\n\n"
            "Install the Avantes SDK / AvaSpec-DLL package (the same one\n"
            "AvaSoft uses), or place avaspecx64.dll next to this script.\n"
            "Searched System32, Program Files\\Avantes and C:\\AvaSpec*.")
        return None

    py_bits = struct.calcsize("P") * 8
    problems = []
    for path in cands:
        bits = _pe_bitness(path)
        if (py_bits == 64 and bits == "32-bit") or \
           (py_bits == 32 and bits == "64-bit"):
            problems.append(f"{path} is {bits}, Python is {py_bits}-bit")
            continue
        try:
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(os.path.dirname(path))
                except Exception:
                    pass
            dll = ctypes.WinDLL(path)
        except OSError as exc:
            problems.append(f"{path}: {exc}")
            continue

        try:
            dll.AVS_Init.argtypes = [ctypes.c_short]
            dll.AVS_Init.restype = ctypes.c_int
            dll.AVS_Done.argtypes = []
            dll.AVS_Done.restype = ctypes.c_int
            dll.AVS_GetNrOfDevices.argtypes = []
            dll.AVS_GetNrOfDevices.restype = ctypes.c_int
            dll.AVS_UpdateUSBDevices.argtypes = []
            dll.AVS_UpdateUSBDevices.restype = ctypes.c_int
            dll.AVS_GetList.argtypes = [ctypes.c_uint,
                                        ctypes.POINTER(ctypes.c_uint),
                                        ctypes.POINTER(_AvsIdentity)]
            dll.AVS_GetList.restype = ctypes.c_int
            dll.AVS_Activate.argtypes = [ctypes.POINTER(_AvsIdentity)]
            dll.AVS_Activate.restype = ctypes.c_int
            dll.AVS_Deactivate.argtypes = [ctypes.c_int]
            dll.AVS_Deactivate.restype = ctypes.c_int
            dll.AVS_GetNumPixels.argtypes = [ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_ushort)]
            dll.AVS_GetNumPixels.restype = ctypes.c_int
            dll.AVS_GetLambda.argtypes = [ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_double)]
            dll.AVS_GetLambda.restype = ctypes.c_int
            dll.AVS_PrepareMeasure.argtypes = [ctypes.c_int,
                                               ctypes.POINTER(_MeasConfig)]
            dll.AVS_PrepareMeasure.restype = ctypes.c_int
            dll.AVS_Measure.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                        ctypes.c_short]
            dll.AVS_Measure.restype = ctypes.c_int
            dll.AVS_PollScan.argtypes = [ctypes.c_int]
            dll.AVS_PollScan.restype = ctypes.c_int
            dll.AVS_StopMeasure.argtypes = [ctypes.c_int]
            dll.AVS_StopMeasure.restype = ctypes.c_int
            dll.AVS_GetScopeData.argtypes = [ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_uint),
                                             ctypes.POINTER(ctypes.c_double)]
            dll.AVS_GetScopeData.restype = ctypes.c_int
            try:
                dll.AVS_UseHighResAdc.argtypes = [ctypes.c_int, ctypes.c_bool]
                dll.AVS_UseHighResAdc.restype = ctypes.c_int
            except Exception:
                pass
            try:
                dll.AVS_GetVersionInfo.argtypes = [
                    ctypes.c_int, ctypes.c_char_p,
                    ctypes.c_char_p, ctypes.c_char_p]
                dll.AVS_GetVersionInfo.restype = ctypes.c_int
            except Exception:
                pass
        except AttributeError as exc:
            problems.append(f"{path}: missing export {exc}")
            continue

        _AVS_DLL = dll
        _AVS_DLL_PATH = path
        return dll

    _AVS_DLL_ERROR = ("The Avantes DLL could not be loaded:\n  " +
                      "\n  ".join(problems[:6]))
    return None


def detect_spectrometers():
    """Return (list_of_dicts, error_message).

    Each dict: {'serial', 'name', 'status'}.  Called before anything else so
    the GUI can state plainly whether the instrument is present.
    """
    if os.name != "nt":
        return [], "The Avantes DLL is available on Windows only."

    dll = load_avs_dll()
    if dll is None:
        return [], _AVS_DLL_ERROR

    with _AVS_LOCK:
        try:
            n = dll.AVS_Init(0)                       # 0 = USB
        except Exception as exc:
            return [], f"AVS_Init raised: {exc}"
        if n < 0:
            return [], (f"AVS_Init failed: {_avs_msg(n)}.\n"
                        "Close AvaSoft — it holds the spectrometer "
                        "exclusively — then rescan.")
        try:
            dll.AVS_UpdateUSBDevices()
        except Exception:
            pass
        n = dll.AVS_GetNrOfDevices()
        if n <= 0:
            return [], ("The DLL loaded but reports no spectrometer.\n"
                        "Check the USB cable, and make sure AvaSoft is "
                        "closed — it claims the device exclusively.")

        req = ctypes.c_uint(0)
        size = ctypes.c_uint(n * ctypes.sizeof(_AvsIdentity))
        arr = (_AvsIdentity * n)()
        rc = dll.AVS_GetList(size, ctypes.byref(req), arr)
        if rc < 0:
            return [], f"AVS_GetList failed: {_avs_msg(rc)}"

        out = []
        for i in range(min(rc, n)):
            try:
                serial = arr[i].m_SerialNumber.decode(errors="replace").strip()
                name = arr[i].m_UserFriendlyName.decode(errors="replace").strip()
            except Exception:
                serial, name = f"dev{i}", ""
            out.append({"serial": serial, "name": name or serial,
                        "status": int(arr[i].m_Status), "_id": arr[i]})
        return out, ""


class Spectrometer:
    """Avantes AvaSpec, polling mode.

    Workflow:
        sp = Spectrometer().open()
        sp.configure(integration_ms=10, averages=10)
        sp.start_continuous()
        y = sp.read_frame(timeout=5)     # repeat
        sp.stop()
        sp.close()

    single() does prepare -> measure(1) -> poll -> read, for Dark/Reference.
    """

    def __init__(self):
        self._dll = None
        self._handle = None
        self._ident = None
        self._npix = 0
        self._wl = None
        self._streaming = False
        self._int_ms = None
        self._avg = None
        self._label = "not connected"

    # ------------------------------------------------------------- lifecycle
    def open(self, serial=None):
        devs, err = detect_spectrometers()
        if not devs:
            raise SpectrometerError(err or "No spectrometer detected.")
        chosen = devs[0]
        if serial:
            for d in devs:
                if d["serial"] == serial:
                    chosen = d
                    break
        self._dll = load_avs_dll()
        self._ident = chosen["_id"]

        with _AVS_LOCK:
            h = self._dll.AVS_Activate(ctypes.byref(self._ident))
            if h == INVALID_AVS_HANDLE or h < 0:
                raise SpectrometerError(
                    f"AVS_Activate failed ({_avs_msg(h)}).\n"
                    "AvaSoft may still hold the device — close it and retry.")
            self._handle = int(h)

            npix = ctypes.c_ushort(0)
            rc = self._dll.AVS_GetNumPixels(self._handle, ctypes.byref(npix))
            if rc < 0:
                raise SpectrometerError(f"AVS_GetNumPixels: {_avs_msg(rc)}")
            self._npix = int(npix.value)

            buf = (ctypes.c_double * max(self._npix, MAX_NR_PIXELS))()
            rc = self._dll.AVS_GetLambda(self._handle, buf)
            if rc < 0:
                raise SpectrometerError(f"AVS_GetLambda: {_avs_msg(rc)}")
            self._wl = np.array(buf[:self._npix], dtype=float)

            # 16-bit ADC: full scale 65535 instead of the 14-bit 16383 the
            # device defaults to.  This is a real readout-mode change, not a
            # rescale -- the extra counts are genuine ADC resolution.
            self.max_counts = 16383
            try:
                rc = self._dll.AVS_UseHighResAdc(self._handle, True)
                if rc >= 0:
                    self.max_counts = 66000
            except Exception:
                pass

        self._label = (f"{chosen['name']} (SN {chosen['serial']}, "
                       f"{self._npix} px, "
                       f"{self._wl[0]:.0f}–{self._wl[-1]:.0f} nm, "
                       f"{'16-bit' if self.max_counts==65535 else '14-bit'} "
                       f"ADC)")
        return self

    def label(self):
        return self._label

    def wavelengths(self):
        return self._wl

    def is_open(self):
        return self._handle is not None

    def close(self):
        if self._handle is None:
            return
        with _AVS_LOCK:
            try:
                self._dll.AVS_StopMeasure(self._handle)
            except Exception:
                pass
            try:
                self._dll.AVS_Deactivate(self._handle)
            except Exception:
                pass
        self._handle = None
        self._streaming = False

    # ------------------------------------------------------------ acquisition
    def _make_config(self, integration_ms, averages):
        cfg = _MeasConfig()
        cfg.m_StartPixel = 0
        cfg.m_StopPixel = max(0, self._npix - 1)
        cfg.m_IntegrationTime = float(max(0.002, integration_ms))
        cfg.m_IntegrationDelay = 0
        cfg.m_NrAverages = int(max(1, averages))
        cfg.m_CorDynDark.m_Enable = 0
        cfg.m_CorDynDark.m_ForgetPercentage = 100
        cfg.m_Smoothing.m_SmoothPix = 0
        cfg.m_Smoothing.m_SmoothModel = 0
        cfg.m_SaturationDetection = 1
        cfg.m_Trigger.m_Mode = 0          # 0 = software / free running
        cfg.m_Trigger.m_Source = 0
        cfg.m_Trigger.m_SourceType = 0
        cfg.m_Control.m_StrobeControl = 0
        cfg.m_Control.m_LaserDelay = 0
        cfg.m_Control.m_LaserWidth = 0
        cfg.m_Control.m_LaserWaveLength = 0.0
        cfg.m_Control.m_StoreToRam = 0
        return cfg

    def configure(self, integration_ms, averages):
        if self._handle is None:
            raise SpectrometerError("Spectrometer is not open.")
        cfg = self._make_config(integration_ms, averages)
        with _AVS_LOCK:
            if self._streaming:
                try:
                    self._dll.AVS_StopMeasure(self._handle)
                except Exception:
                    pass
                self._streaming = False
            rc = self._dll.AVS_PrepareMeasure(self._handle, ctypes.byref(cfg))
            if rc < 0:
                raise SpectrometerError(
                    f"AVS_PrepareMeasure: {_avs_msg(rc)} "
                    f"(integration {integration_ms} ms, averages {averages})")
        self._int_ms, self._avg = float(integration_ms), int(averages)
        return self

    def start_continuous(self):
        """AVS_Measure(-1): the device streams until StopMeasure."""
        if self._handle is None:
            raise SpectrometerError("Spectrometer is not open.")
        with _AVS_LOCK:
            rc = self._dll.AVS_Measure(self._handle, None, -1)
            if rc < 0:
                raise SpectrometerError(f"AVS_Measure: {_avs_msg(rc)}")
            self._streaming = True
        return self

    def stop(self):
        if self._handle is None:
            return
        with _AVS_LOCK:
            try:
                self._dll.AVS_StopMeasure(self._handle)
            except Exception:
                pass
            self._streaming = False

    def _poll_read(self, timeout):
        """Wait for a scan and return it. Assumes a measurement is running."""
        deadline = time.monotonic() + timeout
        tl = ctypes.c_uint(0)
        buf = (ctypes.c_double * self._npix)()
        while time.monotonic() < deadline:
            with _AVS_LOCK:
                ready = self._dll.AVS_PollScan(self._handle)
                if ready == 1:
                    rc = self._dll.AVS_GetScopeData(
                        self._handle, ctypes.byref(tl), buf)
                    if rc < 0:
                        raise SpectrometerError(
                            f"AVS_GetScopeData: {_avs_msg(rc)}")
                    return np.array(buf[:self._npix], dtype=float)
                if ready < 0:
                    raise SpectrometerError(f"AVS_PollScan: {_avs_msg(ready)}")
            time.sleep(0.002)
        raise SpectrometerError(
            f"No scan within {timeout:.1f} s. Integration time may exceed the "
            "timeout, or the device stopped responding.")

    def read_frame(self, timeout=10.0):
        if not self._streaming:
            raise SpectrometerError("No continuous measurement is running.")
        return self._poll_read(timeout)

    def single(self, integration_ms, averages, timeout=None):
        """One spectrum, used for Dark and Reference captures."""
        self.configure(integration_ms, averages)
        if timeout is None:
            timeout = max(10.0, 3.0 * integration_ms * averages / 1000.0 + 5.0)
        with _AVS_LOCK:
            rc = self._dll.AVS_Measure(self._handle, None, 1)
            if rc < 0:
                raise SpectrometerError(f"AVS_Measure: {_avs_msg(rc)}")
            self._streaming = True
        try:
            return self._poll_read(timeout)
        finally:
            self.stop()


def _camera_names_info():
    """Return (DirectShow names, error).

    An empty name list must not be interpreted as "the ISH1000 is absent":
    pygrabber may simply be missing even while OpenCV can open numeric camera
    indexes.  Keeping the error separate prevents the GUI from making that
    misleading claim.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return [], ("DirectShow names are unavailable because pygrabber is not "
                    "installed. Install it with: pip install pygrabber comtypes")

    # pygrabber uses Windows COM.  Tkinter callbacks and camera rescans may run
    # on a thread for which COM has never been initialized, producing
    # 0x800401F0 / -2147221008 ("CoInitialize has not been called").
    uninitialize=None
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
            uninitialize=pythoncom.CoUninitialize
        except ImportError:
            from comtypes import CoInitialize, CoUninitialize
            CoInitialize()
            uninitialize=CoUninitialize
        return FilterGraph().get_input_devices(), ""
    except Exception as exc:
        return [], f"DirectShow name enumeration failed: {exc}"
    finally:
        if uninitialize is not None:
            try:
                uninitialize()
            except Exception:
                pass


def _camera_names():
    return _camera_names_info()[0]


def _is_ish1000_name(name):
    low=(name or "").lower()
    return any(h in low for h in ISH1000_NAME_HINTS)


def _ish1000_usb_devices():
    """Return (matching Windows PnP devices, diagnostic error).

    This read-only query recognizes the physical ISH1000 from its VID/PID even
    when the legacy camera driver does not publish a DirectShow source.
    """
    if os.name!="nt":
        return [], "USB-ID recognition is available on Windows only."
    script=(
        "$d=Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | "
        "Where-Object { $_.InstanceId -match "
        "'VID_5453&PID_A803|VID_0547&PID_A003' } | "
        "Select-Object Status,FriendlyName,InstanceId; "
        "if ($d) { $d | ConvertTo-Json -Compress }"
    )
    try:
        flags=getattr(subprocess,"CREATE_NO_WINDOW",0)
        result=subprocess.run(
            ["powershell.exe","-NoProfile","-NonInteractive","-Command",script],
            capture_output=True,text=True,timeout=6,creationflags=flags)
        if result.returncode!=0:
            return [], (result.stderr.strip() or
                        f"Windows camera query failed ({result.returncode}).")
        raw=result.stdout.strip()
        if not raw:
            return [], ""
        parsed=json.loads(raw)
        if isinstance(parsed,dict):
            parsed=[parsed]
        return parsed, ""
    except FileNotFoundError:
        return [], "PowerShell was not found, so the USB ID could not be checked."
    except Exception as exc:
        return [], f"USB-ID recognition failed: {exc}"


def detect_cameras_named():
    """[(index, name)] for every camera a DirectShow GRAPH can be built on.

    The old implementation validated cameras with cv2.VideoCapture(i).read().
    OpenCV's DSHOW backend cannot instantiate the TUCam source filter, so the
    ISH1000 always failed that test and never reached the GUI's camera list --
    which is why the camera appeared 'not recognised' despite Windows seeing
    it.  Building a real graph is both the correct test and the way frames are
    actually acquired now.
    """
    if DSHOW_BACKEND:
        found, err = probe_cameras()
        if err:
            print(f"[camera] {err}")
        return found
    return _legacy_detect_cameras_named()


def _legacy_detect_cameras_named():
    if not OPENCV_AVAILABLE:
        return []
    os.environ.setdefault("OPENCV_LOG_LEVEL","SILENT")
    try: cv2.setLogLevel(0)
    except Exception: pass
    names=_camera_names()
    found=[]
    # Scan enough indexes for every named DirectShow source, while retaining a
    # small fallback range for systems where name enumeration is unavailable.
    for i in range(max(6,len(names)+2)):
        cap=None
        try:
            backend=cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_ANY
            cap=cv2.VideoCapture(i,backend)
            if not cap.isOpened(): continue
            ret,frame=cap.read()
            if ret and frame is not None:
                nm=names[i] if i<len(names) else f"Camera {i}"
                found.append((i,nm))
        except Exception:
            pass
        finally:
            if cap is not None:
                try: cap.release()
                except Exception: pass
    def is_webcam(nm): return any(h in nm.lower() for h in WEBCAM_HINTS)
    found.sort(key=lambda in_: (is_webcam(in_[1]), in_[0]))
    return found


def detect_cameras():
    return [i for i,_ in detect_cameras_named()]


# ════════════════════════════════════════════════════════════════════════════
#  DEVICE SELECT DIALOG (trimmed)
# ════════════════════════════════════════════════════════════════════════════
class DeviceSelectDialog(tk.Toplevel):
    """Startup chooser: DAQ, CMOS camera, or spectrometer.

    Each button reports whether that instrument is actually present, and the
    chosen device decides which tab the application opens on.  If nothing is
    connected the program can still be entered normally (Optical Response
    tab); missing hardware then produces honest errors at Run time instead
    of synthetic data.
    """

    TAB_FOR_DEVICE = {"DAQ": 0, "CMOS Camera": 1, "Spectrometer": 2}

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Select Device"); self.geometry("560x470")
        self.configure(bg=BG_DARK); self.resizable(False, False)
        self.grab_set()
        self.result = None
        x = (self.winfo_screenwidth() - 560) // 2
        y = (self.winfo_screenheight() - 470) // 2
        self.geometry(f"560x470+{x}+{y}")
        self._build()

    def _build(self):
        """(Re)build the dialog contents INSIDE this window.

        Rescan destroys the children and calls this again.  It must never
        call __init__: tk.Toplevel.__init__ creates a brand-new window, which
        is exactly the "new dialog appears, old one goes blank" bug.
        """
        for w in self.winfo_children():
            w.destroy()

        tk.Label(self, text="LC Characterization · Select Device",
                 font=("Consolas", 13, "bold"), fg=ACCENT_CYAN,
                 bg=BG_DARK).pack(pady=(20, 4))
        tk.Label(self, text="Detecting hardware…", font=("Consolas", 8),
                 fg=TEXT_GRAY, bg=BG_DARK).pack()
        self.update_idletasks()

        # ---- DAQ ----------------------------------------------------------
        real, sim = detect_daq_devices() if NIDAQMX_AVAILABLE else ([], [])
        daq_ok = bool(real or sim)
        ds = (f"OK  Physical: {', '.join(real)}" if real else
              (f"~   NI-MAX simulated: {', '.join(sim)}" if sim else
               "--  No NI DAQ found."))
        self._btn("DAQ  (NI 9264 / 9205)", ds, daq_ok,
                  ACCENT_GREEN if real else ACCENT_YELLOW if sim
                  else TEXT_GRAY,
                  lambda: self._choose("DAQ"))

        # ---- camera -------------------------------------------------------
        cams = detect_cameras_named() if DSHOW_BACKEND else []
        if cams:
            cs = f"OK  {cams[0][1]}  (index {cams[0][0]})"
        elif not DSHOW_BACKEND:
            cs = "--  pygrabber/comtypes not installed."
        else:
            cs = "--  No camera detected."
        self._btn("CMOS Camera  (ISH1000)", cs, bool(cams),
                  ACCENT_CYAN if cams else TEXT_GRAY,
                  lambda: self._choose("CMOS Camera"))

        # ---- spectrometer -------------------------------------------------
        specs, spec_err = detect_spectrometers()
        if specs:
            ss = f"OK  {specs[0]['name']}  (SN {specs[0]['serial']})"
        else:
            ss = "--  " + (spec_err.split("\n")[0] if spec_err
                           else "No spectrometer detected.")
        self._btn("Spectrometer  (Avantes AvaSpec)", ss, bool(specs),
                  ACCENT_ORANGE if specs else TEXT_GRAY,
                  lambda: self._choose("Spectrometer"))

        row = tk.Frame(self, bg=BG_DARK); row.pack(pady=(14, 2))
        tk.Button(row, text="Rescan", font=("Consolas", 9, "bold"),
                  padx=12, pady=5, bd=0, relief="flat", cursor="hand2",
                  bg=BG_CARD, fg=ACCENT_CYAN,
                  command=self._build).pack(side="left", padx=6)
        # Entry with no device: opens the normal application on the
        # characterization (Optical Response) tab.
        tk.Button(row, text="Enter program without a device",
                  font=("Consolas", 9, "bold"),
                  padx=12, pady=5, bd=0, relief="flat", cursor="hand2",
                  bg=BG_CARD, fg=TEXT_WHITE,
                  command=lambda: self._choose("DAQ")).pack(side="left",
                                                            padx=6)
        if not (daq_ok or cams or specs):
            tk.Label(self, text="No hardware detected — you can still enter "
                                "the program above.",
                     font=("Consolas", 8), fg=ACCENT_YELLOW,
                     bg=BG_DARK).pack()
        tk.Label(self, text="Close AvaSoft / TCapture first — they hold "
                            "their device exclusively.",
                 font=("Consolas", 7), fg=TEXT_GRAY,
                 bg=BG_DARK).pack()

    def _btn(self, text, sub, ok, colour, cmd):
        f=tk.Frame(self,bg=BG_DARK); f.pack(fill="x",pady=4,padx=30)
        tk.Button(f,text=text,font=("Consolas",11,"bold"),width=30,pady=8,bd=0,
                  relief="flat",cursor="hand2" if ok else "arrow",bg=BG_CARD,
                  fg=colour,state="normal" if ok else "disabled",command=cmd).pack()
        tk.Label(f,text=sub,font=("Consolas",8),fg=colour,bg=BG_DARK,
                 wraplength=420,justify="left").pack()

    def _choose(self, mode):
        self.result=mode; self.destroy()


# ════════════════════════════════════════════════════════════════════════════
#  DYNAMIC CHANNEL ROW
# ════════════════════════════════════════════════════════════════════════════
class ChannelRow(tk.Frame):
    """One user channel:  [editable name] [ao line dropdown] [voltage] [✕].

    The NAME is yours to choose (must be unique).  The PHYSICAL LINE ao0..aoN
    is fixed hardware — you can only pick which line the channel maps to, and
    a line already used by another channel is not offered."""

    def __init__(self, parent, app, name, line, volts):
        super().__init__(parent, bg=BG_PANEL)
        self.app=app
        self.name_var=tk.StringVar(value=name)
        self.line_var=tk.StringVar(value=line)
        self.v_var=tk.StringVar(value=f"{volts:g}")
        self._last_name=name
        e=tk.Entry(self,textvariable=self.name_var,font=("Consolas",9),width=11,
                   bg=BG_CARD,fg=TEXT_WHITE,insertbackground=TEXT_WHITE,
                   relief="flat",justify="left")
        e.pack(side="left",padx=(2,3))
        e.bind("<FocusOut>",self._validate_name)
        e.bind("<Return>",self._validate_name)
        self._prev_line=line
        self.line_menu=ttk.Combobox(self,textvariable=self.line_var,width=5,
                                    state="readonly",font=("Consolas",9))
        self.line_menu.pack(side="left",padx=3)
        self.line_menu.bind("<<ComboboxSelected>>",
                            lambda e:self.app.on_line_change(self))
        tk.Entry(self,textvariable=self.v_var,font=("Consolas",9),width=6,
                 bg=BG_CARD,fg=ACCENT_CYAN,insertbackground=TEXT_WHITE,
                 relief="flat",justify="center").pack(side="left",padx=3)
        tk.Label(self,text="V",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_PANEL).pack(side="left")
        self.del_btn=tk.Button(self,text="✕",font=("Consolas",9,"bold"),bg=BG_PANEL,
                               fg=ACCENT_RED,relief="flat",cursor="hand2",bd=0,
                               command=lambda:self.app.remove_channel(self))
        self.del_btn.pack(side="right",padx=2)

    # ---- values -----------------------------------------------------------
    def name(self):  return self.name_var.get().strip()
    def line(self):  return self.line_var.get().strip()          # "ao3"
    def phys(self):  return f"{AO_DEVICE}/{self.line()}"          # full path

    def volts(self):
        try: raw=float(self.v_var.get())
        except ValueError: return 0.0
        v=clamp(raw,V_MIN,V_MAX)
        if v!=raw: self.v_var.set(f"{v:g}")
        return v

    # ---- name uniqueness --------------------------------------------------
    def _validate_name(self, *_):
        n=self.name()
        if not n:
            messagebox.showerror("Channel name","Name cannot be empty.")
            self.name_var.set(self._last_name); return
        others={r.name().lower() for r in self.app.channel_rows if r is not self}
        if n.lower() in others:
            messagebox.showerror("Channel name",
                f"'{n}' is already used by another channel.\nNames must be unique.")
            self.name_var.set(self._last_name); return
        self._last_name=n
        self.app.refresh_channel_ui()


# ════════════════════════════════════════════════════════════════════════════
#  STANDALONE LC DRIVE
#
#  Why this exists:  the AO task in _stream_daq is created with
#  `with nidaqmx.Task() as ao:` INSIDE that function, so it lives only for the
#  duration of an Optical Response run.  _worker_camera and _worker_spectrum
#  never created an AO task at all, so the cell saw 0 V whenever the camera or
#  spectrometer was the active tab -- which is why applying channel voltages
#  appeared to do nothing there.
#
#  This task is owned by the application instead of by one worker, so the cell
#  stays driven while ANY acquisition runs.  The NI 9264 regenerates the
#  waveform from its own buffer, so no CPU involvement is needed once started.
#
#  Every channel carries EXACTLY zero DC (equal +/- half cycles, integer
#  number of periods in the buffer).  This is not cosmetic: a DC component
#  across a liquid crystal drives ion migration and electrochemically degrades
#  the cell permanently.
# ════════════════════════════════════════════════════════════════════════════
class DriveOutput:
    """A continuously regenerating AO waveform on one or more channels."""

    def __init__(self):
        self._task = None
        self.channels = []

    def start(self, chans, bufs, ao_rate):
        """chans: ['Dev/ao0', ...]   bufs: 2-D array (channels x samples)."""
        self._task = nidaqmx.Task()
        try:
            for c in chans:
                self._task.ao_channels.add_ao_voltage_chan(
                    c, min_val=AO_VMIN, max_val=AO_VMAX)
            self._task.timing.cfg_samp_clk_timing(
                ao_rate, sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=int(bufs.shape[-1]))
            # nidaqmx wants a flat list for one channel, list-of-lists for many
            data = (bufs[0].tolist() if len(chans) == 1
                    else [b.tolist() for b in bufs])
            self._task.write(data, auto_start=False)
            self._task.start()
            self.channels = list(chans)
        except Exception:
            self.stop()
            raise
        return self

    def is_running(self):
        return self._task is not None

    def stop(self):
        if self._task is None:
            return
        try:
            self._task.stop()
        except Exception:
            pass
        try:
            self._task.close()
        except Exception:
            pass
        self._task = None
        self.channels = []


# ════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════
class LCApp(tk.Tk):

    # ── construction ───────────────────────────────────────────────────────
    def __init__(self, device_mode, initial_tab=0):
        super().__init__()
        self.device_mode=device_mode
        self._initial_tab=int(initial_tab)
        self.running=False
        self.worker=None
        self.dark_offset_v=0.0
        self.settings=self._load_settings()
        self.dark_offset_v=float(self.settings.get("dark_offset_v",0.0))
        # spectral calibration frames (T(λ) = (S − Dark)/(Ref − Dark))
        self.spec=None                # opened lazily on first use
        self.spec_wl=None
        self.spec_dark=None
        self.spec_ref=None
        self._spec_capture_req=None   # "dark"/"ref" while a run is live

        # hardware discovery ------------------------------------------------
        global AO_DEVICE, AI_DEVICE
        ao_dev,ai_dev,self.n_ao,self.n_ai=resolve_daq_modules()
        if ao_dev: AO_DEVICE=ao_dev
        if ai_dev: AI_DEVICE=ai_dev
        self.has_ao=bool(ao_dev) if NIDAQMX_AVAILABLE else False
        self.ao_lines=[f"ao{i}" for i in range(self.n_ao)]
        self.ai_lines=[f"ai{i}" for i in range(self.n_ai)]

        self.title(f"LC Characterization · {device_mode}")
        self.geometry("1500x900"); self.configure(bg=BG_DARK)
        self.protocol("WM_DELETE_WINDOW",self.on_exit)

        self._build_topbar()
        body=tk.Frame(self,bg=BG_DARK); body.pack(fill="both",expand=True)
        self._build_sidebar(body)
        self._build_tabs(body)
        self._restore_channels()
        self.refresh_channel_ui()
        # Open on the tab for the device chosen at startup, instead of always
        # landing on Optical Response.
        try:
            self.nb.select(self._initial_tab)
        except Exception:
            pass
        self._status(f"Ready · AO={AO_DEVICE} ({self.n_ao} lines) · "
                     f"AI={AI_DEVICE} ({self.n_ai} lines)"
                     if NIDAQMX_AVAILABLE else
                     "Ready · NI-DAQmx NOT installed (photodiode disabled)")

    # ── top bar ────────────────────────────────────────────────────────────
    def _build_topbar(self):
        bar=tk.Frame(self,bg=BG_PANEL); bar.pack(fill="x")
        self.run_btn=tk.Button(bar,text="▶  Run",font=("Consolas",10,"bold"),
                               bg=BTN_RUN,fg=TEXT_WHITE,relief="flat",cursor="hand2",
                               padx=16,pady=6,command=self.start_run)
        self.run_btn.pack(side="left",padx=(10,4),pady=6)
        tk.Button(bar,text="■  Stop",font=("Consolas",10,"bold"),bg=BTN_STOP,
                  fg=TEXT_WHITE,relief="flat",cursor="hand2",padx=16,pady=6,
                  command=self.stop_run).pack(side="left",padx=4,pady=6)
        self.status_var=tk.StringVar(value="")
        tk.Label(bar,textvariable=self.status_var,font=("Consolas",9),
                 fg=TEXT_GRAY,bg=BG_PANEL).pack(side="left",padx=16)
        # Device is NOT user-selectable — the active tab decides it.
        df=tk.Frame(bar,bg=BG_PANEL); df.pack(side="right",padx=12)
        tk.Label(df,text="Device:",font=("Consolas",9,"bold"),
                 fg=TEXT_GRAY,bg=BG_PANEL).pack(side="left",padx=(0,4))
        self.device_var=tk.StringVar(value=self.device_mode)
        tk.Label(df,textvariable=self.device_var,font=("Consolas",9,"bold"),
                 fg=ACCENT_CYAN,bg=BG_CARD,padx=10,pady=2,
                 width=14).pack(side="left")

    def _status(self, txt):
        self.status_var.set(txt)

    # ── sidebar ────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        side=tk.Frame(parent,bg=BG_PANEL,width=300); side.pack(side="left",fill="y")
        side.pack_propagate(False)

        def section(t):
            tk.Label(side,text=t,font=("Consolas",9,"bold"),fg=ACCENT_CYAN,
                     bg=BG_PANEL,anchor="w").pack(fill="x",padx=8,pady=(12,2))
            tk.Frame(side,bg=BG_CARD,height=1).pack(fill="x",padx=8)

        # waveform ----------------------------------------------------------
        section("WAVEFORM")
        rf=tk.Frame(side,bg=BG_PANEL); rf.pack(fill="x",padx=8,pady=2)
        tk.Label(rf,text="Freq (Hz):",font=("Consolas",9),fg=TEXT_GRAY,
                 bg=BG_PANEL,width=10,anchor="w").pack(side="left")
        self.freq_var=tk.StringVar(value=str(self.settings.get("freq",1000)))
        tk.Entry(rf,textvariable=self.freq_var,font=("Consolas",9),width=8,
                 bg=BG_CARD,fg=TEXT_WHITE,insertbackground=TEXT_WHITE,
                 relief="flat",justify="center").pack(side="left",padx=4)
        wf=tk.Frame(side,bg=BG_PANEL); wf.pack(fill="x",padx=8,pady=2)
        self.wave_var=tk.StringVar(value=self.settings.get("wave","Square"))
        for w in ("Square","Sine"):
            tk.Radiobutton(wf,text=w,variable=self.wave_var,value=w,
                           font=("Consolas",9),fg=TEXT_WHITE,bg=BG_PANEL,
                           selectcolor=BG_CARD,activebackground=BG_PANEL,
                           activeforeground=TEXT_WHITE).pack(side="left",padx=4)

        # channels ----------------------------------------------------------
        section(f"CHANNELS  (max {self.n_ao})")
        self.chan_frame=tk.Frame(side,bg=BG_PANEL)
        self.chan_frame.pack(fill="x",padx=6,pady=2)
        self.channel_rows=[]
        ab=tk.Frame(side,bg=BG_PANEL); ab.pack(fill="x",padx=8,pady=2)
        self.add_btn=tk.Button(ab,text="＋  Add Channel",font=("Consolas",8,"bold"),
                               bg=BG_CARD,fg=ACCENT_GREEN,relief="flat",
                               cursor="hand2",pady=4,command=self.add_channel)
        self.add_btn.pack(fill="x")
        self.chan_hint=tk.StringVar(value="")
        tk.Label(side,textvariable=self.chan_hint,font=("Consolas",7),fg=TEXT_GRAY,
                 bg=BG_PANEL,anchor="w").pack(fill="x",padx=8)

        # drive selection ---------------------------------------------------
        section("DRIVE CHANNEL")
        tk.Label(side,text="Voltage & AO line are taken from\nthe channel "
                 "selected here:",font=("Consolas",8),fg=TEXT_GRAY,bg=BG_PANEL,
                 justify="left",anchor="w").pack(fill="x",padx=8)
        self.drive_var=tk.StringVar(value=self.settings.get("drive",""))
        self.drive_menu=ttk.Combobox(side,textvariable=self.drive_var,
                                     state="readonly",font=("Consolas",9))
        self.drive_menu.pack(fill="x",padx=8,pady=4)

        # dark --------------------------------------------------------------
        section("DARK / ZERO")
        tk.Button(side,text="🌑  Measure Dark / Zero",font=("Consolas",8,"bold"),
                  bg=BG_CARD,fg=ACCENT_ORANGE,relief="flat",cursor="hand2",pady=4,
                  command=self._measure_dark).pack(fill="x",padx=8,pady=(4,2))
        self.dark_var=tk.StringVar(
            value=f"Dark offset: {self.dark_offset_v:.6f} V")
        tk.Label(side,textvariable=self.dark_var,font=("Consolas",7),fg=TEXT_GRAY,
                 bg=BG_PANEL,anchor="w",wraplength=270,
                 justify="left").pack(fill="x",padx=8)

    # ── tabs ───────────────────────────────────────────────────────────────
    def _build_tabs(self, parent):
        style=ttk.Style(self)
        try: style.theme_use("clam")
        except Exception: pass
        style.configure("TNotebook",background=BG_DARK,borderwidth=0)
        style.configure("TNotebook.Tab",background=BG_PANEL,foreground=TEXT_GRAY,
                        font=("Consolas",10,"bold"),padding=(18,8))
        style.map("TNotebook.Tab",
                  background=[("selected",BG_CARD)],
                  foreground=[("selected",ACCENT_CYAN)])
        style.configure("TCombobox",
                        fieldbackground=BG_CARD,background=BG_PANEL,
                        foreground=ACCENT_YELLOW,arrowcolor=ACCENT_CYAN,
                        bordercolor=ACCENT_CYAN,lightcolor=BG_CARD,
                        darkcolor=BG_CARD,selectbackground=BG_CARD,
                        selectforeground=ACCENT_YELLOW,padding=3)
        style.map("TCombobox",
                  fieldbackground=[("readonly",BG_CARD)],
                  foreground=[("readonly",ACCENT_YELLOW)],
                  arrowcolor=[("hover",ACCENT_YELLOW)])
        # the popup list is a plain Tk Listbox — style it via option database
        self.option_add("*TCombobox*Listbox.background",BG_CARD)
        self.option_add("*TCombobox*Listbox.foreground",TEXT_WHITE)
        self.option_add("*TCombobox*Listbox.selectBackground",ACCENT_CYAN)
        self.option_add("*TCombobox*Listbox.selectForeground",BG_DARK)
        self.option_add("*TCombobox*Listbox.font",("Consolas",10))

        self.nb=ttk.Notebook(parent); self.nb.pack(fill="both",expand=True,
                                                   padx=8,pady=8)
        self.tab_cont=self._make_tab("Optical Response",with_ai=True)
        self.tab_cam=self._make_tab("CMOS Camera",with_ai=False)
        self.tab_spec=self._make_tab("Spectrometer",with_ai=False)
        self.tab_sweep=self.tab_spec        # alias: old sweep code paths
        self.nb.bind("<<NotebookTabChanged>>",self._on_tab_change)
        self._build_cont_extras()
        self._build_camera_extras()

        # ── SPECTROMETER controls ─────────────────────────────────────────
        self._build_spec_extras()

    def _build_spec_extras(self):
        """Two rows: acquisition + mode on top, axis scaling underneath."""
        sf = self.tab_spec["params"]
        row1 = tk.Frame(sf, bg=BG_DARK); row1.pack(fill="x")
        row2 = tk.Frame(sf, bg=BG_DARK); row2.pack(fill="x", pady=(2, 0))

        def lbl(parent, text, pad=(10, 2), fg=TEXT_GRAY):
            tk.Label(parent, text=text, font=("Consolas", 8), fg=fg,
                     bg=BG_DARK).pack(side="left", padx=pad)

        def ent(parent, var, width=6, fg=TEXT_WHITE):
            e = tk.Entry(parent, textvariable=var, font=("Consolas", 8),
                         width=width, bg=BG_CARD, fg=fg,
                         insertbackground=TEXT_WHITE, relief="flat",
                         justify="center")
            e.pack(side="left")
            return e

        def btn(parent, text, colour, cmd, pad=2):
            tk.Button(parent, text=text, font=("Consolas", 8, "bold"),
                      bg=BG_CARD, fg=colour, relief="flat", cursor="hand2",
                      padx=8, pady=2, command=cmd).pack(side="left", padx=pad)

        # ---- row 1: acquisition -------------------------------------------
        lbl(row1, "Integration (ms):")
        self.integ_var = tk.StringVar(
            value=str(self.settings.get("spec_integration_ms", 10)))
        ent(row1, self.integ_var)

        lbl(row1, "Averaging:", pad=(8, 2))
        self.avg_var = tk.StringVar(
            value=str(self.settings.get("spec_averages", 10)))
        ent(row1, self.avg_var, width=5)

        lbl(row1, "Mode:", pad=(10, 2))
        self.spec_mode_var = tk.StringVar(
            value=self.settings.get("spec_mode", self.SPEC_MODES[2]))
        cb = ttk.Combobox(row1, textvariable=self.spec_mode_var,
                          values=list(self.SPEC_MODES), width=18,
                          state="readonly", font=("Consolas", 8))
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", self._on_spec_mode_change)

        btn(row1, "Capture DARK", ACCENT_ORANGE,
            lambda: self._capture_spec("dark"), pad=(12, 2))
        btn(row1, "Capture REFERENCE", ACCENT_YELLOW,
            lambda: self._capture_spec("ref"))
        btn(row1, "Clear", TEXT_GRAY, self._clear_spec_calib)
        btn(row1, "Save spectrum", ACCENT_GREEN, self._save_spectrum,
            pad=(12, 2))

        self.spec_status = tk.StringVar(value="Dark: —   Ref: —")
        tk.Label(row1, textvariable=self.spec_status, font=("Consolas", 8),
                 fg=ACCENT_CYAN, bg=BG_DARK).pack(side="left", padx=10)

        # ---- row 2: axis scaling ------------------------------------------
        lbl(row2, "X [nm]:")
        self.x_min_var = tk.StringVar(value=self.settings.get("spec_x_min", ""))
        self.x_max_var = tk.StringVar(value=self.settings.get("spec_x_max", ""))
        ent(row2, self.x_min_var, width=7, fg=ACCENT_CYAN)
        lbl(row2, "to", pad=(2, 2))
        ent(row2, self.x_max_var, width=7, fg=ACCENT_CYAN)
        self.spec_xauto_var = tk.BooleanVar(
            value=bool(self.settings.get("spec_x_auto", True)))
        tk.Checkbutton(row2, text="auto", variable=self.spec_xauto_var,
                       font=("Consolas", 8), fg=TEXT_GRAY, bg=BG_DARK,
                       selectcolor=BG_CARD, activebackground=BG_DARK,
                       activeforeground=ACCENT_CYAN, relief="flat", bd=0,
                       command=self._spec_apply_scale).pack(side="left",
                                                            padx=(3, 8))

        lbl(row2, "Y:", pad=(6, 2))
        self.y_min_var = tk.StringVar(value=self.settings.get("spec_y_min", ""))
        self.y_max_var = tk.StringVar(value=self.settings.get("spec_y_max", ""))
        ent(row2, self.y_min_var, width=7, fg=ACCENT_CYAN)
        lbl(row2, "to", pad=(2, 2))
        ent(row2, self.y_max_var, width=7, fg=ACCENT_CYAN)
        self.spec_yauto_var = tk.BooleanVar(
            value=bool(self.settings.get("spec_y_auto", True)))
        tk.Checkbutton(row2, text="auto", variable=self.spec_yauto_var,
                       font=("Consolas", 8), fg=TEXT_GRAY, bg=BG_DARK,
                       selectcolor=BG_CARD, activebackground=BG_DARK,
                       activeforeground=ACCENT_CYAN, relief="flat", bd=0,
                       command=self._spec_apply_scale).pack(side="left",
                                                            padx=(3, 8))

        btn(row2, "Apply scale", ACCENT_CYAN, self._spec_apply_scale,
            pad=(6, 2))
        btn(row2, "Grab current view", TEXT_WHITE, self._spec_autoscale_now)
        btn(row2, "Reset to auto", ACCENT_GREEN, self._spec_reset_scale)

        for v in (self.x_min_var, self.x_max_var,
                  self.y_min_var, self.y_max_var):
            v.trace_add("write", lambda *_: None)   # entries applied on demand

        self._update_spec_status()
        self._spec_hint()

    def _on_spec_mode_change(self, *_):
        self._spec_hint()
        self._save_settings()

    def _spec_hint(self):
        """Tell the user exactly what the selected mode still needs."""
        mode = self.spec_mode_var.get()
        need_d, need_r = self._spec_needs(mode)
        missing = []
        if need_d and self.spec_dark is None:
            missing.append("DARK")
        if need_r and self.spec_ref is None:
            missing.append("REFERENCE")
        if missing:
            self._status(f"{mode}: capture {' and '.join(missing)} "
                         f"before pressing Run.")
        else:
            self._status(f"{mode}: ready — press Run for continuous "
                         f"acquisition.")

    def _make_tab(self, title, with_ai):
        outer=tk.Frame(self.nb,bg=BG_DARK)
        self.nb.add(outer,text=title)
        head=tk.Frame(outer,bg=BG_DARK); head.pack(fill="x")
        params=tk.Frame(head,bg=BG_DARK); params.pack(side="left")
        ai_var=None
        if with_ai:
            # AI dropdown, top-right, OUTSIDE the graph — Optical tab only
            af=tk.Frame(head,bg=BG_DARK); af.pack(side="right",padx=10,pady=4)
            tk.Label(af,text="AI channel:",font=("Consolas",9),fg=TEXT_GRAY,
                     bg=BG_DARK).pack(side="left",padx=(0,4))
            ai_var=tk.StringVar(value=self.settings.get("ai_optical",
                                self.ai_lines[0] if self.ai_lines else "ai0"))
            cb=ttk.Combobox(af,textvariable=ai_var,values=self.ai_lines,width=6,
                            state="readonly",font=("Consolas",9))
            cb.pack(side="left")
        fig,ax=plt.subplots(figsize=(9,5.4),dpi=100,facecolor=BG_DARK)
        ax.set_facecolor(PLOT_BG)
        canvas=FigureCanvasTkAgg(fig,master=outer)
        # toolbar (zoom / pan / reset)
        tbf=tk.Frame(outer,bg=BG_CARD); tbf.pack(fill="x",side="bottom")
        try: tb=NavigationToolbar2Tk(canvas,tbf,pack_toolbar=False)
        except TypeError: tb=NavigationToolbar2Tk(canvas,tbf)
        try:
            tb.config(bg=BG_CARD)
            for ch in tb.winfo_children():
                try: ch.config(bg=BG_CARD,fg=TEXT_WHITE,relief="flat",bd=0,
                               padx=6,pady=6,activebackground=ACCENT_CYAN)
                except Exception: pass
        except Exception: pass
        tb.update(); tb.pack(side="left",padx=4,pady=2)
        tk.Label(tbf,text="  ⌕ ZOOM: drag a box  ·  ✥ PAN  ·  ⌂ RESET  ·  "
                          "scroll wheel = zoom at cursor",
                 font=("Consolas",8,"bold"),fg=ACCENT_YELLOW,
                 bg=BG_CARD).pack(side="left",padx=8)
        canvas.get_tk_widget().pack(fill="both",expand=True)
        canvas.mpl_connect("scroll_event",
                           lambda e,c=canvas:self._scroll_zoom(e,c))
        tab={"frame":outer,"fig":fig,"ax":ax,"canvas":canvas,
             "ai_var":ai_var,"params":params,"title":title}
        self._enable_drag_pan(tab)
        return tab

    def _enable_drag_pan(self, tab):
        """Left-click drag moves the view — no toolbar mode needed.

        On the Spectrometer tab the resulting limits are written into the
        manual X/Y boxes and auto-scale is switched off, so the panned view
        SURVIVES the continuous redraw instead of snapping back on the next
        scan.  'Reset to auto' restores auto-scaling."""
        canvas=tab["canvas"]
        st={"anchor":None,"moved":False}

        def press(e):
            if e.button!=1 or e.inaxes is not tab["ax"]:
                return
            tb=getattr(canvas,"toolbar",None)
            if tb is not None and getattr(tb,"mode",""):
                return                      # toolbar zoom/pan owns the mouse
            if e.xdata is None or e.ydata is None:
                return
            st["anchor"]=(e.xdata,e.ydata)
            st["moved"]=False

        def move(e):
            if st["anchor"] is None or e.inaxes is not tab["ax"]:
                return
            if e.xdata is None or e.ydata is None:
                return
            ax=tab["ax"]
            dx=st["anchor"][0]-e.xdata
            dy=st["anchor"][1]-e.ydata
            if abs(dx)<1e-12 and abs(dy)<1e-12:
                return
            x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
            ax.set_xlim(x0+dx,x1+dx)
            ax.set_ylim(y0+dy,y1+dy)
            st["moved"]=True
            canvas.draw_idle()

        def release(e):
            if st["anchor"] is None:
                return
            st["anchor"]=None
            if st["moved"] and getattr(self,"tab_spec",None) is tab:
                # persist the panned view so live redraws keep it
                ax=tab["ax"]
                x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
                self.x_min_var.set(f"{x0:.6g}"); self.x_max_var.set(f"{x1:.6g}")
                self.y_min_var.set(f"{y0:.6g}"); self.y_max_var.set(f"{y1:.6g}")
                self.spec_xauto_var.set(False); self.spec_yauto_var.set(False)
                self._save_settings()

        canvas.mpl_connect("button_press_event",press)
        canvas.mpl_connect("motion_notify_event",move)
        canvas.mpl_connect("button_release_event",release)

    # ── OPTICAL-TAB EXTRAS: save last N samples ────────────────────────────
    def _build_cont_extras(self):
        pf=self.tab_cont["params"]
        tk.Label(pf,text="Save last",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=(10,2))
        self.save_n_var=tk.StringVar(value=str(self.settings.get("save_n",500)))
        tk.Entry(pf,textvariable=self.save_n_var,font=("Consolas",8),width=7,
                 bg=BG_CARD,fg=TEXT_WHITE,insertbackground=TEXT_WHITE,
                 relief="flat",justify="center").pack(side="left")
        tk.Label(pf,text="samples as",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=2)
        self.save_fmt_var=tk.StringVar(value=self.settings.get("save_fmt","txt"))
        ttk.Combobox(pf,textvariable=self.save_fmt_var,
                     values=["txt","csv","xlsx"],width=5,state="readonly",
                     font=("Consolas",8)).pack(side="left",padx=2)
        tk.Button(pf,text="💾 Save data",font=("Consolas",8,"bold"),bg=BG_CARD,
                  fg=ACCENT_GREEN,relief="flat",cursor="hand2",padx=8,pady=2,
                  command=self._save_stream_data).pack(side="left",padx=6)
        # Y-axis scale for the Optical Response plot.  "Fixed" locks the axis
        # to the min/max boxes (default 0–10 V); unchecked = auto-zoom to the
        # signal.  A photodiode reading ~2.4 V ±5 mV becomes a flat line on a
        # 0–10 V axis, so this is a toggle rather than a permanent lock.
        tk.Label(pf,text="Y:",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=(14,2))
        self.opt_ymin_var=tk.StringVar(
            value=str(self.settings.get("opt_y_min",0)))
        self.opt_ymax_var=tk.StringVar(
            value=str(self.settings.get("opt_y_max",10)))
        tk.Entry(pf,textvariable=self.opt_ymin_var,font=("Consolas",8),width=6,
                 bg=BG_CARD,fg=ACCENT_CYAN,insertbackground=TEXT_WHITE,
                 relief="flat",justify="center").pack(side="left")
        tk.Label(pf,text="to",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=2)
        tk.Entry(pf,textvariable=self.opt_ymax_var,font=("Consolas",8),width=6,
                 bg=BG_CARD,fg=ACCENT_CYAN,insertbackground=TEXT_WHITE,
                 relief="flat",justify="center").pack(side="left")
        tk.Label(pf,text="V",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=(1,4))
        self.opt_yfixed_var=tk.BooleanVar(
            value=bool(self.settings.get("opt_y_fixed",True)))
        tk.Checkbutton(pf,text="Fixed",variable=self.opt_yfixed_var,
                       font=("Consolas",8),fg=TEXT_GRAY,bg=BG_DARK,
                       selectcolor=BG_CARD,activebackground=BG_DARK,
                       activeforeground=ACCENT_CYAN,relief="flat",bd=0,
                       command=self._save_settings).pack(side="left")

    def _save_stream_data(self):
        buf=getattr(self,"_save_buf",None)
        if not buf:
            messagebox.showinfo("Save","No streamed data yet — press Run first.")
            return
        try:
            n=max(1,int(float(self.save_n_var.get())))
        except ValueError:
            messagebox.showerror("Save","Sample count must be a number."); return
        y=np.asarray(buf,dtype=float)[-n:]
        t=np.arange(y.size)/float(getattr(self,"_save_rate",AI_RATE))
        os.makedirs(RESULTS_DIR,exist_ok=True)
        fmt=self.save_fmt_var.get()
        stamp=int(time.time())
        try:
            if fmt=="xlsx":
                try:
                    from openpyxl import Workbook
                except ImportError:
                    messagebox.showerror("Save",
                        "Excel export needs the 'openpyxl' package.\n\n"
                        "Install it with:\n    pip install openpyxl\n\n"
                        "or choose txt/csv instead.")
                    return
                path=os.path.join(RESULTS_DIR,f"Optical_{stamp}.xlsx")
                wb=Workbook(); ws=wb.active; ws.title="Optical"
                ws.append(["time_s","optical_V"])
                for a,b in zip(t,y): ws.append([float(a),float(b)])
                wb.save(path)
            else:
                path=os.path.join(RESULTS_DIR,f"Optical_{stamp}.{fmt}")
                np.savetxt(path,np.column_stack((t,y)),
                           delimiter=("\t" if fmt=="txt" else ","),
                           header="time_s\toptical_V" if fmt=="txt"
                                  else "time_s,optical_V",comments="")
            self._save_settings()
            self._status(f"Saved {y.size} samples — {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save",str(e))

    # ── CAMERA-TAB EXTRAS: exposure / gain / image save (exclusive) ────────
    def _build_camera_extras(self):
        pf=self.tab_cam["params"]
        tk.Label(pf,text="Exposure (ms):",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=(10,2))
        self.exp_var=tk.StringVar(value=str(self.settings.get("exposure_ms",30)))
        tk.Entry(pf,textvariable=self.exp_var,font=("Consolas",8),width=6,
                 bg=BG_CARD,fg=TEXT_WHITE,insertbackground=TEXT_WHITE,
                 relief="flat",justify="center").pack(side="left")
        tk.Label(pf,text="Gain:",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=(8,2))
        self.gain_var=tk.StringVar(value=str(self.settings.get("gain",1)))
        tk.Entry(pf,textvariable=self.gain_var,font=("Consolas",8),width=5,
                 bg=BG_CARD,fg=TEXT_WHITE,insertbackground=TEXT_WHITE,
                 relief="flat",justify="center").pack(side="left")
        tk.Label(pf,text="FPS:",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=(8,2))
        self.fps_var=tk.StringVar(value=str(self.settings.get("camera_fps",10)))
        fps_spin=tk.Spinbox(pf,from_=1,to=120,increment=1,
                            textvariable=self.fps_var,font=("Consolas",8),
                            width=4,bg=BG_CARD,fg=TEXT_WHITE,
                            insertbackground=TEXT_WHITE,relief="flat",
                            justify="center",
                            command=self._apply_fps_live)
        fps_spin.pack(side="left")
        # Also react to typed values (not just the spinner arrows) and to
        # the mouse wheel, since that is the fastest way to nudge FPS while
        # watching the live view for the sweet spot.
        fps_spin.bind("<Return>",lambda e:self._apply_fps_live())
        fps_spin.bind("<FocusOut>",lambda e:self._apply_fps_live())
        fps_spin.bind("<MouseWheel>",self._fps_mousewheel)
        tk.Button(pf,text="Apply",font=("Consolas",8,"bold"),bg=BG_CARD,
                  fg=ACCENT_CYAN,relief="flat",cursor="hand2",padx=8,pady=2,
                  command=self._apply_cam_settings).pack(side="left",padx=6)
        tk.Label(pf,text="Save image as",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=(14,2))
        self.img_fmt_var=tk.StringVar(value=self.settings.get("img_fmt","png"))
        ttk.Combobox(pf,textvariable=self.img_fmt_var,
                     values=["png","jpg","tiff","npy"],width=5,state="readonly",
                     font=("Consolas",8)).pack(side="left",padx=2)
        tk.Button(pf,text="📸 Save image",font=("Consolas",8,"bold"),bg=BG_CARD,
                  fg=ACCENT_GREEN,relief="flat",cursor="hand2",padx=8,pady=2,
                  command=self._save_camera_image).pack(side="left",padx=4)
        # camera chooser — lists EVERY readable camera by name so the ISH1000
        # can be picked over a laptop webcam
        tk.Label(pf,text="Camera:",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=(14,2))
        self.cam_sel_var=tk.StringVar(value=self.settings.get("camera_sel",""))
        self.cam_menu=ttk.Combobox(pf,textvariable=self.cam_sel_var,width=24,
                                   state="readonly",font=("Consolas",8))
        self.cam_menu.pack(side="left",padx=2)
        tk.Button(pf,text="↻",font=("Consolas",8,"bold"),bg=BG_CARD,
                  fg=ACCENT_CYAN,relief="flat",cursor="hand2",padx=6,pady=2,
                  command=self._refresh_cameras).pack(side="left",padx=2)
        tk.Button(pf,text="Info",font=("Consolas",8,"bold"),bg=BG_CARD,
                  fg=ACCENT_YELLOW,relief="flat",cursor="hand2",padx=6,pady=2,
                  command=self._show_camera_info).pack(side="left",padx=2)
        # ---- orientation row ------------------------------------------------
        of=tk.Frame(self.tab_cam["params"],bg=BG_DARK); of.pack(fill="x",pady=(2,0))
        tk.Label(of,text="Orientation:",font=("Consolas",8),fg=TEXT_GRAY,
                 bg=BG_DARK).pack(side="left",padx=(10,4))
        self.cam_rot_var=tk.IntVar(value=int(self.settings.get("cam_rot",0))%4)
        self.cam_rot_lbl=tk.StringVar()
        tk.Button(of,textvariable=self.cam_rot_lbl,font=("Consolas",8,"bold"),
                  bg=BG_CARD,fg=ACCENT_CYAN,relief="flat",cursor="hand2",
                  padx=8,pady=2,command=self._cam_rotate_90).pack(side="left")
        self.cam_fliph_var=tk.BooleanVar(
            value=bool(self.settings.get("cam_flip_h",False)))
        self.cam_flipv_var=tk.BooleanVar(
            value=bool(self.settings.get("cam_flip_v",False)))
        for text,var in (("Flip H",self.cam_fliph_var),
                         ("Flip V",self.cam_flipv_var)):
            tk.Checkbutton(of,text=text,variable=var,font=("Consolas",8),
                           fg=TEXT_GRAY,bg=BG_DARK,selectcolor=BG_CARD,
                           activebackground=BG_DARK,
                           activeforeground=ACCENT_CYAN,relief="flat",bd=0,
                           command=self._apply_cam_orientation
                           ).pack(side="left",padx=4)
        self._update_rot_label()
        # Vendor driver dialog: the reliable way to set gain/exposure on
        # filters that do not implement the standard DirectShow control
        # interfaces (most scientific-camera filters, TUCam included).
        tk.Button(of,text="Driver settings…",font=("Consolas",8,"bold"),
                  bg=BG_CARD,fg=ACCENT_YELLOW,relief="flat",cursor="hand2",
                  padx=8,pady=2,
                  command=self._open_cam_driver_dialog).pack(side="left",
                                                             padx=(14,2))
        self._cam_obj=None
        self._cam_lock=threading.Lock()

    def _open_cam_driver_dialog(self):
        """Open the camera filter's own property page (exposure, gain,
        white balance — whatever the vendor exposes).  Streaming pauses
        while the dialog is open and resumes when it closes."""
        cam=getattr(self,"_cam_obj",None)
        if cam is None or not getattr(cam,"is_open",lambda:False)():
            messagebox.showinfo("Driver settings",
                "Start the camera first (press Run), then open the driver "
                "settings.")
            return
        self._status("Driver dialog open — live view pauses until you "
                     "close it…")
        def worker():
            notes=cam.show_properties()
            self.after(0,lambda n=notes:
                       self._status("Driver settings: "+" · ".join(n)))
        threading.Thread(target=worker,daemon=True).start()

    def _update_rot_label(self):
        self.cam_rot_lbl.set(f"Rotate 90° (now {self.cam_rot_var.get()*90}°)")

    def _cam_rotate_90(self):
        self.cam_rot_var.set((self.cam_rot_var.get()+1)%4)
        self._update_rot_label()
        self._apply_cam_orientation()

    def _apply_cam_orientation(self):
        """Push orientation to a running camera immediately, persist always."""
        cam=getattr(self,"_cam_obj",None)
        if cam is not None and getattr(cam,"is_open",lambda:False)():
            cam.set_orientation(flip_h=self.cam_fliph_var.get(),
                                flip_v=self.cam_flipv_var.get(),
                                rot_quarters=self.cam_rot_var.get())
        self._save_settings()
        self._last_frame=None
        self._cam_list=[]
        self._dshow_names=[]
        self._dshow_name_error=""
        self._ish_usb=[]
        self._ish_usb_error=""
        self._refresh_cameras(quiet=True)

    def _refresh_cameras(self, quiet=False):
        if self.running and not quiet:
            messagebox.showinfo("Cameras",
                                "Stop the live camera before rescanning.")
            return
        self._dshow_names,self._dshow_name_error=_camera_names_info()
        self._ish_usb,self._ish_usb_error=_ish1000_usb_devices()
        self._cam_list=detect_cameras_named()
        opts=[f"{i}: {nm}" for i,nm in self._cam_list]
        self.cam_menu["values"]=opts
        if opts and self.cam_sel_var.get() not in opts:
            self.cam_sel_var.set(opts[0])
        if not quiet:
            named_ish=any(_is_ish1000_name(nm) for _,nm in self._cam_list)
            if named_ish:
                self._status(f"ISH1000 verified in DirectShow · "
                             f"{len(opts)} readable camera(s)")
            elif self._ish_usb:
                self._status("ISH1000 detected by USB ID, but its DirectShow "
                             "source is not verified")
            else:
                self._status(f"{len(opts)} readable camera(s) · ISH1000 USB "
                             "device not detected")
            self._show_camera_info()

    def _camera_diagnostic_text(self):
        lines=[]
        if self._ish_usb:
            for dev in self._ish_usb:
                status=str(dev.get("Status","Unknown"))
                name=str(dev.get("FriendlyName") or "Tucsen ISH1000")
                lines.append(f"USB hardware: FOUND — {name} (status: {status})")
        else:
            lines.append("USB hardware: ISH1000 VID/PID not found")
        if self._ish_usb_error:
            lines.append(f"USB query: {self._ish_usb_error}")

        if self._dshow_names:
            lines.append("\nDirectShow device names:")
            for i,name in enumerate(self._dshow_names):
                tag="  ← ISH1000 match" if _is_ish1000_name(name) else ""
                lines.append(f"  {i}: {name}{tag}")
        else:
            lines.append("\nDirectShow device names: unavailable")
        if self._dshow_name_error:
            lines.append(self._dshow_name_error)

        lines.append("\nReadable OpenCV sources:")
        if self._cam_list:
            for i,name in self._cam_list:
                verified="verified ISH1000" if _is_ish1000_name(name) \
                         else "not verified as ISH1000"
                lines.append(f"  {i}: {name} — {verified}")
        else:
            lines.append("  none")

        if self._ish_usb and not any(_is_ish1000_name(nm)
                                     for _,nm in self._cam_list):
            lines.append(
                "\nWindows sees the physical ISH1000, but this GUI cannot yet "
                "prove that any readable DirectShow index is that camera. "
                "Close TCapture completely before scanning. If names are "
                "unavailable, install pygrabber/comtypes. If the ISH1000 still "
                "does not appear by name, the legacy H-Series driver is only "
                "serving TCapture and acquisition will require a DirectShow "
                "component or the Tucsen SDK."
            )
        elif not self._ish_usb:
            lines.append(
                "\nCheck the USB cable/port and Device Manager. The ISH1000 "
                "hardware IDs are VID_5453&PID_A803 or VID_0547&PID_A003."
            )
        return "\n".join(lines)

    def _show_camera_info(self):
        messagebox.showinfo("Camera diagnostics",
                            self._camera_diagnostic_text())

    def _selected_cam_index(self):
        sel=self.cam_sel_var.get()
        m=re.match(r"\s*(\d+)\s*:",sel)
        if m: return int(m.group(1))
        return self._cam_list[0][0] if self._cam_list else 0

    def _fps_mousewheel(self, event):
        try:
            fps=float(self.fps_var.get())
        except ValueError:
            fps=10.0
        step=1 if event.delta>0 else -1
        fps=clamp(fps+step,1,120)
        self.fps_var.set(f"{fps:g}")
        self._apply_fps_live()

    def _apply_fps_live(self):
        """Push the FPS spinbox value to a running camera immediately, and
        persist it either way so the next run starts at the chosen rate."""
        try:
            fps=float(self.fps_var.get())
        except ValueError:
            return
        fps=clamp(fps,1,120)
        if f"{fps:g}"!=self.fps_var.get():
            self.fps_var.set(f"{fps:g}")
        cam=getattr(self,"_cam_obj",None)
        if cam is not None and getattr(cam,"is_open",lambda:False)():
            cam.set_fps(fps)
            self._status(f"Camera: target {fps:g} fps")
        self._save_settings()

    def _apply_cam_settings(self):
        """Push exposure & gain to the open camera.  DirectShow expects
        CAP_PROP_EXPOSURE in log2(seconds) (e.g. 15 ms ≈ −6); the raw ms value
        is also tried for drivers that take it directly."""
        cam=getattr(self,"_cam_obj",None)
        if cam is None or not getattr(cam,"is_open",lambda:False)():
            self._status("Camera settings will apply when the live view runs.")
            self._save_settings(); return
        try:
            ms=float(self.exp_var.get()); g=float(self.gain_var.get())
        except ValueError:
            messagebox.showerror("Camera","Exposure and gain must be numbers.")
            return
        try:
            # Controls are executed on the graph's own thread inside the
            # backend; COM objects are apartment-affine and must never be
            # touched from the tkinter main thread.
            notes=cam.set_controls(ms,g)
            # If the filter refused hardware gain, set_controls already fell
            # back to software gain; make sure it is live either way.
            if abs(getattr(cam,"digital_gain",1.0)-1.0)>1e-3:
                cam.set_digital_gain(g)
            joined=" · ".join(notes)
            if ("NOT supported" in joined or "not supported" in joined
                    or "could not reach" in joined):
                joined+=("   →  this filter ignores the standard controls; "
                         "use 'Driver settings…' on the camera tab instead")
            self._status("Camera: "+joined)
        except Exception as e:
            messagebox.showerror("Camera",str(e))
        self._save_settings()

    def _save_camera_image(self):
        frame=getattr(self,"_last_frame",None)
        if frame is None:
            messagebox.showinfo("Save image",
                "No frame captured yet — press Run on the CMOS Camera tab "
                "first."); return
        os.makedirs(RESULTS_DIR,exist_ok=True)
        fmt=self.img_fmt_var.get(); stamp=int(time.time())
        path=os.path.join(RESULTS_DIR,f"Camera_{stamp}.{fmt}")
        try:
            if fmt=="npy":
                np.save(path,frame)
            else:
                if not OPENCV_AVAILABLE:
                    raise RuntimeError("OpenCV required for image formats.")
                if not cv2.imwrite(path,frame):
                    raise RuntimeError(f"OpenCV could not write .{fmt}")
            self._save_settings()
            self._status(f"Image saved — {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save image",str(e))

    def _worker_camera(self, p):
        """Live camera, with the LC drive running alongside.

        Ordered so a failure in one stage cannot silently kill another:
          1. open the camera and PROVE a frame arrives before anything else
          2. apply exposure/gain  (non-fatal; some vendor filters stall on
             the standard DirectShow control interfaces)
          3. start the LC drive   (non-fatal; imaging still works without it)
          4. stream, with a watchdog that reports a frame stall instead of
             leaving a blank canvas
        """
        tab=self.tab_cam
        try:
            if not DSHOW_BACKEND:
                raise RuntimeError(
                    "The DirectShow camera backend is unavailable.\n"
                    f"{globals().get('_DSHOW_IMPORT_ERROR','')}\n\n"
                    "Install the capture stack with:\n"
                    "    pip install pygrabber comtypes")
            if not self._cam_list:
                raise RuntimeError(
                    "No readable DirectShow camera was found.\n\n"+
                    self._camera_diagnostic_text())
            idx=p["camera_index"]
            label=p["camera_label"]

            # ---- 1. open and confirm the stream actually delivers ----------
            self._cam_obj=CameraIntensity(
                idx,target_fps=p.get("camera_fps",10.0),
                flip_h=self.cam_fliph_var.get(),
                flip_v=self.cam_flipv_var.get(),
                rot_quarters=self.cam_rot_var.get()).open()
            self.after(0,lambda l=label:self._status(
                f"Camera opened ({l}) — waiting for the first frame…"))
            ok,first=self._cam_obj.read(timeout=15.0)
            if not ok or first is None:
                raise RuntimeError(
                    "The camera opened but delivered no frame within 15 s.\n\n"
                    "Most common causes:\n"
                    "  · another program holds the camera (close TCapture / "
                    "Mosaic / Teams / any browser tab using it)\n"
                    "  · the exposure is longer than the timeout\n"
                    "  · the USB link dropped — replug and press Run again")
            # Paint immediately so the view is never blank while we set up.
            self.after(0,lambda f=first.copy():self._draw_camera(tab,f,0.0))

            # ---- 2. exposure / gain — must not be able to kill the run -----
            try:
                notes=self._cam_obj.set_controls(
                    p["camera_exposure_ms"],p["camera_gain"])
            except Exception as exc:
                notes=[f"controls failed: {exc}"]
            # Some vendor filters stop streaming when poked through the
            # standard interfaces.  Verify frames still flow, and say so
            # plainly if they do not.
            ok2,_=self._cam_obj.read(timeout=5.0)
            if not ok2:
                notes.append("frames STOPPED after applying controls — "
                             "use 'Driver settings…' instead")

            # ---- 3. LC drive (independent of imaging) ----------------------
            dnote=self._drive_start(p)
            verified=_is_ish1000_name(label)
            head=(f"Camera live — {label} (index {idx}) — "
                  f"{'verified ISH1000' if verified else 'source not verified'}")
            self.after(0,lambda h=head,n=list(notes),d=dnote:self._status(
                h+" · "+d+(("  ·  "+" · ".join(n)) if n else "")))

            # ---- 4. stream ------------------------------------------------
            last=0.0
            previous=None
            saturation_reported=False
            stall_reported=False
            last_id=-1
            t_prev=None; meas_fps=None
            last_frame_t=time.monotonic()
            while self.running:
                ret,frame,last_id=self._cam_obj.read_new(last_id,timeout=1.5)
                if not ret or frame is None:
                    # Watchdog: never leave the user staring at a blank plot.
                    if (time.monotonic()-last_frame_t>3.0
                            and not stall_reported):
                        stall_reported=True
                        self.after(0,lambda:self._status(
                            "⚠ Camera delivered no frame for 3 s — the "
                            "source may be held by another program, or the "
                            "USB link dropped. Press Stop and Run to retry."))
                    time.sleep(0.02); continue
                last_frame_t=time.monotonic()
                if stall_reported:
                    stall_reported=False
                    self.after(0,lambda:self._status("Camera stream resumed."))
                t_now=time.monotonic()
                if t_prev is not None and t_now>t_prev:
                    inst=1.0/(t_now-t_prev)
                    meas_fps=(inst if meas_fps is None
                              else 0.8*meas_fps+0.2*inst)
                t_prev=t_now
                # frame is RGB uint8 (H, W, 3); a mono sensor simply gives
                # three identical channels.
                self._last_frame=frame
                # Statistics and motion delta are computed on a strided
                # SUBSAMPLE.  Converting a 10 MP frame to float32 twice per
                # frame was the dominant per-frame cost and capped the
                # achievable rate at a few Hz; 1/16 of the pixels gives the
                # same numbers to well within their own noise.
                sub=frame[::4,::4]
                subg=CameraIntensity._gray_of(sub)
                delta=0.0
                if previous is not None and previous.shape==subg.shape:
                    delta=float(np.mean(np.abs(subg-previous)))
                previous=subg

                if np.issubdtype(frame.dtype,np.integer):
                    full=float(np.iinfo(frame.dtype).max)
                else:
                    full=max(1.0,float(np.nanmax(sub)))
                clip=sub>=0.995*full
                if clip.ndim==3: clip=clip.any(axis=-1)
                clipped=float(clip.mean())
                saturated=clipped>0.001
                if saturated and not saturation_reported:
                    saturation_reported=True
                    self.after(0,lambda:self._status(
                        "Camera frame is saturated. Lower Exposure or Gain, "
                        "then press Apply."))
                # The display refresh follows the FPS setting.
                tgt=max(0.5,float(getattr(self._cam_obj,"target_fps",10.0)))
                interval=max(1.0/60.0,min(1.0,1.0/tgt))
                now=time.monotonic()
                if now-last>interval:
                    last=now
                    # Decimate BEFORE handing the frame to the Tk thread:
                    # marshalling a 30 MB array per refresh was the other
                    # half of the cost.
                    step=max(1,int(np.ceil(frame.shape[1]/1200.0)))
                    small=np.ascontiguousarray(frame[::step,::step])
                    stats={"shape":frame.shape,
                           "mean":float(subg.mean()),"std":float(subg.std()),
                           "lo":float(subg.min()),"hi":float(subg.max()),
                           "clipped":clipped,"full":full,
                           "reconnects":getattr(self._cam_obj,"reconnects",0),
                           "dgain":getattr(self._cam_obj,"digital_gain",1.0)}
                    self.after(0,lambda f=small,d=delta,mf=meas_fps,st=stats:
                               self._draw_camera(tab,f,d,mf,st))
                time.sleep(0.001)
        except Exception as e:
            self.after(0,lambda e=e:messagebox.showerror("Camera",str(e)))
        finally:
            if getattr(self,"_cam_obj",None) is not None:
                try: self._cam_obj.close()
                except Exception: pass
                self._cam_obj=None
            self._drive_stop()
            self._zero_all_outputs()
            self.running=False
            self.after(0,lambda:self.run_btn.config(state="normal"))
            self.after(0,lambda:self._status("Camera stopped — outputs = 0 V"))

    def _draw_camera(self, tab, frame, delta=0.0, meas_fps=None, stats=None):
        """Paint the live frame.  `frame` is already decimated and `stats`
        already computed by the worker, so this stays cheap on the Tk
        thread."""
        ax=tab["ax"]
        try:
            ax.clear(); ax.set_facecolor(PLOT_BG)
            if stats is None:                       # fallback path
                g=CameraIntensity._gray_of(frame)
                full=(float(np.iinfo(frame.dtype).max)
                      if np.issubdtype(frame.dtype,np.integer) else 255.0)
                clip=frame>=0.995*full
                if clip.ndim==3: clip=clip.any(axis=-1)
                stats={"shape":frame.shape,"mean":float(g.mean()),
                       "std":float(g.std()),"lo":float(g.min()),
                       "hi":float(g.max()),"clipped":float(clip.mean()),
                       "full":full,"reconnects":0,"dgain":1.0}
            full_shape=stats["shape"]
            saturated=stats["clipped"]>0.001
            if frame.ndim==3:
                ax.imshow(frame,aspect="equal")          # true RGB
            else:
                ax.imshow(frame,cmap="gray",aspect="equal",
                          vmin=0,vmax=stats["full"])
            # Force the view back to the image extent.  Without this, a stray
            # scroll-zoom or drag leaves the axes parked far from the image
            # and every later frame renders off-screen -- which looks exactly
            # like "the camera stopped working".
            h_,w_=frame.shape[:2]
            ax.set_xlim(-0.5,w_-0.5)
            ax.set_ylim(h_-0.5,-0.5)                    # origin at top-left
            dg=stats.get("dgain",1.0)
            rc=stats.get("reconnects",0)
            clip_txt=(f"  ·  CLIPPED {100*stats['clipped']:.2f}%"
                      if saturated else "")
            ax.set_title(
                f"CMOS Camera — live  ·  {full_shape[1]}x{full_shape[0]}"
                f"  ·  min/mean/max {stats['lo']:.0f}/{stats['mean']:.1f}/"
                f"{stats['hi']:.0f}"
                f"  ·  σ {stats['std']:.1f}  ·  Δ {delta:.2f}"
                f"{f'  ·  {meas_fps:.1f} fps' if meas_fps else ''}"
                f"{f'  ·  digital gain x{dg:g}' if abs(dg-1.0)>1e-3 else ''}"
                f"{f'  ·  reconnects {rc}' if rc else ''}"
                f"{clip_txt}",
                color=ACCENT_RED if saturated else ACCENT_CYAN,
                fontsize=11,loc="left")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color(BG_CARD)
        except Exception as exc:
            # Never leave a silently blank canvas: say what went wrong.
            try:
                ax.set_title(f"CMOS Camera — DRAW FAILED: {exc}",
                             color=ACCENT_RED,fontsize=10,loc="left")
            except Exception:
                pass
        tab["canvas"].draw_idle()

    def _scroll_zoom(self, event, canvas):
        ax=event.inaxes
        if ax is None or event.xdata is None: return
        f=0.8 if event.step>0 else 1.25
        x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
        ax.set_xlim(event.xdata-(event.xdata-x0)*f,event.xdata+(x1-event.xdata)*f)
        ax.set_ylim(event.ydata-(event.ydata-y0)*f,event.ydata+(y1-event.ydata)*f)
        canvas.draw_idle()

    def _on_tab_change(self, *_):
        """Each tab selects its own device automatically:
        Optical Response -> DAQ,  CMOS Camera -> camera,
        Spectrometer     -> Avantes AvaSpec.
        No manual device switching needed to run any tab."""
        if self.running:
            self.stop_run(); time.sleep(0.05)
        mode=self.current_mode()
        if mode=="Camera Live":
            want="CMOS Camera"
        elif mode=="Spectrometer":
            want="Spectrometer"
        else:
            want="DAQ"
        if want!=self.device_mode:
            self.device_mode=want
            if hasattr(self,"device_var"): self.device_var.set(want)
            self.title(f"LC Characterization · {want}")
        self._status(f"Mode: {mode} · device {want} — press Run")

    def current_mode(self):
        return ["Continuous","Camera Live","Spectrometer"][
            self.nb.index("current")]

    def current_tab(self):
        return [self.tab_cont,self.tab_cam,self.tab_spec][
            self.nb.index("current")]

    # ── channel management ─────────────────────────────────────────────────
    def _restore_channels(self):
        saved=self.settings.get("channels")
        if saved:
            for c in saved[:self.n_ao]:
                line=c.get("line","ao0")
                if line not in self.ao_lines: line=self.ao_lines[0]
                self._append_channel(c.get("name","CH1"),line,
                                     float(c.get("volts",1.0)))
        if not self.channel_rows:
            self._append_channel("CH1",self.ao_lines[0] if self.ao_lines
                                 else "ao0",1.0)

    def _append_channel(self, name, line, volts):
        row=ChannelRow(self.chan_frame,self,name,line,volts)
        row.pack(fill="x",pady=1)
        self.channel_rows.append(row)

    def add_channel(self):
        if len(self.channel_rows)>=self.n_ao:
            messagebox.showinfo("Channels",
                f"This DAQ has {self.n_ao} analog-output line"
                f"{'s' if self.n_ao!=1 else ''} (ao0–ao{self.n_ao-1}).\n"
                "All of them are already assigned — no more channels can be "
                "added.")
            return
        used_lines={r.line() for r in self.channel_rows}
        free=[l for l in self.ao_lines if l not in used_lines]
        used_names={r.name().lower() for r in self.channel_rows}
        i=1
        while f"ch{i}" in used_names: i+=1
        self._append_channel(f"CH{i}",free[0],1.0)
        self.refresh_channel_ui()

    def remove_channel(self, row):
        if len(self.channel_rows)<=1:
            messagebox.showinfo("Channels","At least one channel is required.")
            return
        row.destroy(); self.channel_rows.remove(row)
        self.refresh_channel_ui()

    def on_line_change(self, row):
        """Line exclusivity by SWAP: choosing a line that another channel holds
        gives that channel this row's previous line; if that line is somehow
        unusable, the displaced channel takes the next free ao line."""
        new=row.line()
        for r in self.channel_rows:
            if r is not row and r.line()==new:
                old=row._prev_line
                taken={x.line() for x in self.channel_rows if x is not r}
                if old in self.ao_lines and old not in taken:
                    r.line_var.set(old)
                else:
                    free=[l for l in self.ao_lines if l not in taken]
                    if free: r.line_var.set(free[0])
                r._prev_line=r.line()
                break
        row._prev_line=new
        self.refresh_channel_ui()

    def refresh_channel_ui(self):
        """Rebuild dropdowns (full ao list — conflicts resolve by swapping),
        keep the drive selector valid."""
        for r in self.channel_rows:
            r.line_menu["values"]=self.ao_lines
            r.del_btn.config(state="normal" if len(self.channel_rows)>1
                             else "disabled")
        names=[r.name() for r in self.channel_rows]
        self.drive_menu["values"]=names
        if self.drive_var.get() not in names:
            self.drive_var.set(names[0] if names else "")
        n=len(self.channel_rows)
        self.chan_hint.set(f"{n}/{self.n_ao} channels · lines are exclusive · "
                           "names must be unique")
        self.add_btn.config(state="normal" if n<self.n_ao else "disabled")
        self._save_settings()

    def drive_row(self):
        want=self.drive_var.get()
        for r in self.channel_rows:
            if r.name()==want:
                return r
        return self.channel_rows[0] if self.channel_rows else None

    # ── settings ───────────────────────────────────────────────────────────
    def _load_settings(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f: return json.load(f)
            except Exception: pass
        return {}

    def _save_settings(self):
        try:
            s={"freq":self.freq_var.get(),"wave":self.wave_var.get(),
               "drive":self.drive_var.get(),
               "dark_offset_v":float(self.dark_offset_v),
               "channels":[{"name":r.name(),"line":r.line(),
                            "volts":r.volts()} for r in self.channel_rows]}
            tc=getattr(self,"tab_cont",None)
            if tc and tc.get("ai_var"): s["ai_optical"]=tc["ai_var"].get()
            for attr,key in (("save_n_var","save_n"),("save_fmt_var","save_fmt"),
                             ("opt_ymin_var","opt_y_min"),
                             ("opt_ymax_var","opt_y_max"),
                             ("opt_yfixed_var","opt_y_fixed"),
                             ("exp_var","exposure_ms"),("gain_var","gain"),
                             ("fps_var","camera_fps"),
                             ("cam_rot_var","cam_rot"),
                             ("cam_fliph_var","cam_flip_h"),
                             ("cam_flipv_var","cam_flip_v"),
                             ("img_fmt_var","img_fmt"),
                             ("cam_sel_var","camera_sel"),
                             ("integ_var","spec_integration_ms"),
                             ("avg_var","spec_averages"),
                             ("spec_mode_var","spec_mode"),
                             ("x_min_var","spec_x_min"),
                             ("x_max_var","spec_x_max"),
                             ("y_min_var","spec_y_min"),
                             ("y_max_var","spec_y_max"),
                             ("spec_xauto_var","spec_x_auto"),
                             ("spec_yauto_var","spec_y_auto")):
                v=getattr(self,attr,None)
                if v is not None: s[key]=v.get()
            with open(CONFIG_FILE,"w") as f: json.dump(s,f,indent=2)
        except Exception:
            pass

    # ── validated inputs ───────────────────────────────────────────────────
    def _params(self):
        try:
            f=float(self.freq_var.get())
        except ValueError:
            messagebox.showerror("Input","Frequency must be a number."); return None
        fc=clamp(f,FREQ_MIN,FREQ_MAX)
        if fc!=f: self.freq_var.set(f"{fc:g}")
        d=self.drive_row()
        if d is None:
            messagebox.showerror("Channels","No drive channel."); return None
        return {"freq":fc,"wave":self.wave_var.get(),
                "drive_chan":d.phys(),"v_peak":d.volts()/2.0,   # entry is Vpp
                "drive_name":d.name()}

    def _ai_phys(self, tab=None):
        """All DAQ reads use the AI channel chosen on the Optical Response tab
        (the selector is exclusive to that tab by design)."""
        return f"{AI_DEVICE}/{self.tab_cont['ai_var'].get()}"

    # ── RUN / STOP ─────────────────────────────────────────────────────────
    def start_run(self):
        if self.running: return
        p=self._params()
        if p is None: return
        mode=self.current_mode()
        if mode=="Camera Live":
            try:
                p["camera_exposure_ms"]=float(self.exp_var.get())
                p["camera_gain"]=float(self.gain_var.get())
                p["camera_fps"]=clamp(float(self.fps_var.get()),1,120)
                if p["camera_exposure_ms"]<=0:
                    raise ValueError("Exposure must be greater than 0 ms.")
            except ValueError as exc:
                messagebox.showerror("Camera",
                    str(exc) if str(exc) else
                    "Exposure and gain must be valid numbers.")
                return
            # Run discovery on the Tk thread. Updating widgets or Tk variables
            # from the worker thread can freeze or corrupt the interface.
            self._refresh_cameras(quiet=True)
            if not self._cam_list:
                messagebox.showerror("Camera",
                    "No readable camera source was found.\n\n"+
                    self._camera_diagnostic_text())
                return
            p["camera_index"]=self._selected_cam_index()
            p["camera_label"]=dict(self._cam_list).get(
                p["camera_index"],f"index {p['camera_index']}")
        elif mode=="Spectrometer":
            try:
                ms,avg=self._spec_settings()
            except ValueError as exc:
                messagebox.showerror("Spectrometer",str(exc)); return
            p["spec_integration_ms"]=ms
            p["spec_averages"]=avg
            p["spec_mode"]=self.spec_mode_var.get()
        self._save_settings()
        self.running=True; self.run_btn.config(state="disabled")
        target={"Continuous":self._worker_continuous,
                "Camera Live":self._worker_camera,
                "Spectrometer":self._worker_spectrum}[mode]
        self.worker=threading.Thread(target=target,args=(p,),daemon=True)
        self.worker.start()

    def stop_run(self):
        self.running=False
        self.run_btn.config(state="normal")

    # ── CONTINUOUS STREAMING ───────────────────────────────────────────────
    def _worker_continuous(self, p):
        tab=self.tab_cont
        try:
            if not NIDAQMX_AVAILABLE:
                raise RuntimeError(
                    "NI-DAQmx is not available, so the photodiode cannot be "
                    "read.\n\nInstall the NI-DAQmx driver and the Python "
                    "package:\n    pip install nidaqmx\n\n"
                    "Simulation has been removed deliberately: showing "
                    "synthetic traces here would hide a real hardware fault.")
            self._stream_daq(p,tab)
        except Exception as e:
            self.after(0,lambda e=e:messagebox.showerror("Error",str(e)))
        finally:
            self.running=False
            self._zero_all_outputs()
            self.after(0,lambda:self.run_btn.config(state="normal"))
            self.after(0,lambda:self._status("Stopped — outputs = 0 V"))

    def _carrier_buffer(self, p, seconds=0.1):
        """(buffer, ao_rate) with EXACTLY zero DC and a seamless loop.

        The AO rate is chosen so one carrier period is an EVEN integer number
        of samples (equal + and - half-cycles -> DC = 0 exactly, which the
        project requires to avoid electrochemical damage to the LC).  The
        requested frequency is then produced EXACTLY at that rate."""
        f=float(p["freq"])
        spc_min=2 if p["wave"]=="Square" else 4      # sine needs >=4 pts/cycle
        spc=int(round(SAMPLE_RATE/f/2.0))*2          # even samples per cycle
        while spc>spc_min and f*spc>SAMPLE_RATE:     # never exceed 9264 max rate
            spc-=2
        spc=max(spc_min,spc)
        if f*spc>SAMPLE_RATE:
            # hardware ceiling: snap the frequency down to what 25 kS/s allows
            f=SAMPLE_RATE/spc
            self._freq_note=(f"frequency snapped to {f:.0f} Hz "
                             f"(9264 limit: {spc} samples/cycle @ 25 kS/s)")
        else:
            self._freq_note=""
        ao_rate=f*spc
        periods=max(1,int(round(seconds*f)))
        n=spc*periods
        if p["wave"]=="Square":
            half=spc//2
            cycle=np.concatenate([np.full(half, p["v_peak"]),
                                  np.full(half,-p["v_peak"])])
            return np.tile(cycle,periods), ao_rate
        k=np.arange(n)
        return p["v_peak"]*np.sin(2*np.pi*k/spc), ao_rate

    # ── standalone LC drive (camera / spectrometer tabs) ──────────────────
    def _drive_plan(self, p):
        """Work out what to drive.

        Returns (chans, names, vpps, bufs, ao_rate).  Every channel with a
        non-zero voltage is driven at its OWN amplitude, sharing one sample
        clock so all cells stay phase-locked.  A unit carrier is built once
        and scaled per channel, which keeps the exact-zero-DC property of
        _carrier_buffer for every channel rather than re-deriving it.
        """
        unit, ao_rate = self._carrier_buffer(dict(p, v_peak=1.0))
        chans, names, vpps = [], [], []
        for r in self.channel_rows:
            vpp = r.volts()
            if abs(vpp) < 1e-9:
                continue
            chans.append(r.phys()); names.append(r.name()); vpps.append(vpp)
        if not chans:
            return [], [], [], None, ao_rate
        bufs = np.vstack([(v / 2.0) * unit for v in vpps])
        return chans, names, vpps, bufs, ao_rate

    def _drive_start(self, p):
        """Start the cell drive for a tab whose worker owns no AO task.
        Returns a human-readable note; never raises."""
        self._drive_stop()
        if not (NIDAQMX_AVAILABLE and self.has_ao):
            return "cell NOT driven (no AO hardware detected)"
        try:
            chans, names, vpps, bufs, ao_rate = self._drive_plan(p)
            if not chans:
                return ("cell NOT driven (every channel voltage is 0 V — "
                        "set a voltage in the CHANNELS panel)")
            self._drive = DriveOutput().start(chans, bufs, ao_rate)
            return ("driving " +
                    ", ".join(f"{n}={v:g} Vpp" for n, v in zip(names, vpps)) +
                    f" @ {p['freq']:g} Hz {p['wave'].lower()}")
        except Exception as exc:
            self._drive = None
            return f"drive FAILED: {exc}"

    def _drive_stop(self):
        d = getattr(self, "_drive", None)
        if d is not None:
            try:
                d.stop()
            except Exception:
                pass
        self._drive = None

    def _stream_daq(self, p, tab):
        ai_chan=self._ai_phys(tab)
        buf,ao_rate=self._carrier_buffer(p)
        window_s=0.05                       # rolling display window
        maxlen=int(window_s*AI_RATE)
        data=deque(maxlen=maxlen)
        self._save_buf=deque(maxlen=int(5.0*AI_RATE))   # last 5 s kept for saving
        self._save_rate=AI_RATE
        tc=ai_terminal_config()
        with nidaqmx.Task() as ao, nidaqmx.Task() as ai:
            if self.has_ao:
                ao.ao_channels.add_ao_voltage_chan(p["drive_chan"],
                                                   min_val=AO_VMIN,max_val=AO_VMAX)
                ao.timing.cfg_samp_clk_timing(ao_rate,
                        sample_mode=AcquisitionType.CONTINUOUS,
                        samps_per_chan=buf.size)
                ao.write(buf.tolist(),auto_start=False)
            ai.ai_channels.add_ai_voltage_chan(ai_chan,terminal_config=tc,
                                               min_val=AI_VMIN,max_val=AI_VMAX)
            ai.timing.cfg_samp_clk_timing(AI_RATE,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=int(AI_RATE))
            ai.start()
            if self.has_ao: ao.start()
            fn=getattr(self,"_freq_note","")
            self.after(0,lambda:self._status(
                f"Streaming · drive {p['drive_name']} ({p['drive_chan']}) "
                f"@ {p['v_peak']*2:g} Vpp {p['freq']:g} Hz · read {ai_chan} — "
                "press Stop to end"+(f"  ·  ⚠ {fn}" if fn else "")))
            last=0.0; total=0
            while self.running:
                avail=int(ai.in_stream.avail_samp_per_chan)
                if avail>0:
                    chunk=np.asarray(ai.read(
                        number_of_samples_per_channel=avail,timeout=2.0),
                        dtype=float)
                    vals=(chunk-self.dark_offset_v).tolist()
                    data.extend(vals); self._save_buf.extend(vals)
                    total+=chunk.size
                now=time.monotonic()
                if now-last>0.08 and len(data)>10:
                    last=now
                    y=np.asarray(data,dtype=float)
                    t=np.arange(y.size)/AI_RATE        # SECONDS
                    note=self._pd_note(y)
                    self.after(0,lambda t=t,y=y,n=note:
                               self._draw_stream(tab,t,y,n))
                time.sleep(0.005)
            try: ai.stop()
            except Exception: pass
            if self.has_ao:
                try: ao.stop()
                except Exception: pass

    def _stream_sim(self, p, tab):
        data=deque(maxlen=int(0.05*AI_RATE))
        self._save_buf=deque(maxlen=int(5.0*AI_RATE)); self._save_rate=AI_RATE
        v_rms=p["v_peak"] if p["wave"]=="Square" else p["v_peak"]/np.sqrt(2)
        base=LCModel.transmission(v_rms)
        self.after(0,lambda:self._status(
            f"Streaming (simulation) · {p['drive_name']} — press Stop"))
        last=0.0
        while self.running:
            n=int(0.01*AI_RATE)
            vals=(base+np.random.normal(0,0.01,n)).tolist()
            data.extend(vals); self._save_buf.extend(vals)
            now=time.monotonic()
            if now-last>0.08:
                last=now
                y=np.asarray(data,dtype=float)
                t=np.arange(y.size)/AI_RATE            # SECONDS
                self.after(0,lambda t=t,y=y:self._draw_stream(tab,t,y,""))
            time.sleep(0.01)

    @staticmethod
    def _pd_note(y):
        y=y[np.isfinite(y)]
        if y.size>=10 and float(np.std(y))<1e-3 and abs(float(np.mean(y)))>9.0:
            return "INPUT FLOATING (nothing connected)"
        return ""

    def _draw_stream(self, tab, t, y, note):
        ax=tab["ax"]; ax.clear(); ax.set_facecolor(PLOT_BG)
        ax.plot(t,y,color=ACCENT_CYAN,linewidth=0.9)
        ax.set_xlabel("Time (s)",color=TEXT_GRAY,fontsize=8)
        ax.set_ylabel("Optical (V)",color=TEXT_GRAY,fontsize=8)
        title="Optical Response — live"
        if note: title+=f"   ⚠ {note}"
        ax.set_title(title,color=ACCENT_RED if note else ACCENT_CYAN,
                     fontsize=11,loc="left")
        ax.tick_params(colors=TEXT_GRAY,labelsize=7)
        for s in ax.spines.values(): s.set_color(BG_CARD)
        ax.grid(color=BG_CARD,linewidth=0.4,alpha=0.6)
        # Fixed Y scale (default 0–10 V).  Blank/invalid boxes fall back to
        # matplotlib\'s autoscale; unchecking "Fixed" also autoscales.
        if getattr(self,"opt_yfixed_var",None) is not None \
           and self.opt_yfixed_var.get():
            try:
                y0=float(self.opt_ymin_var.get())
                y1=float(self.opt_ymax_var.get())
                if y1>y0:
                    ax.set_ylim(y0,y1)
            except (ValueError,AttributeError):
                pass
        tab["canvas"].draw_idle()

    # ── SPECTROMETER: calibration captures, live T(λ), saving ──────────────
    # ── SPECTROMETER ───────────────────────────────────────────────────────
    SPEC_MODES = ("Scope (raw counts)",
                  "Scope − Dark",
                  "Transmission %",
                  "Absorbance")

    def _spec_needs(self, mode):
        """(needs_dark, needs_reference) for a display mode."""
        if mode.startswith("Scope (raw"):
            return (False, False)
        if mode.startswith("Scope −"):
            return (True, False)
        return (True, True)          # Transmission %, Absorbance

    def _spec_open(self):
        """Open the Avantes device, or raise with a plain explanation."""
        if self.spec is None or not self.spec.is_open():
            self.spec = Spectrometer().open()
            self.spec_wl = self.spec.wavelengths()
            self._status(f"Spectrometer: {self.spec.label()}")
        return self.spec

    def _spec_settings(self):
        """Validated (integration_ms, averages) from the tab controls."""
        try:
            ms = float(self.integ_var.get())
        except ValueError:
            raise ValueError("Integration time must be a number (ms).")
        if ms <= 0:
            raise ValueError("Integration time must be greater than 0 ms.")
        try:
            avg = int(float(self.avg_var.get()))
        except ValueError:
            raise ValueError("Averaging must be a whole number.")
        if avg < 1:
            raise ValueError("Averaging must be at least 1.")
        return ms, avg

    def _spec_grab_single(self):
        """One spectrum for Dark / Reference capture."""
        sp = self._spec_open()
        ms, avg = self._spec_settings()
        return np.asarray(sp.single(ms, avg), dtype=float)

    def _capture_spec(self, which):
        """Store DARK (light blocked) or REFERENCE (light on, no sample).

        Transmission is then  T(λ) = (S − Dark) / (Ref − Dark).
        Works BOTH ways:
          · while running:  the next scan from the live stream is stored,
            without interrupting acquisition (same integration/averaging
            as the run, which is exactly what a calibration should use);
          · while stopped:  a single dedicated acquisition is made.
        """
        if self.running and self.current_mode() == "Spectrometer":
            self._spec_capture_req = which
            self._status(f"Capturing {which.upper()} from the next scan — "
                         "keep the setup as it is…")
            return
        if self.running:
            messagebox.showinfo("Spectrometer",
                                "Stop the current (non-spectrometer) run "
                                "before capturing.")
            return
        try:
            y = self._spec_grab_single()
        except Exception as e:
            messagebox.showerror("Spectrometer", str(e))
            return
        self._store_spec_capture(which, y)

    def _store_spec_capture(self, which, y):
        """Validate and store a captured spectrum (main thread only)."""
        name = which.upper()
        if which == "ref" and self.spec_dark is not None and \
           float(np.median(y - self.spec_dark)) <= 0:
            messagebox.showerror("Capture FAILED",
                "REFERENCE capture FAILED.\n\n"
                "The reference is not brighter than the dark spectrum. "
                "Capture it with the light ON and no sample in the beam.")
            return
        if which == "dark":
            self.spec_dark = y
            self._status("Dark captured — the light must have been blocked.")
        else:
            self.spec_ref = y
            self._status("Reference captured — full brightness, no sample.")

        # Warn on saturation: a clipped reference silently caps every later
        # transmission value, and the curve still looks plausible.  During a
        # live run this must NOT be a modal dialog (it steals focus mid-
        # stream); the status line carries the warning instead.  Either way,
        # suggest a concrete integration time computed from the scan itself.
        try:
            fs = float(getattr(self.spec, "max_counts", 65535))
            peak = float(np.max(y))
            if peak >= 0.98 * fs:
                try:
                    cur_ms, _ = self._spec_settings()
                    # Aim the peak at 80 % of full scale.  The true peak is
                    # unknown (it is clipped), so this is a lower bound —
                    # repeat once if the next capture still clips.
                    sug = f"{max(0.05, cur_ms * 0.8 * fs / peak):.3g} ms"
                except Exception:
                    sug = "a shorter time"
                msg = (f"{which.upper()} is SATURATED "
                       f"({peak:.0f}/{fs:.0f} counts). "
                       f"Set Integration ≈ {sug} and capture again.")
                if self.running:
                    self._status("⚠ " + msg)
                else:
                    messagebox.showwarning("Saturation",
                        msg + "\n\nA clipped spectrum caps every "
                        "transmission value derived from it, and the "
                        "resulting curve still looks plausible — which is "
                        "exactly why it must be recaptured.")
            else:
                messagebox.showinfo("Capture successful",
                    f"{name} captured successfully.\n\n"
                    f"Peak {peak:.0f} of {fs:.0f} counts "
                    f"({100.0 * peak / fs:.0f} % of full scale).")
        except Exception:
            pass

        self._update_spec_status()
        if not self.running:
            self._draw_spectrum_calib()

    def _update_spec_status(self):
        self.spec_status.set(
            f"Dark: {'OK' if self.spec_dark is not None else '—'}   "
            f"Ref: {'OK' if self.spec_ref is not None else '—'}")

    def _clear_spec_calib(self):
        self.spec_dark = None
        self.spec_ref = None
        self._update_spec_status()
        self._status("Dark and Reference cleared.")

    def _spec_transmittance(self, sample):
        """T(λ) as a PERCENTAGE, NaN where the lamp gave no usable signal."""
        d, r = self.spec_dark, self.spec_ref
        denom = r - d
        floor = max(1e-9, 0.02 * float(np.nanmax(denom)))
        valid = denom > floor
        t = np.full_like(sample, np.nan)
        t[valid] = 100.0 * (sample[valid] - d[valid]) / denom[valid]
        return t

    def _spec_reduce(self, sample, mode):
        """Turn a raw scan into the selected display quantity."""
        if mode.startswith("Scope (raw"):
            return sample, "Counts"
        if mode.startswith("Scope −"):
            return sample - self.spec_dark, "Counts − Dark"
        t = self._spec_transmittance(sample)
        if mode.startswith("Transmission"):
            return t, "Transmittance [%]"
        with np.errstate(divide="ignore", invalid="ignore"):
            a = -np.log10(np.clip(t / 100.0, 1e-12, None))
        a[~np.isfinite(a)] = np.nan
        return a, "Absorbance [OD]"

    def _worker_spectrum(self, p):
        """Continuous acquisition until Stop, in the selected mode."""
        tab = self.tab_spec
        sp = None
        try:
            mode = p["spec_mode"]
            sp = self._spec_open()
            sp.configure(p["spec_integration_ms"], p["spec_averages"])
            sp.start_continuous()
            # Drive the LC cell: the spectrometer worker owns no AO task of
            # its own, so without this the cell sits at 0 V while measuring.
            dnote = self._drive_start(p)
            self.after(0, lambda m=mode, lbl=sp.label(), d=dnote: self._status(
                f"{m} live · {lbl} · {d} — press Stop to end"))

            wl = self.spec_wl
            n = 0
            last_eff = None
            timeout = max(10.0, 4.0 * p["spec_integration_ms"] *
                          p["spec_averages"] / 1000.0 + 5.0)
            last_draw = 0.0
            while self.running:
                sample = sp.read_frame(timeout=timeout)
                self._last_spec_sample = sample
                n += 1
                req = self._spec_capture_req
                if req:
                    self._spec_capture_req = None
                    self.after(0, lambda w=req, y=sample.copy():
                               self._store_spec_capture(w, y))
                # If the selected mode still lacks its Dark/Reference, fall
                # back to the richest mode that IS available and say what is
                # missing -- so Run always works and the calibration can be
                # captured live, mid-stream, with the run's own settings.
                need_d, need_r = self._spec_needs(mode)
                missing = []
                if need_d and self.spec_dark is None:
                    missing.append("DARK")
                if need_r and self.spec_ref is None:
                    missing.append("REFERENCE")
                if missing:
                    eff = ("Scope − Dark" if self.spec_dark is not None
                           else "Scope (raw counts)")
                    hint = " + ".join(missing)
                else:
                    eff, hint = mode, ""
                if eff != last_eff:
                    last_eff = eff
                    msg = (f"{mode} live · waiting for {hint} — press the "
                           f"Capture button(s); showing {eff} meanwhile"
                           if hint else f"{mode} live — press Stop to end")
                    self.after(0, lambda m=msg: self._status(m))
                y, ylabel = self._spec_reduce(sample, eff)
                # Throttle redraws: the device can stream faster than
                # matplotlib can repaint, and queueing every scan through
                # after() would grow an unbounded backlog on the Tk thread.
                now = time.monotonic()
                if now - last_draw > 0.12:
                    last_draw = now
                    self.after(0, lambda wl=wl, y=y.copy(), lab=ylabel,
                               k=n, m=eff:
                               self._draw_spectrum(tab, wl, y, lab, k, m))
        except Exception as e:
            self.after(0, lambda e=e:
                       messagebox.showerror("Spectrometer", str(e)))
        finally:
            if sp is not None:
                try:
                    sp.stop()
                except Exception:
                    pass
            self._drive_stop()
            self._zero_all_outputs()
            self.running = False
            self.after(0, lambda: self.run_btn.config(state="normal"))
            self.after(0, lambda:
                       self._status("Spectrometer stopped — outputs = 0 V"))

    # ── axis scale control ────────────────────────────────────────────────
    def _spec_apply_scale(self, ax=None, redraw=True):
        """Apply the X/Y limit boxes. Blank or invalid entries mean auto."""
        tab = self.tab_spec
        ax = ax if ax is not None else tab["ax"]

        def lim(lo_var, hi_var):
            try:
                lo = float(lo_var.get())
            except (ValueError, AttributeError):
                lo = None
            try:
                hi = float(hi_var.get())
            except (ValueError, AttributeError):
                hi = None
            if lo is not None and hi is not None and lo >= hi:
                return None, None          # nonsense range -> ignore
            return lo, hi

        if not self.spec_xauto_var.get():
            x0, x1 = lim(self.x_min_var, self.x_max_var)
            if x0 is not None or x1 is not None:
                cur = ax.get_xlim()
                ax.set_xlim(x0 if x0 is not None else cur[0],
                            x1 if x1 is not None else cur[1])
        if not self.spec_yauto_var.get():
            y0, y1 = lim(self.y_min_var, self.y_max_var)
            if y0 is not None or y1 is not None:
                cur = ax.get_ylim()
                ax.set_ylim(y0 if y0 is not None else cur[0],
                            y1 if y1 is not None else cur[1])
        if redraw:
            tab["canvas"].draw_idle()
        self._save_settings()

    def _spec_autoscale_now(self):
        """Read the current view back into the boxes, then switch to manual."""
        ax = self.tab_spec["ax"]
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        self.x_min_var.set(f"{x0:.6g}"); self.x_max_var.set(f"{x1:.6g}")
        self.y_min_var.set(f"{y0:.6g}"); self.y_max_var.set(f"{y1:.6g}")
        self.spec_xauto_var.set(False); self.spec_yauto_var.set(False)
        self._status("Axis limits captured from the current view.")
        self._save_settings()

    def _spec_reset_scale(self):
        self.spec_xauto_var.set(True); self.spec_yauto_var.set(True)
        self._status("Axes back to auto-scale.")
        self._save_settings()

    # ── drawing ───────────────────────────────────────────────────────────
    def _draw_spectrum(self, tab, wl, y, ylabel="", n=0, mode=None):
        ax = tab["ax"]; ax.clear(); ax.set_facecolor(PLOT_BG)
        ax.plot(wl, y, color=ACCENT_CYAN, linewidth=1.0)
        if mode is None:
            mode = self.spec_mode_var.get()
        # Calibration badge: shows at a glance that DARK / REFERENCE have
        # been captured during this run (green once both are in).
        d_ok = self.spec_dark is not None
        r_ok = self.spec_ref is not None
        ax.text(0.99, 0.98,
                f"Dark {'✓' if d_ok else '—'}   Ref {'✓' if r_ok else '—'}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=ACCENT_GREEN if (d_ok and r_ok) else ACCENT_YELLOW,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_CARD,
                          edgecolor=BG_CARD, alpha=0.9))
        if mode.startswith("Transmission"):
            ax.axhline(0.0, color=BG_CARD, linewidth=0.8)
            ax.axhline(100.0, color=ACCENT_ORANGE, linewidth=0.8,
                       linestyle="--", alpha=0.7)
        elif mode.startswith("Scope −"):
            ax.axhline(0.0, color=BG_CARD, linewidth=0.8)
        ax.set_xlabel("Wavelength [nm]", color=TEXT_GRAY, fontsize=8)
        ax.set_ylabel(ylabel or mode, color=TEXT_GRAY, fontsize=8)
        try:
            finite = y[np.isfinite(y)]
            extra = (f"  ·  min/max {finite.min():.3g}/{finite.max():.3g}"
                     if finite.size else "")
        except Exception:
            extra = ""
        ax.set_title(f"{mode}  ·  scan {n}{extra}",
                     color=ACCENT_CYAN, fontsize=10, loc="left")
        ax.grid(True, color=BG_CARD, linewidth=0.5, alpha=0.6)
        ax.tick_params(colors=TEXT_GRAY, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(BG_CARD)
        if self.spec_yauto_var.get():
            # Open with a generous window instead of zooming onto noise:
            # autoscale first, then guarantee the view spans at least
            # -500..+500 counts (or %), growing further if the data does.
            y0, y1 = ax.get_ylim()
            ax.set_ylim(min(y0, -500.0), max(y1, 500.0))
        self._spec_apply_scale(ax=ax, redraw=False)
        tab["canvas"].draw_idle()

    def _draw_spectrum_calib(self):
        """Show the stored Dark and Reference after a capture."""
        if self.spec_wl is None:
            return
        tab = self.tab_spec
        ax = tab["ax"]; ax.clear(); ax.set_facecolor(PLOT_BG)
        wl = self.spec_wl
        if self.spec_dark is not None:
            ax.plot(wl, self.spec_dark, color=TEXT_GRAY, linewidth=1.0,
                    label="Dark")
        if self.spec_ref is not None:
            ax.plot(wl, self.spec_ref, color=ACCENT_YELLOW, linewidth=1.0,
                    label="Reference")
        if self.spec_dark is not None and self.spec_ref is not None:
            ax.fill_between(wl, self.spec_dark, self.spec_ref,
                            color=ACCENT_CYAN, alpha=0.10,
                            label="usable dynamic range")
        ax.set_xlabel("Wavelength [nm]", color=TEXT_GRAY, fontsize=8)
        ax.set_ylabel("Counts", color=TEXT_GRAY, fontsize=8)
        ax.set_title("Calibration spectra — press Run to measure",
                     color=ACCENT_YELLOW, fontsize=10, loc="left")
        ax.grid(True, color=BG_CARD, linewidth=0.5, alpha=0.6)
        ax.tick_params(colors=TEXT_GRAY, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(BG_CARD)
        leg = ax.legend(facecolor=BG_CARD, edgecolor=BG_CARD, fontsize=7)
        for t in leg.get_texts():
            t.set_color(TEXT_WHITE)
        self._spec_apply_scale(ax=ax, redraw=False)
        tab["canvas"].draw_idle()

    # ── saving ────────────────────────────────────────────────────────────
    def _save_spectrum(self):
        """Save wavelength, dark, reference, raw scan and the derived curve."""
        if self.spec_wl is None or \
           getattr(self, "_last_spec_sample", None) is None:
            messagebox.showinfo("Save spectrum",
                "No spectrum yet — press Run (or capture a Dark/Reference) "
                "first.")
            return
        os.makedirs(RESULTS_DIR, exist_ok=True)
        wl = self.spec_wl
        s_ = self._last_spec_sample
        nan = np.full_like(wl, np.nan)
        d = self.spec_dark if self.spec_dark is not None else nan
        r = self.spec_ref if self.spec_ref is not None else nan
        mode = self.spec_mode_var.get()
        try:
            y, ylabel = self._spec_reduce(s_, mode)
        except Exception:
            y, ylabel = nan, mode

        try:
            ms, avg = self._spec_settings()
        except Exception:
            ms, avg = float("nan"), 0

        stamp = int(time.time())
        path = os.path.join(RESULTS_DIR, f"Spectrum_{stamp}.csv")
        hdr = (f"Avantes spectrum export\n"
               f"instrument,{self.spec.label() if self.spec else 'unknown'}\n"
               f"mode,{mode}\n"
               f"integration_ms,{ms:g}\n"
               f"averages,{avg}\n"
               f"columns,wavelength_nm|dark_counts|reference_counts|"
               f"scope_counts|{ylabel}\n"
               f"wavelength_nm,dark,reference,scope,value")
        np.savetxt(path, np.column_stack((wl, d, r, s_, y)),
                   delimiter=",", header=hdr, comments="")
        self._status(f"Spectrum saved — {os.path.basename(path)}")

    # ── DARK ───────────────────────────────────────────────────────────────
    def _measure_dark(self):
        if self.running:
            messagebox.showinfo("Dark","Stop the current run first."); return
        tab=self.current_tab()
        if not NIDAQMX_AVAILABLE:
            messagebox.showerror("Dark measurement",
                "NI-DAQmx is not available, so a dark offset cannot be "
                "measured.\n\nA fabricated 0.000000 V offset would silently "
                "bias every later reading, so none is set.")
            return
        try:
            n=max(10,int(0.5*AI_RATE))
            with nidaqmx.Task() as t:
                t.ai_channels.add_ai_voltage_chan(self._ai_phys(tab),
                        terminal_config=ai_terminal_config(),
                        min_val=AI_VMIN,max_val=AI_VMAX)
                t.timing.cfg_samp_clk_timing(AI_RATE,
                        sample_mode=AcquisitionType.FINITE,samps_per_chan=n)
                d=np.asarray(t.read(n,timeout=5.0),dtype=float)
            d=d[np.isfinite(d)]
            if d.size<3: raise ValueError("Not enough dark samples acquired.")
            if float(np.std(d))<1e-3 and abs(float(np.mean(d)))>9.0:
                raise ValueError(
                    f"{self._ai_phys(tab)} is railed at {float(np.mean(d)):.2f} V "
                    "with no noise — the input is floating.\n\nConnect the "
                    "photodiode before measuring dark.")
            self.dark_offset_v=float(np.mean(d))
            self.dark_var.set(f"Dark offset: {self.dark_offset_v:.6f} V  "
                              f"STD {float(np.std(d)):.2e}  N={d.size}")
            self._save_settings()
            self._status("Dark measured — do not move the setup.")
        except Exception as e:
            messagebox.showerror("Dark measurement",str(e))

    # ── SAFETY ─────────────────────────────────────────────────────────────
    def _zero_all_outputs(self):
        if not (NIDAQMX_AVAILABLE and self.has_ao): return
        try:
            chans=sorted({r.phys() for r in self.channel_rows})
            with nidaqmx.Task() as t:
                for c in chans:
                    t.ao_channels.add_ao_voltage_chan(c,min_val=AO_VMIN,
                                                      max_val=AO_VMAX)
                t.write([0.0]*len(chans) if len(chans)>1 else 0.0)
        except Exception:
            pass

    def on_exit(self):
        self.running=False
        self._drive_stop()
        if self.spec is not None:
            try: self.spec.close()
            except Exception: pass
        time.sleep(0.05)
        self._save_settings()
        self._zero_all_outputs()
        self.destroy(); os._exit(0)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root=tk.Tk(); root.withdraw()
    dlg=DeviceSelectDialog(root); root.wait_window(dlg)
    device=dlg.result; root.destroy()
    if device is None:
        import sys; sys.exit(0)
    tab=DeviceSelectDialog.TAB_FOR_DEVICE.get(device,0)
    app=LCApp(device_mode=device,initial_tab=tab)
    app.mainloop()