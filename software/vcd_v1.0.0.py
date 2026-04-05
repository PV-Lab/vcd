import sys
import time
import threading
from collections import deque
import datetime as dt
import csv
import os

# --- Third-party ---
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QPushButton, QComboBox, QDialogButtonBox, QMessageBox,
    QMainWindow, QWidget, QFileDialog, QStatusBar
)
from PyQt5.QtCore import QTimer
import serial
import serial.tools.list_ports

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# ---------------- Config ----------------
DEFAULT_BAUD = 9600
SER_TIMEOUT = 0.05       # must be << SAMPLE_PERIOD
SAMPLE_PERIOD = 0.2      # 200 ms = 5 Hz acquisition
PLOT_PERIOD_MS = 500     # redraw every 0.5 s (Qt timer, in ms)
WINDOW_SECONDS = 60      # rolling window for display
CMD = b"?V752\r"
DEFAULT_PREFIX = "data_VCD"
# ---------------------------------------


def now_dt():
    return dt.datetime.now()


def default_csv_name(prefix=DEFAULT_PREFIX):
    return now_dt().strftime('%Y.%m.%d_%H.%M.%S') + f'_{prefix}.csv'


class Sampler:
    """
    Stream pressure readings from the selected COM port in a background thread,
    keep a rolling window for plotting, and (optionally) log data to CSV while
    in recording mode. Recording can be toggled on/off without stopping the stream.
    Tracks recording intervals for on-plot highlighting.
    """
    def __init__(self, port: str, baud: int, timeout: float):
        try:
            self.ser = serial.Serial(
                port=port, baudrate=baud, bytesize=8,
                parity='N', stopbits=1, timeout=timeout
            )
        except Exception as e:
            raise RuntimeError(f"Could not open {port}: {e}")

        cap = int(WINDOW_SECONDS * (1.0 / SAMPLE_PERIOD)) + 200
        self.t = deque(maxlen=cap)   # rolling window timestamps
        self.p = deque(maxlen=cap)   # rolling window pressures

        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)

        # Recording state
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self._rows_written = 0
        # Recording interval tracking (list of (start_dt, end_dt or None))
        self._rec_intervals = []

    # --- Public API for recording control ---
    def start(self):
        self.thread.start()

    def start_recording(self, path: str):
        # Open CSV and write header; safe to call from UI thread
        if self.recording:
            return
        try:
            self.csv_file = open(path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Timestamp', 'Pressure (Pa)'])
            self.csv_path = path
            self._rows_written = 0
            with self.lock:
                # Start a new open interval
                self._rec_intervals.append([now_dt(), None])
            self.recording = True
        except Exception as e:
            raise RuntimeError(f"Failed to start recording to {path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to start recording to {path}: {e}")

    def stop_recording(self) -> str:
        # Close CSV; return saved path
        if not self.recording:
            return ''
        self.recording = False
        # Close last open interval
        with self.lock:
            if self._rec_intervals and self._rec_intervals[-1][1] is None:
                self._rec_intervals[-1][1] = now_dt()
        saved = self.csv_path or ''
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        finally:
            self.csv_file = None
            self.csv_writer = None
            self.csv_path = None
        return saved 

    def stop(self):
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        # Ensure recording file is closed even if user forgot to stop
        try:
            self.stop_recording()
        except Exception:
            pass
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    # --- Internal ---
    def _read_once(self):
        try:
            self.ser.reset_input_buffer()
            self.ser.write(CMD)
            response = self.ser.readline().decode('ascii', errors='ignore').strip()
            if not response:
                return None
            if response and response[0] == "=":
                # Example: '= V752 101325; ...'  -> take the numeric token before ';'
                try:
                    pressure_string = response.split()[1].split(';')[0]
                    return float(pressure_string)
                except Exception:
                    return None
        except Exception:
            return None
        return None

    def _loop(self):
        target = time.perf_counter()
        while not self._stop.is_set():
            p = self._read_once()
            ts = now_dt()
            if p is not None:
                with self.lock:
                    self.t.append(ts)
                    self.p.append(p)
                    # Only log while recording is active
                    if self.recording and self.csv_writer is not None:
                        self.csv_writer.writerow([ts.strftime('%Y/%m/%d %H:%M:%S.%f'), p])
                        self._rows_written += 1
                        if self._rows_written % 10 == 0:
                            try:
                                self.csv_file.flush()
                            except Exception:
                                pass

            target += SAMPLE_PERIOD
            sleep_for = target - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                target = time.perf_counter()


class ConnectionDialog(QDialog):
    """Modal dialog that enumerates COM ports, lets the user pick one, and test."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Sensor")
        self.resize(420, 360)
        self.selected_port = None
        self.selected_baud = DEFAULT_BAUD

        layout = QVBoxLayout(self)

        # Ports list
        layout.addWidget(QLabel("Available serial ports:"))
        self.port_list = QListWidget()
        layout.addWidget(self.port_list)

        # Controls row: Refresh + Baud
        row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.populate_ports)
        row.addWidget(self.refresh_btn)

        row.addWidget(QLabel("Baud:"))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"]) 
        self.baud_combo.setCurrentText(str(DEFAULT_BAUD))
        row.addWidget(self.baud_combo)
        row.addStretch(1)
        layout.addLayout(row)

        # Test + Status
        test_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self.test_connection)
        test_row.addWidget(self.test_btn)

        self.status_label = QLabel("Select a port and click Test.")
        self.status_label.setWordWrap(True)
        test_row.addWidget(self.status_label, stretch=1)
        layout.addLayout(test_row)

        # OK/Cancel
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Connect")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.populate_ports()

    def populate_ports(self):
        self.port_list.clear()
        for p in serial.tools.list_ports.comports():
            item_text = f"{p.device} — {p.description}"
            self.port_list.addItem(item_text)
        if self.port_list.count() == 0:
            self.port_list.addItem("(No serial ports found)")
            self.port_list.setEnabled(False)
            self.test_btn.setEnabled(False)
        else:
            self.port_list.setEnabled(True)
            self.test_btn.setEnabled(True)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.status_label.setText("Select a port and click Test.")

    def _current_device(self):
        item = self.port_list.currentItem()
        if not item:
            return None
        text = item.text()
        # Expect the form "COMx — description"; split on space
        dev = text.split(" ")[0]
        return dev if dev.upper().startswith("COM") or dev.startswith("/dev/") else None

    def test_connection(self):
        dev = self._current_device()
        if not dev:
            QMessageBox.warning(self, "No port selected", "Please select a serial port first.")
            return
        baud = int(self.baud_combo.currentText())
        self.status_label.setText(f"Testing {dev} @ {baud}…")
        QApplication.processEvents()

        try:
            with serial.Serial(dev, baudrate=baud, timeout=0.4) as ser:
                pressure = None
                attempts = 10
                for _ in range(attempts):
                    ser.write(CMD)
                    line = ser.readline().decode('ascii', errors='ignore').strip()
                    if line and line.startswith('='):
                        try:
                            token = line.split()[1].split(';')[0]
                            pressure = float(token)
                            break
                        except Exception:
                            pass
                    QtCore.QThread.msleep(100)
        except Exception as e:
            self.status_label.setText(f"Failed to open {dev}: {e}")
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return

        if pressure is not None:
            self.status_label.setText(f"Success! Read pressure ≈ {pressure:.2f} Pa")
            self.selected_port = dev
            self.selected_baud = baud
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.status_label.setText("Connected but did not receive valid sensor data. Check wiring/port and try again.")
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)


class PlotWindow(QMainWindow):
    """Qt main window hosting a Matplotlib canvas and a single QTimer for updates.
    Adds Start/Stop Recording controls. Streaming/plotting always on; CSV only
    during recording. Shades recording intervals on the plot.
    """
    def __init__(self, port: str, baud: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Pressure vs. Time  —  {port} @ {baud}")
        self.resize(1000, 560)

        # Sampler
        self.sampler = Sampler(port, baud, SER_TIMEOUT)
        self.sampler.start()

        # Matplotlib Figure/Canvas
        self.fig = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylabel("Pressure (Pa)")
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # Keep refs to span artists so we can remove/update cleanly
        self._span_artists = []

        # Controls row
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Recording")
        self.stop_btn  = QPushButton("Stop Recording")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.on_start_recording)
        self.stop_btn.clicked.connect(self.on_stop_recording)
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addStretch(1)

        # Central widget
        cw = QWidget()
        lay = QVBoxLayout()
        lay.addLayout(ctrl_row)
        lay.addWidget(self.canvas)
        cw.setLayout(lay)
        self.setCentralWidget(cw)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._set_status_idle()

        # Single QTimer for refresh
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(PLOT_PERIOD_MS)

    # --- UI helpers ---
    def _set_status_idle(self):
        self.status.showMessage("Idle (not recording)")

    def _set_status_rec(self, path):
        self.status.showMessage(f"Recording → {os.path.basename(path)}")

    # --- Recording actions ---
    def on_start_recording(self):
        # Ask for save location; suggest timestamped default
        suggested = default_csv_name()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose CSV file to save",
            suggested,
            "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            self.sampler.start_recording(path)
        except Exception as e:
            QMessageBox.critical(self, "Failed to start recording", str(e))
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status_rec(path)

    def on_stop_recording(self):
        saved = self.sampler.stop_recording()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status_idle()
        if saved:
            QMessageBox.information(self, "Recording saved", f"Data saved to: {saved}")

    # --- Plot update ---
    def update_plot(self):
        with self.sampler.lock:
            times = list(self.sampler.t)
            vals  = list(self.sampler.p)
            # Copy intervals safely; convert lists->tuples
            intervals = [(s, e) for (s, e) in self.sampler._rec_intervals]
            # If currently recording and last interval is open, represent it with now
            if self.sampler.recording:
                if intervals and intervals[-1][1] is None:
                    intervals[-1] = (intervals[-1][0], now_dt())

        if not times:
            return
        tmax = times[-1]
        tmin = tmax - dt.timedelta(seconds=WINDOW_SECONDS)

        # Keep last WINDOW_SECONDS
        i0 = 0
        for i in range(len(times)-1, -1, -1):
            if times[i] < tmin:
                i0 = i + 1
                break
        times_w = times[i0:]
        vals_w  = vals[i0:]

        # Update line
        self.line.set_data(times_w, vals_w)
        self.ax.set_xlim(tmin, tmax)

        vmin = min(vals_w)
        vmax = max(vals_w)
        pad = max(1.0, (vmax - vmin) * 0.1)
        self.ax.set_ylim(vmin - pad, vmax + pad)

        # Remove old spans
        for art in self._span_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._span_artists.clear()

        # Draw spans for intervals that overlap the visible window
        for (start, end) in intervals:
            if end is None:
                end = now_dt()
            if end < tmin or start > tmax:
                continue
            s = max(start, tmin)
            e = min(end, tmax)
            span = self.ax.axvspan(s, e, alpha=0.15, zorder=0)
            self._span_artists.append(span)

        self.canvas.draw_idle()

    def closeEvent(self, event):
        try:
            if self.sampler.recording:
                saved = self.sampler.stop_recording()
                if saved:
                    self.status.showMessage(f"Saved before exit: {saved}")
        except Exception:
            pass
        try:
            self.timer.stop()
        except Exception:
            pass
        # Clean up span artists explicitly
        for art in self._span_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._span_artists.clear()
        try:
            self.sampler.stop()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    # Ensure exactly ONE QApplication and ONE Qt event loop runs
    app = QApplication(sys.argv)

    # 1) Connection dialog (modal)
    dlg = ConnectionDialog()
    if dlg.exec_() != QDialog.Accepted or not dlg.selected_port:
        sys.exit(0)

    # 2) Main window with embedded Matplotlib + single QTimer
    win = PlotWindow(dlg.selected_port, dlg.selected_baud)
    win.show()

    # 3) Run the ONE Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
