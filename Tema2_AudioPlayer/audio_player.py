# Maxim Sabin 613AB
"""
APLICAȚIE: AUDIOPLAYER CU CONTROL PRIN GESTURI
PySimpleGUI-4-foss + MediaPipe Tasks (Python 3.13 compatible)
"""

import os
import glob
import threading
import time
import cv2
import pygame
import PySimpleGUI as sg
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request

# Incercare import drawing utils (pot varia intre versiuni)
try:
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe import solutions
    HAS_DRAWING_UTILS = True
except ImportError:
    HAS_DRAWING_UTILS = False


# Formate audio suportate de pygame
AUDIO_FORMATS = ('*.mp3', '*.ogg', '*.wav', '*.mid', '*.midi', '*.mod', '*.xm', '*.flac')

# Conexiuni HAND_CONNECTIONS (din documentația MediaPipe)
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),           # deget mare
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # mijlociu
    (0, 13), (13, 14), (14, 15), (15, 16),    # inelar
    (0, 17), (17, 18), (18, 19), (19, 20),    # mic
    (5, 9), (9, 13), (13, 17)                 # palmă
])


# --- PARTEA II: RECUNOAȘTERE GESTURI ---
class GestureRecognizer:
    """Recunoaște gesturi folosind MediaPipe HandLandmarker (Tasks API)"""

    # Landmark indices pentru detectare gesturi
    WRIST, THUMB_TIP, INDEX_TIP, INDEX_PIP = 0, 4, 8, 6
    MIDDLE_TIP, MIDDLE_PIP, RING_TIP, RING_PIP = 12, 10, 16, 14
    PINKY_TIP, PINKY_PIP = 20, 18

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"

    def __init__(self):
        self._descarca_model()
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _descarca_model(self):
        """Descarcă modelul dacă nu există"""
        if not os.path.exists(self.MODEL_PATH):
            print("Se descarcă modelul MediaPipe...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print("Model descărcat!")

    def detecteaza_gest(self, frame):
        """Detectează gestul din frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.detector.detect(mp_image)

        if not results.hand_landmarks:
            return None, None

        landmarks = results.hand_landmarks[0]

        # Verifică gesturi
        if self._este_pumn(landmarks): return "FIST", landmarks
        if self._este_ok(landmarks): return "OK", landmarks
        if self._este_rock(landmarks): return "ROCK", landmarks
        if self._este_palma(landmarks): return "PALM", landmarks
        if self._este_index_sus(landmarks): return "UP", landmarks
        if self._este_index_jos(landmarks): return "DOWN", landmarks

        return None, landmarks

    def deseneaza(self, frame, landmarks):
        """Desenează landmarkurile pe frame"""
        if not landmarks:
            return frame

        h, w, _ = frame.shape

        # Desenează puncte
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Desenează conexiuni folosind HAND_CONNECTIONS
        for start, end in HAND_CONNECTIONS:
            if start < len(landmarks) and end < len(landmarks):
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return frame

    def _dist(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _indoit(self, lm, tip, pip):
        return lm[tip].y > lm[pip].y

    def _intins(self, lm, tip, pip):
        return lm[tip].y < lm[pip].y

    def _este_pumn(self, lm):
        return all(self._indoit(lm, t, p) for t, p in [
            (self.INDEX_TIP, self.INDEX_PIP), (self.MIDDLE_TIP, self.MIDDLE_PIP),
            (self.RING_TIP, self.RING_PIP), (self.PINKY_TIP, self.PINKY_PIP)
        ])

    def _este_palma(self, lm):
        return all(self._intins(lm, t, p) for t, p in [
            (self.INDEX_TIP, self.INDEX_PIP), (self.MIDDLE_TIP, self.MIDDLE_PIP),
            (self.RING_TIP, self.RING_PIP), (self.PINKY_TIP, self.PINKY_PIP)
        ])

    def _este_ok(self, lm):
        return self._dist(lm[self.INDEX_TIP], lm[self.THUMB_TIP]) < 0.08

    def _este_rock(self, lm):
        return (self._intins(lm, self.INDEX_TIP, self.INDEX_PIP) and
                self._intins(lm, self.PINKY_TIP, self.PINKY_PIP) and
                self._indoit(lm, self.MIDDLE_TIP, self.MIDDLE_PIP) and
                self._indoit(lm, self.RING_TIP, self.RING_PIP))

    def _este_index_sus(self, lm):
        return (self._intins(lm, self.INDEX_TIP, self.INDEX_PIP) and
                self._indoit(lm, self.MIDDLE_TIP, self.MIDDLE_PIP) and
                self._indoit(lm, self.RING_TIP, self.RING_PIP) and
                self._indoit(lm, self.PINKY_TIP, self.PINKY_PIP))

    def _este_index_jos(self, lm):
        return (lm[self.INDEX_TIP].y > lm[self.WRIST].y and
                self._indoit(lm, self.MIDDLE_TIP, self.MIDDLE_PIP) and
                self._indoit(lm, self.RING_TIP, self.RING_PIP) and
                self._indoit(lm, self.PINKY_TIP, self.PINKY_PIP))


# --- PARTEA III: REDARE VIDEO ---
class VideoPlayer:
    def __init__(self, window, key):
        self.window = window
        self.key = key
        self.cap = None
        self.playing = False
        self.paused = False
        self.thread = None
        self.lock = threading.Lock()
        pygame.mixer.init()

    def incarca(self, cale):
        self.cap = cv2.VideoCapture(cale)
        if not self.cap.isOpened():
            return False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_delay = 1.0 / self.fps
        return True

    def reda(self):
        with self.lock:
            self.playing = True
            self.paused = False
        self.thread = threading.Thread(target=self._bucla, daemon=True)
        self.thread.start()

    def pauza(self):
        with self.lock:
            self.paused = True

    def resume(self):
        with self.lock:
            self.paused = False

    def stop(self):
        with self.lock:
            self.playing = False
            self.paused = False
        if self.cap:
            self.cap.release()
        pygame.mixer.music.stop()

    def _bucla(self):
        while True:
            with self.lock:
                if not self.playing:
                    break
                if self.paused:
                    time.sleep(0.05)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            img = cv2.imencode('.png', frame_rgb)[1].tobytes()

            try:
                self.window.write_event_value("-FRAME-", img)
            except:
                break

            time.sleep(self.frame_delay)

        with self.lock:
            self.playing = False


# --- PARTEA IV: PLAYLIST ---
class Playlist:
    def __init__(self, folder):
        self.folder = folder
        self.melodii = []
        self.idx = -1
        self.scaneaza()

    def scaneaza(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder, exist_ok=True)
        # Caută toate formatele audio suportate
        self.melodii = []
        for fmt in AUDIO_FORMATS:
            self.melodii.extend(glob.glob(os.path.join(self.folder, fmt)))
        self.melodii = sorted(self.melodii)
        self.idx = 0 if self.melodii else -1

    def curent(self):
        if 0 <= self.idx < len(self.melodii):
            return self.melodii[self.idx]
        return None

    def nume(self):
        c = self.curent()
        return os.path.basename(c) if c else "Nicio melodie"

    def prev(self):
        if self.melodii:
            self.idx = (self.idx - 1) % len(self.melodii)

    def next(self):
        if self.melodii:
            self.idx = (self.idx + 1) % len(self.melodii)


# --- PARTEA V: INTERFAȚA GRAFICĂ ---
class AudioPlayerGUI:
    def __init__(self, folder):
        sg.theme("LightBlue3")
        self.playlist = Playlist(folder)
        self.gest = GestureRecognizer()
        self.camera = cv2.VideoCapture(0)
        self.player = None
        self.mod = "BROWSE"
        self.win_browse = None
        self.win_play = None
        # Cooldown pentru gesturi (secunde)
        self.ultimul_gest_timp = 0
        self.cooldown_gest = 0.8  # 800ms intre gesturi
        self.creaza_browse()

    def creaza_browse(self):
        melodii = [os.path.basename(m) for m in self.playlist.melodii]
        layout = [
            [sg.Text("AUDIOPLAYER - Selectează melodia cu gesturi", font=("Helvetica", 14, "bold"))],
            [sg.Listbox(melodii, size=(50, 10), key="-LIST-", select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
                       default_values=[melodii[self.playlist.idx]] if melodii else [])],
            [sg.Image(filename="", key="-CAM-", size=(320, 240))],
            [sg.Text("☝️ Sus/Jos: navigare | ✊ Pumn: redare", font=("Helvetica", 10))],
            [sg.Text("Status: Așteaptă gest...", key="-STATUS-", size=(40, 1))]
        ]
        self.win_browse = sg.Window("AudioPlayer", layout, finalize=True)
        if melodii:
            self.win_browse["-LIST-"].update(set_to_index=self.playlist.idx)

    def creaza_play(self):
        layout = [
            [sg.Text("REDARE", font=("Helvetica", 14, "bold"))],
            [sg.Image(filename="", key="-VIDEO-", size=(640, 480))],
            [sg.Text("Gest detectat: -", key="-GEST-")],
            [sg.Text("✋ Palmă: pauză | 👌 OK: resume | 🤘 Rock: stop", font=("Helvetica", 10))],
            [sg.Button("Înapoi", key="-BACK-")]
        ]
        self.win_play = sg.Window("Redare", layout, finalize=True, size=(700, 600))
        self.player = VideoPlayer(self.win_play, "-VIDEO-")

    def actualizeaza_cam(self):
        ret, frame = self.camera.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        gest, landmarks = self.gest.detecteaza_gest(frame)
        frame = self.gest.deseneaza(frame, landmarks)
        frame = cv2.resize(frame, (320, 240))

        img = cv2.imencode('.png', frame)[1].tobytes()
        self.win_browse["-CAM-"].update(data=img)

        return gest

    def proceseaza(self, gest):
        if not gest:
            return
        
        # Verifică cooldown
        timp_curent = time.time()
        if timp_curent - self.ultimul_gest_timp < self.cooldown_gest:
            return  # Ignoră gestul dacă e în cooldown
        
        self.ultimul_gest_timp = timp_curent

        if self.mod == "BROWSE":
            if gest == "UP":
                self.playlist.prev()
                self._update_list()
            elif gest == "DOWN":
                self.playlist.next()
                self._update_list()
            elif gest == "FIST" and self.playlist.curent():
                self._start_play()
        else:
            if gest == "PALM":
                self.player.pauza()
            elif gest == "OK":
                self.player.resume()
            elif gest == "ROCK":
                self._stop_play()

    def _update_list(self):
        self.win_browse["-LIST-"].update(set_to_index=self.playlist.idx)
        self.win_browse["-STATUS-"].update(f"Selectat: {self.playlist.nume()}")

    def _start_play(self):
        self.mod = "PLAYBACK"
        self.win_browse.hide()
        self.creaza_play()

        if self.player.incarca(self.playlist.curent()):
            self.player.reda()
        else:
            sg.popup_error("Eroare la încărcarea video-ului!")
            self._stop_play()

    def _stop_play(self):
        if self.player:
            self.player.stop()
        self.win_play.close()
        self.win_browse.un_hide()
        self.mod = "BROWSE"
        self.player = None

    def run(self):
        while True:
            win = self.win_play if self.mod == "PLAYBACK" else self.win_browse
            event, values = win.read(timeout=50)

            if event == sg.WIN_CLOSED:
                break

            if event == "-BACK-":
                self._stop_play()

            if event == "-FRAME-" and self.mod == "PLAYBACK":
                try:
                    self.win_play["-VIDEO-"].update(data=values[event])
                except:
                    pass

            # Procesează gesturi
            if self.mod == "BROWSE":
                gest = self.actualizeaza_cam()
                if gest:
                    self.proceseaza(gest)
                    self.win_browse["-STATUS-"].update(f"Detectat: {gest}")
            else:
                ret, frame = self.camera.read()
                if ret:
                    gest, _ = self.gest.detecteaza_gest(cv2.flip(frame, 1))
                    if gest:
                        self.proceseaza(gest)
                        self.win_play["-GEST-"].update(f"Gest detectat: {gest}")

        # Cleanup
        self.camera.release()
        if self.player:
            self.player.stop()
        self.win_browse.close()


# --- PARTEA VI: BLOCUL PRINCIPAL ---
if __name__ == "__main__":
    folder = os.path.join(os.path.dirname(__file__), "melodii")
    os.makedirs(folder, exist_ok=True)

    # Verifică dacă există fișiere audio
    melodii_existente = []
    for fmt in AUDIO_FORMATS:
        melodii_existente.extend(glob.glob(os.path.join(folder, fmt)))

    if not melodii_existente:
        sg.popup(
            f"Folderul cu melodii este gol!\n\n"
            f"PATH complet: {folder}\n\n"
            f"Formate suportate: MP3, OGG, WAV, MIDI, MOD, XM, FLAC\n\n"
            f"Adăugați fișiere audio în acest folder înainte de rulare.",
            title="Atenție - Nu există melodii",
            custom_text="Închide"
        )
        raise SystemExit(0)

    try:
        app = AudioPlayerGUI(folder)
        app.run()
    except Exception as e:
        sg.popup_error(f"Eroare aplicație: {e}")
