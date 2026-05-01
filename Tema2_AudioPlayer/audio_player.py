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

# Formate audio suportate de pygame
AUDIO_FORMATS = ('*.mp3', '*.ogg', '*.wav', '*.mid', '*.midi', '*.mod', '*.xm', '*.flac')

# Conexiuni HAND_CONNECTIONS (din documentația MediaPipe)
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
])


class GestureRecognizer:
    """Recunoaște gesturi folosind MediaPipe HandLandmarker"""

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
        if not os.path.exists(self.MODEL_PATH):
            print("Se descarcă modelul MediaPipe...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print("Model descărcat!")

    def detecteaza_gest(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        if not results.hand_landmarks:
            return None, None

        landmarks = results.hand_landmarks[0]

        # Verifică gesturi în ordinea specificității (cele mai unice primele)
        if self._este_ok(landmarks): return "OK", landmarks
        if self._este_rock(landmarks): return "ROCK", landmarks
        if self._este_index_sus(landmarks): return "UP", landmarks
        if self._este_index_jos(landmarks): return "DOWN", landmarks
        if self._este_palma(landmarks): return "PALM", landmarks
        if self._este_pumn(landmarks): return "FIST", landmarks

        return None, landmarks

    def deseneaza(self, frame, landmarks):
        if not landmarks:
            return frame

        h, w, _ = frame.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

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
        # Toate degetele pliate clar (nu doar comparând y, ci distanță față de palmă)
        degete_pliate = all(
            lm[tip].y > lm[pip].y + 0.03
            for tip, pip in [
                (self.INDEX_TIP, self.INDEX_PIP),
                (self.MIDDLE_TIP, self.MIDDLE_PIP),
                (self.RING_TIP, self.RING_PIP),
                (self.PINKY_TIP, self.PINKY_PIP)
            ]
        )
        # Degetele trebuie să fie aproape de palmă (nu întinse)
        distanta_mica = all(
            self._dist(lm[tip], lm[self.WRIST]) < 0.25
            for tip in [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        )
        return degete_pliate and distanta_mica

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
        # Index pliat clar în jos, celelalte degete întinse în sus
        index_pliat = lm[self.INDEX_TIP].y > lm[self.INDEX_PIP].y + 0.05
        mijlociu_intins = lm[self.MIDDLE_TIP].y < lm[self.MIDDLE_PIP].y - 0.05
        inelar_intins = lm[self.RING_TIP].y < lm[self.RING_PIP].y - 0.05
        mic_intins = lm[self.PINKY_TIP].y < lm[self.PINKY_PIP].y - 0.05
        # Index trebuie să fie vizibil sub celelalte degete
        index_jos = lm[self.INDEX_TIP].y > lm[self.MIDDLE_TIP].y
        return index_pliat and mijlociu_intins and inelar_intins and mic_intins and index_jos


class AudioPlayer:
    """Redă fișiere audio folosind pygame.mixer"""

    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.playing = False
        self.paused = False
        self.current_file = None

    def load(self, filepath):
        """Încarcă fișierul audio"""
        try:
            pygame.mixer.music.load(filepath)
            self.current_file = filepath
            return True
        except Exception as e:
            print(f"Eroare la încărcarea audio: {e}")
            return False

    def play(self):
        """Pornește redarea"""
        if self.current_file:
            pygame.mixer.music.play()
            self.playing = True
            self.paused = False

    def pause(self):
        """Pauză"""
        if self.playing and not self.paused:
            pygame.mixer.music.pause()
            self.paused = True

    def resume(self):
        """Reluare"""
        if self.playing and self.paused:
            pygame.mixer.music.unpause()
            self.paused = False

    def stop(self):
        """Oprește redarea"""
        pygame.mixer.music.stop()
        self.playing = False
        self.paused = False

    def is_playing(self):
        return pygame.mixer.music.get_busy()


class Playlist:
    def __init__(self, folder):
        self.folder = folder
        self.melodii = []
        self.idx = -1
        self.scaneaza()

    def scaneaza(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder, exist_ok=True)
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


class AudioPlayerGUI:
    def __init__(self, folder):
        sg.theme("LightBlue3")
        self.playlist = Playlist(folder)
        self.gest = GestureRecognizer()
        self.camera = cv2.VideoCapture(0)
        self.audio = AudioPlayer()
        self.mod = "BROWSE"
        self.win_browse = None
        self.win_play = None
        self.running = True
        self.ultimul_gest_timp = 0
        self.cooldown_gest = 0.8
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
            [sg.Image(filename="", key="-CAM-", size=(320, 240))],
            [sg.Text(self.playlist.nume(), font=("Helvetica", 12), key="-NUME-")],
            [sg.Text("Gest detectat: -", key="-GEST-")],
            [sg.Text("✋ Palmă: pauză | 👌 OK: resume | 🤘 Rock: stop", font=("Helvetica", 10))],
            [sg.Button("Înapoi", key="-BACK-")]
        ]
        self.win_play = sg.Window("Redare", layout, finalize=True, size=(500, 500))

    def actualizeaza_cam(self, window, key="-CAM-"):
        ret, frame = self.camera.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        gest, landmarks = self.gest.detecteaza_gest(frame)
        frame = self.gest.deseneaza(frame, landmarks)
        frame = cv2.resize(frame, (320, 240))

        img = cv2.imencode('.png', frame)[1].tobytes()
        try:
            window[key].update(data=img)
        except:
            pass

        return gest

    def proceseaza_gest(self, gest):
        if not gest:
            return

        timp_curent = time.time()
        if timp_curent - self.ultimul_gest_timp < self.cooldown_gest:
            return

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
                self.audio.pause()
                self._update_status("Pauză")
            elif gest == "OK":
                self.audio.resume()
                self._update_status("Redare...")
            elif gest == "ROCK":
                self._stop_play()

    def _update_list(self):
        try:
            self.win_browse["-LIST-"].update(set_to_index=self.playlist.idx)
            self.win_browse["-STATUS-"].update(f"Selectat: {self.playlist.nume()}")
        except:
            pass

    def _update_status(self, text):
        try:
            if self.mod == "PLAYBACK":
                self.win_play["-GEST-"].update(f"Gest detectat: {text}")
        except:
            pass

    def _start_play(self):
        cale = self.playlist.curent()
        if not cale:
            return

        if not self.audio.load(cale):
            sg.popup_error("Eroare la încărcarea fișierului audio!")
            return

        self.audio.play()
        self.mod = "PLAYBACK"
        self.win_browse.hide()
        self.creaza_play()

    def _stop_play(self):
        self.audio.stop()
        self.mod = "BROWSE"
        try:
            self.win_play.close()
            self.win_browse.un_hide()
        except:
            pass

    def run(self):
        while self.running:
            if self.mod == "BROWSE":
                event, values = self.win_browse.read(timeout=50)

                if event == sg.WIN_CLOSED:
                    self.running = False
                    break

                gest = self.actualizeaza_cam(self.win_browse)
                if gest:
                    self.proceseaza_gest(gest)
                    self.win_browse["-STATUS-"].update(f"Detectat: {gest}")

            else:  # PLAYBACK mode
                event, values = self.win_play.read(timeout=50)

                if event == sg.WIN_CLOSED:
                    self.audio.stop()
                    self.mod = "BROWSE"
                    try:
                        self.win_play.close()
                        self.win_browse.un_hide()
                    except:
                        pass
                    continue

                if event == "-BACK-":
                    self._stop_play()
                    continue

                # Verifică dacă modul s-a schimbat (gest ROCK a apelat _stop_play)
                if self.mod != "PLAYBACK":
                    continue

                gest = self.actualizeaza_cam(self.win_play)
                if gest and self.mod == "PLAYBACK":
                    self.proceseaza_gest(gest)
                    if self.mod == "PLAYBACK":
                        try:
                            self.win_play["-GEST-"].update(f"Gest detectat: {gest}")
                        except:
                            pass

        # Cleanup
        self.camera.release()
        self.audio.stop()
        try:
            self.win_browse.close()
        except:
            pass


if __name__ == "__main__":
    folder = os.path.join(os.path.dirname(__file__), "melodii")
    os.makedirs(folder, exist_ok=True)

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
