# Maxim Sabin 613AB
"""
APLICAȚIE: SISTEM BANCAR CU THREADING
PySimpleGUI-4-foss + MySQL + Threading
"""

import PySimpleGUI as sg  # GUI framework
import mysql.connector      # Conexiune MySQL
import random               # Generare IBAN
import re                   # Validare CNP/PIN
import threading            # Threading pentru operatii async
import queue                # Coada pentru rezultate thread-uri
import pyperclip            # Copiere in clipboard

sg.theme("LightBlue3")


class DatabaseManager:
    """Gestionează conexiunea cu MySQL (thread-safe)"""
    
    def __init__(self, host="localhost", user="root", password="tema", database="banca"):
        self.lock = threading.Lock()
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        
        try:
            # Conectare initiala fara baza de date pentru a o crea daca nu exista
            conn_temp = mysql.connector.connect(
                host=host, user=user, password=password,
                connection_timeout=5
            )
            cursor_temp = conn_temp.cursor()
            cursor_temp.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            conn_temp.commit()
            cursor_temp.close()
            conn_temp.close()
            
            # Conectare la baza de date
            self.conn = mysql.connector.connect(
                host=host, user=user, password=password, database=database,
                connection_timeout=5
            )
            self.cursor = self.conn.cursor(dictionary=True)
            self._init_db()
        except mysql.connector.Error as e:
            sg.popup_error(
                f"Eroare conectare MySQL: {e}\n\n"
                "Verificați:\n"
                "1. MySQL server rulează (Services.msc)\n"
                "2. User '{user}' cu parola corectă\n\n"
                "Porniți MySQL și încercați din nou!"
            )
            raise SystemExit(1)
    
    def _init_db(self):
        """Creează tabela dacă nu există"""
        with self.lock:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conturi (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    nume VARCHAR(50) NOT NULL,
                    prenume VARCHAR(50) NOT NULL,
                    cnp VARCHAR(13) UNIQUE NOT NULL,
                    iban VARCHAR(34) UNIQUE NOT NULL,
                    sold DECIMAL(15,2) DEFAULT 0,
                    pin VARCHAR(4) NOT NULL,
                    status ENUM('activ','inchis') DEFAULT 'activ'
                )
            """)
            self.conn.commit()
    
    def executa(self, sql, params=(), fetch=False):
        """Execută query thread-safe"""
        with self.lock:
            self.cursor.execute(sql, params)
            self.conn.commit()
            if fetch:
                # Consumă toate rezultatele pentru a evita "Unread result found"
                results = self.cursor.fetchall()
                return results[0] if results else None
            return self.cursor.rowcount
    
    def genereaza_iban(self):
        """Generează IBAN unic"""
        while True:
            iban = f"RO{random.randint(10,99)}BANK{random.randint(10**9,10**10-1)}{random.randint(10**9,10**10-1)}"
            with self.lock:
                self.cursor.execute("SELECT 1 FROM conturi WHERE iban=%s", (iban,))
                if not self.cursor.fetchone():
                    return iban


class ContBancar:
    def __init__(self, date):
        if date:
            self.__dict__.update(date)
            self.sold = float(self.sold)
    
    def __str__(self):
        return f"{self.nume} {self.prenume} | IBAN: {self.iban} | Sold: {self.sold:.2f} RON"


class BancaGUI:
    def __init__(self, db):
        self.db = db
        self.cont = None
        self.win = None
        self.op = None
        self.result_queue = queue.Queue()
    
    def popup(self, msg, titlu="Info", eroare=False):
        if eroare:
            sg.popup(msg, title=titlu, button_color=('white', 'red'), 
                    line_width=50, custom_text="Încearcă din nou")
        else:
            sg.popup(msg, title=titlu)
    
    def valid(self, cnp, pin):
        return bool(re.match(r'^\d{13}$', cnp)) and bool(re.match(r'^\d{4}$', pin))
    
    def fereastra(self, titlu, layout, **kw):
        if self.win:
            self.win.close()
        self.win = sg.Window(titlu, layout, finalize=True, **kw)
    
    def login(self):
        self.fereastra("Login", [
            [sg.Text("SISTEM BANCAR", font=("Helvetica", 18, "bold"))],
            [sg.Text("CNP:"), sg.Input(key="cnp", size=15)],
            [sg.Text("PIN:"), sg.Input(key="pin", password_char="*", size=15)],
            [sg.Button("Login"), sg.Button("Cont Nou"), sg.Button("Ieșire")]
        ], element_justification="center")
    
    def cont_nou(self):
        self.fereastra("Cont Nou", [
            [sg.Text("DESCHIDERE CONT", font=("Helvetica", 14, "bold"))],
            [sg.Text("Nume:"), sg.Input(key="nume")],
            [sg.Text("Prenume:"), sg.Input(key="prenume")],
            [sg.Text("CNP (13 cifre):"), sg.Input(key="cnp")],
            [sg.Text("PIN (4 cifre):"), sg.Input(key="pin", password_char="*")],
            [sg.Text("Sold inițial:"), sg.Input("0", key="sold")],
            [sg.Button("Creează"), sg.Button("Înapoi")]
        ])
    
    def meniu(self):
        self.fereastra("Meniu", [
            [sg.Text(f"Bine ai venit, {self.cont.nume}!", font=("Helvetica", 14, "bold"))],
            [sg.Text(f"IBAN: {self.cont.iban} | Sold: {self.cont.sold:.2f} RON")],
            [sg.Button("Depunere"), sg.Button("Retragere"), sg.Button("Interogare")],
            [sg.Button("Transfer"), sg.Button("Închide Cont"), sg.Button("Logout")]
        ], element_justification="center")
    
    def operatiune(self, tip):
        layouts = {
            "depunere": [
                [sg.Text("Sumă de depus (RON):"), sg.Input(key="suma")],
                [sg.Button("OK"), sg.Button("Anulare")]
            ],
            "retragere": [
                [sg.Text(f"Sold disponibil: {self.cont.sold:.2f} RON")],
                [sg.Text("Sumă de retras (RON):"), sg.Input(key="suma")],
                [sg.Button("OK"), sg.Button("Anulare")]
            ],
            "transfer": [
                [sg.Text(f"Sold disponibil: {self.cont.sold:.2f} RON")],
                [sg.Text("IBAN destinație:"), sg.Input(key="dest")],
                [sg.Text("Sumă (RON):"), sg.Input(key="suma")],
                [sg.Button("OK"), sg.Button("Anulare")]
            ],
            "interogare": [
                [sg.Text(str(self.cont))],
                [sg.Button("Copiază IBAN", key="-COPY-IBAN-"), sg.Button("OK")]
            ]
        }
        self.fereastra(tip.capitalize(), [[sg.Text(tip.upper(), font=("Helvetica", 14, "bold"))]] + layouts.get(tip, []))
        return tip
    
    def run_async(self, func, callback=None):
        """Rulează funcție în thread separat"""
        def wrapper():
            try:
                result = func()
                if callback:
                    self.result_queue.put((callback, result, None))
            except Exception as e:
                if callback:
                    self.result_queue.put((callback, None, str(e)))
        
        threading.Thread(target=wrapper, daemon=True).start()
    
    def check_async_results(self):
        """Verifică rezultate din queue"""
        try:
            while True:
                callback, result, error = self.result_queue.get_nowait()
                if error:
                    self.popup(f"Eroare: {error}", eroare=True)
                else:
                    callback(result)
        except queue.Empty:
            pass
    
    def run(self):
        self.login()
        while True:
            ev, val = self.win.read(timeout=100)
            
            self.check_async_results()
            
            if ev == sg.WIN_CLOSED or ev == "Ieșire":
                break
            
            # ========== LOGIN ==========
            if ev == "Login":
                cnp = val["cnp"].strip()
                pin = val["pin"].strip()
                
                if not self.valid(cnp, pin):
                    self.popup(f"CNP sau PIN invalid!\n\nCNP introdus: {len(cnp)} caractere\nPIN introdus: {len(pin)} caractere\n\nCNP trebuie să aibă 13 cifre\nPIN trebuie să aibă 4 cifre", eroare=True)
                    continue
                
                def do_login():
                    self.db.cursor.execute(
                        "SELECT * FROM conturi WHERE cnp=%s AND pin=%s AND status='activ'",
                        (cnp, pin)
                    )
                    r = self.db.cursor.fetchone()
                    return ContBancar(r) if r else None
                
                def on_login(result):
                    if result:
                        self.cont = result
                        self.meniu()
                    else:
                        self.popup("Autentificare eșuată!", eroare=True)
                
                self.run_async(do_login, on_login)
            
            elif ev == "Cont Nou":
                self.cont_nou()
            
            elif ev == "Creează":
                nume = val["nume"].strip()
                prenume = val["prenume"].strip()
                cnp = val["cnp"].strip()
                pin = val["pin"].strip()
                
                if not (nume and prenume):
                    self.popup("Numele și prenumele sunt obligatorii!", eroare=True)
                    continue
                
                if not self.valid(cnp, pin):
                    self.popup(f"CNP sau PIN invalid!\n\nCNP introdus: {len(cnp)} caractere\nPIN introdus: {len(pin)} caractere\n\nCNP trebuie să aibă 13 cifre\nPIN trebuie să aibă 4 cifre", eroare=True)
                    continue
                
                try:
                    sold = float(val["sold"] or 0)
                    iban = self.db.genereaza_iban()
                    self.db.executa(
                        "INSERT INTO conturi (nume,prenume,cnp,iban,pin,sold) VALUES (%s,%s,%s,%s,%s,%s)",
                        (nume, prenume, cnp, iban, pin, sold)
                    )
                    self.popup(f"Cont creat!\nIBAN: {iban}")
                    self.login()
                except Exception as e:
                    self.popup(f"Eroare: {e}", eroare=True)
            
            elif ev == "Înapoi":
                self.login()
            
            # ========== MENIU ==========
            elif ev == "Logout":
                self.cont = None
                self.login()
            
            elif ev in ("Depunere", "Retragere", "Transfer", "Interogare"):
                self.op = self.operatiune(ev.lower())
            
            elif ev == "Închide Cont":
                if sg.popup_yes_no("Sigur doriți să închideți contul?\nSoldul trebuie să fie 0!") == "Yes":
                    if self.cont.sold != 0:
                        self.popup(f"Sold rămas: {self.cont.sold:.2f} RON. Retrageți banii!", eroare=True)
                    else:
                        def do_close():
                            return self.db.executa("UPDATE conturi SET status='inchis' WHERE iban=%s", (self.cont.iban,))
                        
                        def on_close(_):
                            self.popup("Cont închis!")
                            self.cont = None
                            self.login()
                        
                        self.run_async(do_close, on_close)
            
            # ========== OPERAȚIUNI ==========
            elif ev == "OK":
                try:
                    if self.op == "depunere":
                        suma = float(val["suma"])
                        
                        def do_depunere():
                            self.db.executa("UPDATE conturi SET sold=sold+%s WHERE iban=%s", (suma, self.cont.iban))
                            return suma
                        
                        def on_depunere(s):
                            self.cont.sold += s
                            self.popup(f"Depus {s:.2f} RON\nSold: {self.cont.sold:.2f} RON")
                            self.meniu()
                        
                        self.run_async(do_depunere, on_depunere)
                    
                    elif self.op == "retragere":
                        suma = float(val["suma"])
                        if suma > self.cont.sold:
                            self.popup("Fonduri insuficiente!", eroare=True)
                        else:
                            def do_retragere():
                                self.db.executa("UPDATE conturi SET sold=sold-%s WHERE iban=%s", (suma, self.cont.iban))
                                return suma
                            
                            def on_retragere(s):
                                self.cont.sold -= s
                                self.popup(f"Retras {s:.2f} RON\nSold: {self.cont.sold:.2f} RON")
                                self.meniu()
                            
                            self.run_async(do_retragere, on_retragere)
                    
                    elif self.op == "transfer":
                        suma = float(val["suma"])
                        if suma > self.cont.sold:
                            self.popup("Fonduri insuficiente!", eroare=True)
                        elif val["dest"].strip().upper() == self.cont.iban:
                            self.popup("Nu puteți transfera către propriul cont!", eroare=True)
                        else:
                            dest = val["dest"].strip().upper()
                            
                            def do_transfer():
                                if not self.db.executa("SELECT 1 FROM conturi WHERE iban=%s AND status='activ'", (dest,), True):
                                    raise Exception("Cont destinație invalid!")
                                self.db.executa("UPDATE conturi SET sold=sold-%s WHERE iban=%s", (suma, self.cont.iban))
                                self.db.executa("UPDATE conturi SET sold=sold+%s WHERE iban=%s", (suma, dest))
                                return suma
                            
                            def on_transfer(s):
                                self.cont.sold -= s
                                self.popup(f"Transferat {s:.2f} RON\nSold: {self.cont.sold:.2f} RON")
                                self.meniu()
                            
                            self.run_async(do_transfer, on_transfer)
                    
                    elif self.op == "interogare":
                        self.meniu()
                except ValueError:
                    self.popup("Sumă invalidă!", eroare=True)
            
            elif ev == "-COPY-IBAN-":
                if self.cont:
                    pyperclip.copy(self.cont.iban)
                    sg.popup("IBAN copiat în clipboard!", title="Succes", auto_close=True, auto_close_duration=1)
            
            elif ev == "Anulare":
                self.meniu()
        
        self.win.close()


if __name__ == "__main__":
    BancaGUI(DatabaseManager()).run()