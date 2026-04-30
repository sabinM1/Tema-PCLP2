**Implementati una din cele doua aplicati pentru tema obligatorie.**

**1. Sistem Bancar cu interfata PySimpleGUI (versiunea 4foss) si MySQL (1.5 pct)**
O aplicatie bancara in care, un client poate face urmatoarele operatiuni:
    - deschidere cont;
    - depunere;
    - retragere;
    - interogare;
    - transfer intre doi clienti;
    - inchidere cont.

Detalii de implementare:
- datele clientilor se vor salva intr-o baza de date (hint: mysql, o singura tabela);
- fiecare student va decide numarul si tipul campurilor din baza de date de care vor avea
    nevoie pentru implementarea aplicatiei;
- evident, va fi nevoie de un camp te tip string care sa preczeze statusul contului unui
    client (activ/inchis);
- interfata grafica va fi minimala (strict butoanele necesare pentru operatiunile descrise
    mai sus).

**2. Tema recuperare punctaj pe parcurs (0.5 pct)**
Implementare a unui AudioPlayer in Python, folosind PySimpleGUI (hint: versiunea 4foss) si
MediaPipe (pentru recunoasterea a sase gesturi ale unei maini) astfel incat, sa existe un folder
cu melodii mp4 pe care aplicatia sa-l poata gestiona cu semne ale mainii astfel:
    - doua gesturi pentru selectarea unei melodii;
    - un gest pentru a reda melodia selectata;
    - un gest pentru a pune pe pauza melodia redata;
    - un gest pentru a reporni melodia pusa pe pauza;
    - un gest pentru inchiderea melodiei curente, astfel incat sa ajungi din nou in folderul cu
       melodii pentru a putea selecta o alta melodie.
