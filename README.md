# AirportWaitingTimePredictor

Projekat za predmet **Računarska inteligencija** – predikcija vremena čekanja na aerodromima (pasoška / bezbednosna kontrola).

## Opis
Na osnovu podataka o aerodromu, datumu/vremenu i broju putnika, model predviđa prosečno vreme čekanja u minutima ili kategoriju čekanja (kratko/srednje/dugo).

## Korišćene tehnologije
- Python 3.10+
- pandas, scikit-learn, XGBoost
- matplotlib / seaborn za vizualizaciju
- (Opcionalno) TensorFlow ili PyTorch za neuronske mreže

## Struktura
- `data/` – Dataset CSV fajlovi (nije deo repozitorijuma)
- `notebooks/` – Jupyter notebook-i za EDA
- `src/` – Izvorni kod (pretprocesiranje, treniranje modela, evaluacija)
- `requirements.txt` – Spisak zavisnosti

## Pokretanje
1. Klonirati repozitorijum i ući u folder:
   ```bash
   git clone <repo-url>
   cd AirportWaitingTimePredictor
   ```

2. Kreirati i aktivirati virtuelno okruženje:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Instalirati zavisnosti:
   ```bash
   pip install -r requirements.txt
   ```

4. Dodati dataset u data/ i pokreni analizu u Jupyter-u ili main.py.

## Autor

Vladimir Zdravković