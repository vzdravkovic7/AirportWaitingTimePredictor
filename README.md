# Airport Waiting Time Predictor

Projekat iz predmeta **Računarska inteligencija**  
Autor: **Vladimir Zdravković**  
Tema: *Predikcija vremena čekanja na aerodromima (pasoška / bezbednosna kontrola)*

---

## Opis projekta

Cilj projekta je da se na osnovu podataka o letu, vremenu, sezoni, broju putnika i tipu aerodroma, predvidi:

- **Prosečno vreme čekanja (u minutima)** → regresioni modeli  
- **Kategorija čekanja** (*short / medium / long*) → klasifikacioni modeli  

Projekat obuhvata kompletan **machine learning pipeline**:  
pripremu podataka, inženjering karakteristika, treniranje više modela, evaluaciju, vizualizacije i API servis za predikciju.

---

## Korišćene tehnologije

- **Python 3.10+**
- **pandas**, **scikit-learn**, **xgboost**
- **matplotlib**, **seaborn** – vizualizacije
- **imbalanced-learn (SMOTE)** – balansiranje klasa
- **FastAPI** – REST API serving
- **joblib / json** – serijalizacija modela i metrika

---

## Struktura projekta
```
AirportWaitingTimePredictor/
│
├── data/ # Dataset CSV fajlovi (lokalno)
├── notebooks/ # EDA i eksperimenti
├── src/
│ ├── preprocessing.py # Pretprocesiranje podataka
│ ├── features.py # Feature engineering i encoding
│ ├── models.py # Definicije modela (regresija i klasifikacija)
│ ├── train_regression.py # Treniranje regresionih modela
│ ├── train_classification.py # Treniranje klasifikacionih modela
│ ├── visualizations.py # Plot funkcije i prikaz rezultata
│ ├── main.py # Glavni pipeline (train + evaluacija)
│ └── serve.py # API endpointi (/predict, /predict_class, /predict_class_batch)
│
├── requirements.txt
└── README.md
```

---

## Pokretanje projekta

### Instalacija

```bash
git clone <repo-url>
cd AirportWaitingTimePredictor

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### Pokretanje treniranja

Pokreće se ceo flow za regresiju i klasifikaciju:
```bash
python -m src.main
```
Ovaj proces:

izvršava preprocessing i SMOTE balansiranje,

trenira Linear, Random Forest, XGBoost i MLP modele,

evaluira performanse na validation/test setu,

čuva metrike u results.json.

Tipično trajanje: ~6–7 minuta

### Pokretanje CLI servisa
```bash
python -m src.serve --csv src/examples/sample_input.csv
```

### Pokretanje API servisa
```bash
python -m src.serve --api
```
| Endpoint               | Opis                                           | Primer                                                |
| ---------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| `/predict`             | Predikcija vremena čekanja (minuti)            | `POST {"airport":"JFK","passengers":1200,...}`        |
| `/predict_class`       | Klasifikacija u kategorije (short/medium/long) | `POST {"airport":"LAX","hour":15,...}`                |
| `/predict_batch`       | Batch predikcije za više redova                | `POST [{"airport":"JFK",...}, {"airport":"ORD",...}]` |
| `/predict_class_batch` | Batch predikcije za više redova (klasifikacija)| `POST [{"airport":"BOS",...}, {"airport":"LAX",...}]` |

### Vizualizacije
Prolaskom kroz flow u main.py fajlu, funkcije za vizualizaciju prikazuju:

Grafike preko podataka iz results.json fajla

Bar chart uporednih metrika (MAE, RMSE, R²)

Tačnost klasifikacionih modela (Accuracy, Precision, Recall, F1)

Scatter plot Predicted vs Actual vrednosti

Distribucije reziduala

---

## Modeli koji su implementirani

### Regresioni

Linear Regression (baseline)

Random Forest Regressor

XGBoost Regressor

MLPRegressor (neuronska mreža)

### Klasifikacioni

Logistic Regression (baseline)

Random Forest Classifier

MLPClassifier (neuronska mreža)

---

## Evaluacija performansi

Rezultati su automatski sačuvani u results.json, primer:
```bash
{
  "Random Forest Regressor": {
    "validation": {
      "MAE": 2.84,
      "RMSE": 3.95,
      "R2": 0.87
    }
  },
  "MLP Classifier": {
    "validation": {
      "Accuracy": 0.91,
      "Precision": 0.90,
      "Recall": 0.89,
      "F1-score": 0.89
    }
  }
}
```
---

## Napomene o performansama

Treniranje traje 6–7 minuta na prosečnoj mašini (CPU).

MLP modeli koriste manji broj slojeva i neurona za brže izvođenje.

Dataset koristi srednji sample radi balansa između tačnosti i vremena.

---

## Reference

Scikit-learn: https://scikit-learn.org

XGBoost docs: https://xgboost.readthedocs.io

FastAPI: https://fastapi.tiangolo.com

---

## Status projekta

Završeno – Projekat pokriva sve tačke specifikacije:

```bash
preprocessing i feature engineering

regresija i klasifikacija

evaluacija i vizualizacije

neuronske mreže (MLP showcase)

API servis za predikciju
```
