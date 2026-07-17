# Personality Type Classifier

An AI-powered web application that predicts whether a person is an
**Introvert** or **Extrovert** based on seven behavioural and lifestyle
survey responses.

---

## Project Description

Personality typing has applications in education, career counselling,
team building, and self-awareness. This project trains a Logistic
Regression classifier on a behavioural dataset and wraps it in a clean,
interactive Streamlit interface — allowing anyone to discover their
predicted personality type in under a minute.

---

## About the Machine-Learning Model

| Property | Detail |
|---|---|
| **Algorithm** | Logistic Regression |
| **Selection method** | 5-fold GridSearchCV over `C = (0.1, 1, 10)` and `solver = (lbfgs, saga)` |
| **Optimisation metric** | Accuracy |
| **Preprocessing** | OneHotEncoder (drop=first) for binary categoricals · StandardScaler for numerics |
| **Target classes** | Introvert (0) · Extrovert (1) |
| **Test Accuracy** | ~93 % |
| **Dataset** | 2 900 behavioural survey responses (after cleaning: ~2 400 records) |

The full pipeline (preprocessing + model) is serialised in
`personality_classifier_model.pkl` so that the app can pass raw user inputs
directly without any manual feature engineering.

---

## Application Features

- **Seven intuitive inputs** — sliders for continuous behavioural
  scores and radio buttons for binary yes/no questions.
- **Confidence breakdown** — colour-coded probability bars for both
  Introvert and Extrovert classes.
- **Personality traits panel** — key characteristics of the predicted
  type plus a personalised tip.
- **Input summary expander** — review every submitted answer alongside
  the result.
- **Persistent sidebar** — project overview, how-to-use guide, and
  model details always visible.
- **Production-ready** — cached model loading (`@st.cache_resource`),
  explicit dtype handling to prevent sklearn/pandas compatibility issues,
  and clean error messaging.

---

## Project Structure

```
personality-classifier/
│
├── app.py                             # Streamlit application (main entry point)
├── personality_classifier_model.pkl   # Pre-trained Logistic Regression pipeline
├── personality_dataset.csv            # Original training dataset (reference only)
├── Personality_Classifier.ipynb       # Model-training notebook (reference only)
├── requirements.txt                   # Pinned Python dependencies
└── README.md                          # This file
```

> **Note:** `personality_dataset.csv` and `Personality_Classifier.ipynb` are
> not required at runtime. Only `app.py`, `personality_classifier_model.pkl`,
> and `requirements.txt` are needed for deployment.

---

## Installation & Local Setup

### Prerequisites

- Python **3.11** or **3.12** (recommended)
- `pip` (bundled with Python)
- `git` (optional, for cloning)

### 1 — Clone or download the repository

```bash
git clone https://github.com/your-username/personality-classifier.git
cd personality-classifier
```

Or place `app.py`, `personality_classifier_model.pkl`, and
`requirements.txt` in the same folder.

### 2 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 — Run the application

```bash
streamlit run app.py
```

Streamlit will automatically open `http://localhost:8501` in your browser.

---

## How to Use the Application

1. **Adjust the sliders** for your daily habits — time alone, social event
   attendance, going outside, friends circle size, and posting frequency.
2. **Select Yes / No** for stage fright and whether you feel drained after
   socialising.
3. **Click "Predict My Personality"** — the model runs instantly.
4. **Read your result** — a colour-coded card shows your predicted type
   (Introvert or Extrovert ) with a tagline.
5. **Check the confidence bars** — see the probability the model assigns
   to each class.
6. **Explore your traits** — key personality characteristics and a
   personalised tip appear alongside the result.
7. **Expand the input summary** — verify every value that was submitted.

---

## Assumptions & Limitations

| Item | Detail |
|---|---|
| **Binary classification only** | The model distinguishes only between Introvert and Extrovert. Ambiverts or mixed profiles are not represented. |
| **Self-reported data** | The training dataset is based on survey responses, which may contain bias or inconsistency. |
| **Feature ranges** | Sliders are capped at the ranges observed in the training data (e.g. Time Alone: 0–11 hrs, Friends Circle: 0–15). |
| **No medical validity** | This is not a psychological assessment tool. Results should not be used for clinical or diagnostic purposes. |
| **Model version sensitivity** | The pickle was retrained with scikit-learn 1.9.0. Using a different sklearn version may produce warnings or errors. |

---

## Technologies & Libraries

| Library | Version | Purpose |
|---|---|---|
| [Streamlit](https://streamlit.io) | 1.59.2 | Web application framework |
| [scikit-learn](https://scikit-learn.org) | 1.9.0 | Logistic Regression, preprocessing pipeline |
| [pandas](https://pandas.pydata.org) | 3.0.2 | DataFrame construction for model input |
| [NumPy](https://numpy.org) | 2.4.4 | Numerical operations |
| [joblib](https://joblib.readthedocs.io) | 1.5.3 | Model serialisation / deserialisation |
| [PyArrow](https://arrow.apache.org/docs/python/) | 24.0.0 | pandas 3.x string-dtype backend |

---

## Live Demo

**Live Demo: https://personality-classifier2208.streamlit.app/**

---

## Deploying to Streamlit Community Cloud

1. Push your project folder to a **public GitHub repository** containing:
   - `app.py`
   - `personality_classifier_model.pkl`
   - `requirements.txt`
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select your repository and branch, set the main file
   path to `app.py`.
4. Click **Deploy** — Streamlit installs the requirements automatically.
5. Copy the generated URL and update the **Live Demo** link above.

---

*Developed by: Farooq Hassnain Sheikh.*
