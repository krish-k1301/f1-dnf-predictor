#  F1 DNF Predictor🏎️

This project predicts whether a Formula 1 driver is likely to **DNF (Did Not Finish)** a race using machine learning.

I’ve always been a huge F1 fan - especially interested in what happens **behind the scenes**: strategy calls, mechanical failures, race conditions, and all the small factors that can completely change an outcome.
This project is my attempt to explore that side of the sport through data.

---

##  What this does

You input a few race/driver details, and the model predicts:

 **DNF or No DNF**

It’s quick, simple, and meant to give a rough idea based on historical patterns.

---

##  How it works

* A trained ML model (`classifier.pkl`) is loaded
* User inputs are processed into the right format
* The model makes a prediction
* The result is displayed instantly

Nothing fancy - just a clean pipeline from input → prediction.

---

##  Tech used

* Python
* scikit-learn
* pandas / numpy
* (your app framework - Streamlit or Flask)

---

##  Project structure

```
f1-dnf-predictor/
├── app.py
├── classifier.pkl
├── requirements.txt
└── README.md
```

---

## Running it locally

Clone the repo:

```
git clone https://github.com/krish-k1301/f1-dnf-predictor.git
cd f1-dnf-predictor
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
python app.py
```

(or `streamlit run app.py` if you're using Streamlit)

---

## Notes about the model

* It’s a basic classification model
* Trained on historical F1-style data
* Output is binary: DNF / No DNF

This isn’t meant to be perfectly accurate - more of an exploration into how unpredictable F1 can be.

---

## What I’d improve next

* Better feature engineering
* More realistic / larger dataset
* UI improvements
* Deployment (so anyone can use it online)

---

## About me

Made by **Krish Kubadia**

I like building things with ML, but more importantly, understanding *why* they work.
Big F1 fan, especially the strategy, chaos, and everything that happens off-camera.

---

## If you found this interesting

Give it a star - or feel free to suggest improvements.
