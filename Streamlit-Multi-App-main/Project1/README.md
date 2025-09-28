# Perceptron Model Project

This project explores the **Perceptron algorithm** — one of the simplest types of neural networks — and demonstrates how to process data, train a perceptron, tune its parameters, and visualize results using **Streamlit**.

---

## 🔹 What is a Perceptron?

A perceptron is a **binary classifier** that maps an input vector **x** into a binary output using a **linear prediction function**.
It consists of just one neuron:

1. **Inputs & Weights**

   * Each input has an associated **weight** that determines its importance.
   * Weights are adjusted during training to minimize errors.

2. **Bias**

   * A constant offset that shifts the decision boundary (e.g., moves a separating line up/down).

3. **Weighted Sum**

   * The perceptron computes:

     $$
     net = w_0x_0 + w_1x_1 + \dots + w_nx_n
     $$

4. **Activation Function**

   * Applies the weighted sum + bias, compares to a threshold, and outputs **0 or 1**.

5. **Learning Rule**

   * The perceptron updates its weights based on **prediction errors**, improving decision boundaries over time.

---

## 🔹 Project Goals

* Understand the **full pipeline**: data preprocessing → model training → evaluation → tuning.
* Provide a **Streamlit app** so users can upload their own datasets and experiment with perceptrons interactively.
* Make the workflow modular, organized, and reusable.

---

## 🔹 Data

* Users can **upload their own dataset** (CSV or Excel).
* If no dataset is uploaded, the app falls back to a **default dataset**.

---

## 🔹 Code Structure

This project is written in a **modular style** — each file has a clear responsibility:

* **`data_loader.py`** → Loads uploaded datasets or provides a default dataset.
* **`data_cleaner.py`** → Cleans data (drops missing values). Label Encoding & standardization are applied in `main_app.py`.
* **`train_test_split.py`** → Prepares features/targets and splits into train/test sets.
* **`perceptron.py`** → Creates and trains the perceptron model.
* **`model_evaluation.py`** → Evaluates model accuracy, precision, recall, and more.
* **`hyperparameter_tuning.py`** → Runs parameter search (learning rate, iterations) and retrains model.
* **`visualization.py`** → Generates plots (confusion matrix, ROC curve, feature importance, etc.).
* **`main_app.py`** → The **Streamlit app**, tying everything together with an interactive UI.

---

## 🔹 Features of the App

* ✔️ Upload any dataset and train a perceptron model
* ✔️ Automatic preprocessing (cleaning, encoding, splitting)
* ✔️ Adjustable hyperparameters (iterations, learning rate)
* ✔️ Hyperparameter search for optimal settings
* ✔️ Visual analysis (confusion matrix, ROC curve, feature importance)
* ✔️ Easy-to-use **web interface with Streamlit**

Great question 👍 — for a GitHub README, citations should go in a **dedicated section at the end** so they don’t clutter your project explanation but are still visible and properly credit the dataset authors.

I’d suggest adding a **“📖 References / Citation”** section after your existing content. Example:

---

## 📖 References

**Dataset**

* Rice (Cammeo and Osmancik) Dataset

  * Source: [Murat Koklu Datasets](https://www.muratkoklu.com/datasets/) | [UCI Repository](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)
  * Authors:

    * İlKay Cinar (Selcuk University, Konya, Turkey)
    * Murat Koklu (Selcuk University, Konya, Turkey)
  * Abstract: 3810 rice grains were imaged and 7 morphological features were extracted for classification.

**Citation**
Cinar, I. and Koklu, M. (2019). *Classification of Rice Varieties Using Artificial Intelligence Methods.* International Journal of Intelligent Systems and Applications in Engineering, vol.7, no.3 (Sep. 2019), pp.188–194. [https://doi.org/10.18201/ijisae.2019355381](https://doi.org/10.18201/ijisae.2019355381)
