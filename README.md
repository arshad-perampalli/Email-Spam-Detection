# Email Spam Detection

a simple Streamlit app that trains a TF-IDF + MultinomialNB classifier on the included SMS/Email spam dataset and provides a UI to test messages.

**Project Structure**
- **Email.py**: Main Streamlit application and training script.
- **spam.csv**: Dataset used to train and evaluate the model.

**Quick Start**

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install required packages:

```bash
pip install streamlit scikit-learn pandas matplotlib seaborn
```

3. Run the app:

```bash
streamlit run Email.py
```

4. Open the URL printed by Streamlit in your browser, enter an email/message in the text area, and click "Check Spam".

**How it works**
- **Data loading**: `spam.csv` is read with `latin-1` encoding and reduced to `label` and `message` columns.
- **Preprocessing**: Labels are mapped to 0 (`ham`) and 1 (`spam`); TF-IDF vectorization (English stop words, max 3000 features) is applied.
- **Model**: `MultinomialNB` is trained on the TF-IDF vectors.
- **Evaluation**: The app shows accuracy and a confusion matrix for the held-out test set.

**Notes & Recommendations**
- The model is trained on every app start — for production, train offline and load a serialized model.
- If you add new data, consider increasing `max_features` in `TfidfVectorizer` and retraining.
- To reproduce results deterministically, `train_test_split` uses `random_state=42`.

## Confusion Matrix

 ![Input Form](https://github.com/Arshad-5068/AML-Task-4-/blob/main/confusion%20matrix.png?raw=true)  


## Dashboard

 ![Input Form](https://github.com/Arshad-5068/AML-Task-4-/blob/main/output.png?raw=true)  
