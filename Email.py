import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Email Spam Detection", layout="centered")

st.title("Email Spam Detection")
st.write("Enter an email message below to check whether it is **Spam** or **Not Spam**.")

@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

st.subheader("Check Your Email")

user_input = st.text_area(
    "Enter Email Text",
    placeholder="Type or paste the email message here...",
    height=150
)

if st.button("Check Spam"):
    if user_input.strip() == "":
        st.warning("Please enter an email message.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("This email is **SPAM**")
        else:
            st.success("This email is **NOT SPAM**")


accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"],
    ax=ax
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

st.markdown("---")
st.caption("Machine Learning Project | Email Spam Detection")
