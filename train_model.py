import streamlit as st
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detection System")

# -----------------------------
# Load Dataset
# -----------------------------
try:
    dataset = pd.read_csv("emails.csv", encoding="latin-1")

    # Display column names for debugging
    st.write("Dataset Columns:", dataset.columns.tolist())

    # Handle different dataset formats
    if "text" in dataset.columns and "spam" in dataset.columns:
        X = dataset["text"]
        y = dataset["spam"]

    elif "v1" in dataset.columns and "v2" in dataset.columns:
        dataset = dataset[["v1", "v2"]]
        dataset.columns = ["spam", "text"]

        dataset["spam"] = dataset["spam"].map({
            "ham": 0,
            "spam": 1
        })

        X = dataset["text"]
        y = dataset["spam"]

    elif "Message" in dataset.columns and "Category" in dataset.columns:
        X = dataset["Message"]

        y = dataset["Category"].map({
            "ham": 0,
            "spam": 1
        })

    else:
        st.error(
            f"Unsupported dataset format.\n\n"
            f"Columns found: {dataset.columns.tolist()}"
        )
        st.stop()

except FileNotFoundError:
    st.error("emails.csv file not found.")
    st.stop()

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10
)

# -----------------------------
# Vectorization
# -----------------------------
vect = CountVectorizer(stop_words="english")

X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()

model.fit(X_train_vect, y_train)

# -----------------------------
# Model Accuracy
# -----------------------------
predictions = model.predict(X_test_vect)

accuracy = accuracy_score(y_test, predictions)

st.success(f"Model Accuracy: {accuracy:.2%}")

# -----------------------------
# Spam Detection Function
# -----------------------------
def classify(message):
    vector = vect.transform([message])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    return prediction, probability


# -----------------------------
# User Input
# -----------------------------
st.subheader("Check an Email")

user_input = st.text_area(
    "Enter email text:",
    placeholder="Type or paste an email here..."
)

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        prediction, probability = classify(user_input)

        if prediction == 1:
            st.error("SPAM EMAIL")
            

            st.write(
                f"Confidence: {probability[1] * 100:.2f}%"
            )

        else:
            st.success("HAM (Not Spam)")

            st.write(
                f"Confidence: {probability[0] * 100:.2f}%"
            )

# -----------------------------
# Dataset Preview
# -----------------------------
with st.expander("View Dataset"):

    st.write("Shape:", dataset.shape)

    st.dataframe(dataset.head())

    st.write("Class Distribution:")

    st.write(y.value_counts())