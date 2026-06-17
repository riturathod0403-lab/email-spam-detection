import joblib

# Load trained model
model = joblib.load("model.pkl")

while True:
    sms = input("\nEnter SMS message (or type 'exit'): ")

    if sms.lower() == "exit":
        break

    prediction = model.predict([sms])[0]

    if prediction == 1:
        print("Spam Message")
    else:
        print("Not Spam (Ham)")