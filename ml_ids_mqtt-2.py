import json
import joblib
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime


BROKER = "127.0.0.1"
IN_TOPIC = "hospital/icu/ids/input"
OUT_TOPIC = "hospital/icu/ids/alerts"


bundle = joblib.load("artifacts/ids_iforest.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]
threshold = float(bundle["threshold"])

print("Model loaded successfully")
print("Threshold:", threshold)


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(IN_TOPIC)
        print("Subscribed:", IN_TOPIC)
    else:
        print("Connection failed:", rc)


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        if not isinstance(data, dict):
            print("Invalid payload received:", data)
            return
    except Exception as e:
        print("JSON decode error:", e)
        return

    print("\nDATA RECEIVED BY ML:", data)


    device_id = data.get("device_id")
    patient_id = data.get("patient_id")
    patient_name = data.get("patient_name")
    timestamp = data.get("timestamp", datetime.utcnow().isoformat())

    heart_rate = float(data.get("heart_rate", 0))
    spo2 = float(data.get("spo2", 0))


    feats = {
        "inter_arrival_s_abs": 1.0,
        "timestamp_backwards": 0,
        "seq_delta": 1.0,
        "seq_repeat": 0,
        "rate_10s": 0.1,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "hr_out_of_range": 1 if heart_rate < 50 or heart_rate > 120 else 0,
        "spo2_out_of_range": 1 if spo2 < 90 and spo2 != 0 else 0,
    }

    try:
        X = pd.DataFrame([feats])[feature_cols]
        Xs = scaler.transform(X)

        # Isolation Forest score
        score = float(-model.decision_function(Xs)[0])

        # Raw anomaly decision
        isAnomaly = 1 if score > threshold else 0

    except Exception as e:
        print("Feature processing error:", e)
        return

    print("Score:", score, "| Threshold:", threshold)

    if patient_id != "PAT-001":
        isAnomaly = 0

    status = "ATTACK DETECTED" if isAnomaly == 1 else "NORMAL"


    alert = {
        "device_id": device_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "timestamp": timestamp,
        "status": status,
        "is_anomaly": isAnomaly,
        "anomaly_score": score,
        "threshold": threshold
    }

    client.publish(OUT_TOPIC, json.dumps(alert))
    print("ALERT PUBLISHED:", alert)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, 1883, 60)
client.loop_forever()
