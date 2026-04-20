import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
data={
    "time(hours)":[0,1,2,3,4,5,6,7,8],
    "without_ampicillin":[0.05,0.08,0.15,0.30,0.60,0.95,1.20,1.35,1.40],
    "with_ampicillin":[0.05,0.06,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
}
df=pd.DataFrame(data)
print(df)
plt.plot(df["time(hours)"],df["without_ampicillin"],label="without antibiotic",marker="o")
plt.plot(df["time(hours)"],df["with_ampicillin"],label="with antibiotic",marker="o")
plt.xlabel("time(hours)")
plt.ylabel("optical density(OD600)")
plt.title("bacterial growth comparison")
plt.legend()
plt.show()
df["difference"]=df["without_ampicillin"]-df["with_ampicillin"]
print(df)
plt.plot(df["time(hours)"],df["difference"], label="Difference in growth")
plt.xlabel("time(hours)")
plt.ylabel("difference")
plt.title("difference in growth")
plt.legend()
plt.show()
x=df[["time(hours)"]]
y_with=df[["with_ampicillin"]]
model_with=LinearRegression()
model_with.fit(x,y_with)
prediction_with=model_with.predict(x)
print("Predictions for 'with_ampicillin':", prediction_with)
x=df[["time(hours)"]]
y_no=df[["without_ampicillin"]]
model_no=LinearRegression()
model_no.fit(x,y_no)
prediction_no=model_no.predict(x)
print("Predictions for 'without_ampicillin':", prediction_no)
future_time = pd.DataFrame([[10]], columns=["time(hours)"])

pred_no_future = model_no.predict(future_time)
pred_with_future = model_with.predict(future_time)

print("Predicted growth WITHOUT antibiotic at 10 hrs:", pred_no_future)
print("Predicted growth WITH antibiotic at 10 hrs:", pred_with_future)
r2_no = r2_score(y_no, prediction_no)
r2_with = r2_score(y_with, prediction_with)

print("R² score (without antibiotic):", r2_no)
print("R² score (with antibiotic):", r2_with)


y_log = np.log(y)


model_exp = LinearRegression()
model_exp.fit(x, y_log)


log_pred = model_exp.predict(x)
exp_pred = np.exp(log_pred)

plt.scatter(x, y, label="Actual Growth")
plt.plot(x, exp_pred, label="Exponential Fit")

plt.xlabel("Time (hours)")
plt.ylabel("OD600")
plt.title("Exponential Growth Model (Without Antibiotic)")
plt.legend()
plt.show()

y_decay = np.log(y_with)

model_decay = LinearRegression()
model_decay.fit(x, y_decay)

log_pred_decay = model_decay.predict(x)
decay_pred = np.exp(log_pred_decay)


plt.figure()
plt.scatter(x, y_with, label="Actual (With Antibiotic)")
plt.plot(x, decay_pred, label="Exponential Decay Fit")

plt.xlabel("Time (hours)")
plt.ylabel("OD600")
plt.title("Exponential Decay Model (With Ampicillin)")
plt.legend()
plt.show()
print("\nInterpretation")
print("the E.coli growth is significantly reduced with the presence of ampicillin.")
print("the difference increases over time showing the effectiveness of the antibiotic.")
print("Linear regression approximates trends, though real growth is exponential/logistic.")
