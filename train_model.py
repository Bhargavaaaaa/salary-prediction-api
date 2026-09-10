import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "Experience": [1,2,3,4,5,6,7,8,9,10],
    "Projects": [1,2,2,3,3,4,5,5,6,7],
    "Certification": [0,0,1,1,1,2,2,2,3,3],
    "Salary": [25000,29000,34000,39000,43000,49000,56000,62000,70000,78000]
})

X=df[["Experience","Projects","Certification"]]
Y=df["Salary"]
model=LinearRegression()
model.fit(X,Y)
joblib.dump(model,"salary_prediction.pkl")
print("Model saved Successfully")