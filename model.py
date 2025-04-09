import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report 

df = pd.read_csv("cleaned_ebay_deals.csv") 
print("Total rows loaded:", len(df)) 
df.head() 

df = df.dropna(subset=["price", "original_price", "shipping", "discount_percentage"])
print("Rows after cleaning:", len(df)) 

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='discount_percentage', bins=10, kde=True)
plt.title("Discount Percentage Distridution")
plt.xlabel("Percent (%)")
plt.ylabel("Product")
plt.show()

df['low_discount']=df['discount_percentage']<10
df['medium_discount']=(df['discount_percentage']>10) & (df['discount_percentage']<30)
df['high_discount']=df['discount_percentage']>30
df[['price', 'original_price','shipping', 'discount_percentage','low_discount', 'medium_discount', 'high_discount']].head() 

X = df[["price", "original_price","shipping" ]] 
y = df["low_discount","medium_discount","high_discount"] 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42) 

model = LogisticRegression() 
model.fit(X_train, y_train) 

y_pred = model.predict(X_test) 
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred)) 

print(df["low_discount"]).value_counts(normalize=True)
print(df["medium_discount"]).value_counts(normalize=True)
print(df["high_discount"].value_counts(normalize=True)) 


df_major_sample = df["high_discount"].sample(len(df["low_discount"]), random_state=1) 
df_balanced = pd.concat([df_major_sample, df["low_discount"]]) 

X_bal = df_balanced[["price", "original_price", "shipping"]] 
y_bal = df_balanced["low_discount", "medium_discount", "high_discount"] 
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split( X_bal, y_bal, test_size=0.2, random_state=42) 
model_b = LogisticRegression() 
model_b.fit(X_train_b, y_train_b) 
y_pred_b = model_b.predict(X_test_b) 