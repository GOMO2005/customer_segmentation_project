# 🧑‍💼 Customer Segmentation Project (RFM + KMeans Clustering)

## 📌 Project Overview
This project performs **customer segmentation** using behavioral data from sales transactions.  
The goal is to group customers based on their purchasing behavior using **RFM analysis (Recency, Frequency, Monetary)** and **K-Means clustering**.

This helps businesses identify:
- High-value customers
- Loyal customers
- At-risk customers
- Low-engagement customers

---

## 📊 Dataset
The dataset contains synthetic customer transaction data with the following columns:

- CustomerID
- OrderID
- OrderDate
- ProductCategory
- Quantity
- UnitPrice
- TotalAmount
- Age
- Gender
- City

---

## ⚙️ Technologies Used
- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🧠 Methodology

### 1. Data Preprocessing
- Converted date columns to datetime
- Created TotalAmount feature
- Cleaned and structured dataset

### 2. RFM Feature Engineering
- **Recency**: Days since last purchase
- **Frequency**: Number of orders per customer
- **Monetary**: Total spending per customer

### 3. Scaling
- StandardScaler applied to normalize features

### 4. Clustering
- KMeans algorithm used to segment customers
- Optimal clusters selected manually (k=3)

### 5. Analysis
- Each cluster analyzed based on RFM values
- Business insights derived from customer behavior

---



