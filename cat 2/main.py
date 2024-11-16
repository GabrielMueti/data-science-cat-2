import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_csv('student.csv')
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Enrollment_status'] = data['Enrollment_status'].map({'Not Enrolled': 0, 'Enrolled': 1})
data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data = data.dropna()

# Define support need based on GPA (example threshold)
data['Support_Needed'] = data['GPA'] < 2.5

# Define features and target
features = ['GPA', 'Age', 'Gender']
target = 'Enrollment_status'
X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Get prediction probabilities and calculate ROC curve metrics
y_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot Enrollment and Support Needs with ROC Curve in one figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Adjust figure size for balance

# Enrollment and Support Needs bar chart with explicit color assignment
grouped_data = data.groupby(['Enrollment_status', 'Support_Needed']).size().unstack(fill_value=0)

# Define the colors for each support status explicitly
grouped_data.plot(
    kind='bar',
    stacked=True,
    color={False: 'skyblue', True: 'salmon'},  # Explicitly set colors for each support status
    width=0.8,  # Set the width of bars to 0.8 for smaller, more readable bars
    ax=ax1
)

ax1.set_xlabel("Enrollment Status (0 = Not Enrolled, 1 = Enrolled)")
ax1.set_ylabel("Number of Students")
ax1.set_title("Enrolled  and need for additional Support")
ax1.set_xticklabels(["Unenrolled", "Enrolled"], rotation=0)
ax1.legend(["No Additional Support Needed", "Additional Support Needed"])

# ROC curve
ax2.plot(fpr * 100, tpr * 100, color='black', label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 100], [0, 100], color='gray', linestyle='--')
ax2.set_xlabel("False Positive Rate (%)")
ax2.set_ylabel("True Positive Rate (%)")
ax2.set_title("ROC Curve for Enrollment Prediction")
ax2.legend(loc="lower right")

plt.tight_layout()
plt.show()
