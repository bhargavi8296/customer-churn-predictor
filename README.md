# Customer Churn Predictor

A machine learning web application that predicts whether a telecom customer is likely to leave the company’s services.

## Project Overview

Customer churn occurs when a customer stops using a company’s products or services.

This application analyzes customer details and predicts:

- **Churn: Yes** — The customer is likely to leave.
- **Churn: No** — The customer is likely to continue using the service.

It can help telecom companies identify customers at risk and take retention actions such as offering discounts, improving support or recommending better plans.

## Features

- Simple and user-friendly Streamlit interface
- Predicts whether a customer is likely to churn
- Uses the most relevant customer details
- Handles categorical and numerical data
- Uses a trained machine learning pipeline
- Displays an easy-to-understand prediction result

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Workflow

1. Loaded and explored the telecom customer churn dataset.
2. Removed unnecessary columns such as the customer ID.
3. Cleaned blank and missing values.
4. Converted `TotalCharges` into numerical data.
5. Selected the most useful customer features.
6. Split the dataset into training and testing data.
7. Preprocessed numerical and categorical columns.
8. Trained a machine learning classification model.
9. Evaluated the model using unseen test data.
10. Saved the complete pipeline using Joblib.
11. Connected the trained model to a Streamlit application.

## Input Features

The application uses 10 important customer details:

1. Senior citizen status
2. Customer tenure
3. Internet service
4. Online security
5. Technical support
6. Contract type
7. Paperless billing
8. Payment method
9. Monthly charges
10. Total charges

## Project Structure

```text
customer-churn-predictor/
├── app.py
├── churn_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

> If your model file has a different name, replace `churn_model.pkl` with its actual filename.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/bhargvai8296/customer-churn-predictor.git
cd customer-churn-predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For Windows Command Prompt:

```bash
venv\Scripts\activate
```

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

For macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will usually open at:

```text
http://localhost:8501
```

## Requirements

The `requirements.txt` file contains:

```text
streamlit
pandas
scikit-learn==1.6.1
joblib==1.5.3
```

The Scikit-learn version used to run the application should match the version used to train and save the model. A version mismatch can cause errors involving `_RemainderColsList`.

## Business Use

If a customer is predicted to churn, the telecom company can:

- Provide a personalized discount
- Recommend a more suitable plan
- Improve technical support
- Offer benefits for a long-term contract
- Contact the customer before cancellation

## Important Note

This application is an educational machine learning project. Its prediction depends on the training dataset and should not be considered a guaranteed business decision.

## Future Improvements

- Display churn probability
- Add model-performance charts
- Explain the factors behind each prediction
- Compare multiple machine learning algorithms
- Retrain the model with newer customer data

## Live Application

Try the deployed application here:

🔗 [Customer Churn Predictor](https://customer-churn-predictor-inrzsbgl4qmhydcknpoart.streamlit.app/)

## Author

**Bhargavi Goyal**

- GitHub: [bhargvai8296](https://github.com/bhargvai8296)
