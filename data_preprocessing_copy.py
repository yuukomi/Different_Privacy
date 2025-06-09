import pandas as pd
import numpy as np
import re

class LabelEncoder:
    """Custom implementation of LabelEncoder"""
    def __init__(self):
        self.classes_ = None
        self.label_dict = None
        
    def fit(self, y):
        """Fit label encoder to a list of values"""
        unique_values = sorted(set(y))
        self.classes_ = unique_values
        self.label_dict = {val: idx for idx, val in enumerate(unique_values)}
        return self
        
    def transform(self, y):
        """Transform labels to normalized encoding"""
        return np.array([self.label_dict[val] for val in y])
        
    def fit_transform(self, y):
        """Fit and transform in one step"""
        self.fit(y)
        return self.transform(y)
        
    def inverse_transform(self, y):
        """Convert normalized labels back to original encoding"""
        return np.array([self.classes_[val] for val in y])

def extract_numeric(value):
    """Extract numeric value from string containing numbers and text"""
    if pd.isna(value):
        return np.nan
    matches = re.findall(r'(\d+\.?\d*)', str(value))
    return float(matches[0]) if matches else np.nan

def clean_power(value):
    """Extract bhp value from power string"""
    if pd.isna(value):
        return np.nan
    matches = re.findall(r'(\d+\.?\d*)\s*bhp', str(value))
    return float(matches[0]) if matches else extract_numeric(value)

def clean_torque(value):
    """Extract Nm value from torque string"""
    if pd.isna(value):
        return np.nan
    matches = re.findall(r'(\d+\.?\d*)\s*Nm', str(value))
    return float(matches[0]) if matches else extract_numeric(value)

def clean_engine(value):
    """Extract cc value from engine string"""
    if pd.isna(value):
        return np.nan
    matches = re.findall(r'(\d+\.?\d*)\s*cc', str(value))
    return float(matches[0]) if matches else extract_numeric(value)

def preprocess_data(df):
    """Preprocess the car dataset"""
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Clean numeric columns
    if 'Engine' in df.columns:
        df['Engine'] = df['Engine'].apply(clean_engine)
    if 'Max Power' in df.columns:
        df['Max Power'] = df['Max Power'].apply(clean_power)
    if 'Max Torque' in df.columns:
        df['Max Torque'] = df['Max Torque'].apply(clean_torque)
    
    # Convert Year to age
    if 'Year' in df.columns:
        current_year = 2024
        df['Age'] = current_year - df['Year']
    
    # Define all possible categorical and numeric columns
    all_categorical = ['Make', 'Fuel Type', 'Transmission', 'Drivetrain', 'Color', 'Owner']
    all_numeric = ['Age', 'Kilometer', 'Engine', 'Max Power', 'Max Torque',
                  'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity']
    
    # Filter to only include columns that exist in the dataset
    categorical_cols = [col for col in all_categorical if col in df.columns]
    numeric_features = [col for col in all_numeric if col in df.columns]
    
    # Label encode categorical columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values before encoding
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Fill missing values with median for numeric columns
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Combine features
    features = numeric_features + categorical_cols
    
    # Return preprocessed features and target
    if 'Price' not in df.columns:
        raise ValueError("Price column not found in the dataset")
        
    X = df[features]
    y = df['Price']
    
    return X, y, encoders

def load_and_preprocess():
    """Load and preprocess both train and test datasets"""
    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Preprocess train and test data
    X_train, y_train, encoders = preprocess_data(train_df)
    X_test, y_test, _ = preprocess_data(test_df)
    
    return X_train, y_train, X_test, y_test, encoders

if __name__ == "__main__":
    # Test the preprocessing
    X_train, y_train, X_test, y_test, encoders = load_and_preprocess()
    
    print("\nTraining Data Shape:", X_train.shape)
    print("Test Data Shape:", X_test.shape)
    print("\nFeature Names:", X_train.columns.tolist())
    
    # Print sample of preprocessed data
    print("\nSample of preprocessed training data:")
    print(X_train.head())
    print("\nSample of target values:")
    print(y_train.head()) 