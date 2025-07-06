import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# 1. Load the dataset
def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

# 2. Visualize missing values and class distribution
def visualize_data(df):
    """Visualize missing values and class distribution."""
    # Visualize missing values heatmap
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()
    # Visualize missing values per column
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        plt.figure(figsize=(12,6))
        missing_counts.sort_values(ascending=False).plot(kind='bar')
        plt.title('Missing Values per Column')
        plt.ylabel('Count')
        plt.show()
    # Visualize class distribution
    df['fraudulent'].value_counts().plot(kind='bar')
    plt.title('Class Distribution: Fraudulent vs. Real')
    plt.xlabel('Fraudulent')
    plt.ylabel('Count')
    plt.show()
    # Visualize top categorical features
    top_cats = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in top_cats:
        if col in df.columns:
            plt.figure(figsize=(10,4))
            df[col].value_counts(dropna=False).head(10).plot(kind='bar')
            plt.title(f'Top 10 {col} values')
            plt.ylabel('Count')
            plt.show()
    # Visualize numeric features
    num_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    for col in num_cols:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            df[col].value_counts().plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.ylabel('Count')
            plt.show()

# 3. Handle missing values
def handle_missing_values(df, col_thresh=50, row_thresh=50):
    """Drop columns and rows with too many missing values."""
    # Drop columns with more than col_thresh% missing values
    missing_percent = df.isnull().mean() * 100
    cols_to_drop = missing_percent[missing_percent > col_thresh].index
    df = df.drop(columns=cols_to_drop)
    # Drop rows with more than row_thresh% missing values
    row_missing_percent = df.isnull().mean(axis=1) * 100
    rows_to_drop = row_missing_percent[row_missing_percent > row_thresh].index
    df = df.drop(index=rows_to_drop)
    return df

# 4. Feature selection
def feature_selection(df):
    """Drop identifier, constant, and highly correlated columns."""
    # Drop identifier columns
    if 'job_id' in df.columns:
        df = df.drop(columns=['job_id'])
    # Drop columns with only one unique value
    to_drop = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=to_drop)
    # Drop highly correlated columns (correlation > 0.95)
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr().abs()
    upper = corr_matrix.where((corr_matrix > 0.95) & (corr_matrix < 1))
    high_corr_cols = [column for column in upper.columns if any(upper[column].notnull())]
    df = df.drop(columns=high_corr_cols)
    return df

# 5. Feature engineering
def feature_engineering(df):
    """Create new features from text and structure."""
    # Text length features
    df['desc_length'] = df['description'].fillna('').apply(len)
    df['req_length'] = df['requirements'].fillna('').apply(len)
    df['profile_length'] = df['company_profile'].fillna('').apply(len)
    # Keyword flags
    keywords = ['work from home', 'quick money', 'no experience', 'urgent', 'immediate start']
    for kw in keywords:
        df[f'kw_{kw.replace(" ", "_")}'] = df['description'].fillna('').str.contains(kw, case=False).astype(int)
    # Count features
    df['num_benefits'] = df['benefits'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)
    df['num_requirements'] = df['requirements'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)
    # Missingness indicators
    df['desc_missing'] = df['description'].isnull().astype(int)
    df['profile_missing'] = df['company_profile'].isnull().astype(int)
    return df

# 6. Split data into features and target
def split_features_target(df):
    """Split DataFrame into features (X) and target (Y)."""
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    return X, Y

# 7. Split into train and test sets
def split_train_test(X, Y, test_size=0.2, random_state=42):
    """Split features and target into training and test sets."""
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

# 8. Encode categorical features
def encode_categoricals(x_train, x_test):
    """Label encode categorical columns, handling unseen labels."""
    categorical_cols = x_train.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        x_train[col] = x_train[col].astype(str).fillna('Unknown')
        x_test[col] = x_test[col].astype(str).fillna('Unknown')
        le.fit(list(x_train[col]) + ['Unknown'])
        x_train[col] = le.transform(x_train[col])
        x_test[col] = x_test[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        x_test[col] = le.transform(x_test[col])
        le_dict[col] = le
    return x_train, x_test, le_dict

# 9. Scale numerical features
def scale_numericals(x_train, x_test):
    """Standard scale numerical columns."""
    numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
    x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])
    return x_train, x_test, scaler

# 10. Oversample the minority class
def oversample_smote(x_train, y_train):
    """Apply SMOTE to balance the classes in the training set."""
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    return X_train_resampled, y_train_resampled

# --- Main pipeline ---
def main():
    # Load data
    df = load_data("./fake_job_postings.csv")
    # Visualize data
    # visualize_data(df)
    # Handle missing values
    df = handle_missing_values(df)
    # Feature selection
    df = feature_selection(df)
    # Feature engineering
    df = feature_engineering(df)
    # Split features and target
    X, Y = split_features_target(df)
    print(X.shape)
    # Split into train and test (80/20)
    x_train, x_test, y_train, y_test = split_train_test(X, Y)
    print(x_train.shape)
    # Encode categoricals
    x_train, x_test, le_dict = encode_categoricals(x_train, x_test)
    # Scale numericals
    x_train, x_test, scaler = scale_numericals(x_train, x_test)
    
    # Split training data into train and validation (80/20 of training data)
    x_train_final, x_val, y_train_final, y_val = split_train_test(x_train, y_train, test_size=0.2, random_state=42)
    
    # Apply SMOTE only to the final training data
    X_train_resampled, y_train_resampled = oversample_smote(x_train_final, y_train_final)
    print("After SMOTE - Training data class distribution:")
    print(y_train_resampled.value_counts())
    
    # Convert back to DataFrames
    train_df = pd.DataFrame(X_train_resampled, columns=x_train_final.columns)
    train_df['fraudulent'] = y_train_resampled
    
    val_df = pd.DataFrame(x_val, columns=x_val.columns)
    val_df['fraudulent'] = y_val
    
    test_df = pd.DataFrame(x_test, columns=x_test.columns)
    test_df['fraudulent'] = y_test
    
    # Save the three datasets separately
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    print(f"Saved datasets:")
    print(f"Train: {train_df.shape} (SMOTE applied)")
    print(f"Validation: {val_df.shape} (no SMOTE)")
    print(f"Test: {test_df.shape} (no SMOTE)")
    
    # Also save the processed full dataset for backward compatibility
    final_df = pd.concat([train_df, val_df, test_df], axis=0)
    final_df.to_csv('processed_fake_job_postings.csv', index=False)

if __name__ == "__main__":
    main()
 

