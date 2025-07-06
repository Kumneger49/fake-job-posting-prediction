# Fake Job Posting Detection

A machine learning project that uses DeBERTa-v3-base to detect fraudulent job postings using natural language processing techniques.

## Project Overview

This project implements a complete pipeline for detecting fake job postings:
1. **Data Processing**: Comprehensive data cleaning, feature engineering, and class balancing
2. **Model Training**: Fine-tuned DeBERTa-v3-base transformer model
3. **API Development**: REST API for model inference (planned)
4. **UI Development**: User interface for job posting analysis (planned)

## Features

- **Advanced NLP**: Uses DeBERTa-v3-base transformer model
- **Class Imbalance Handling**: SMOTE oversampling for balanced training
- **Comprehensive Data Processing**: Missing value handling, feature engineering, encoding
- **Proper Evaluation**: Train/validation/test separation with no data leakage
- **Production Ready**: Model saved and ready for deployment

## Model Performance

### Final Results
- **F1-Score**: 35.92% on test set
- **Recall**: 97.63% (catches almost all fraudulent jobs)
- **Precision**: 22.01% (some false positives, requires human review)
- **Accuracy**: 34.28% (expected for imbalanced dataset)

### Training Progress
- **Epoch 1**: F1 = 32.32% (learning phase)
- **Epoch 2**: F1 = 34.22% (improving)
- **Epoch 3**: F1 = 34.97% (converged)

## Project Structure

```
fake-job-posting-prediction/
├── data/
│   ├── data_processing.py      # Data preprocessing pipeline
│   ├── data_processing.log     # Processing logs
│   ├── train_data.csv          # Training data (SMOTE applied)
│   ├── val_data.csv            # Validation data
│   ├── test_data.csv           # Test data
│   └── processed_fake_job_postings.csv  # Combined processed data
├── model/
│   ├── train_deberta.py        # Model training script
│   ├── model_training.log      # Training logs
│   ├── final_results.md        # Complete results documentation
│   └── deberta_best_model/     # Best trained model
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fake-job-posting-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install transformers torch datasets scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
   ```

## Usage

### Data Processing
```bash
cd data
python data_processing.py
```

### Model Training
```bash
cd model
python train_deberta.py
```

## Technical Details

### Model Configuration
- **Model**: microsoft/deberta-v3-base
- **Parameters**: 184M
- **Max sequence length**: 256
- **Batch size**: 8
- **Learning rate**: 2e-5
- **Epochs**: 3

### Data Processing Pipeline
1. **Missing value handling** (drop columns/rows with >50% missing)
2. **Feature selection** (remove identifiers, constants, highly correlated)
3. **Feature engineering** (text length, keyword flags, count features)
4. **Categorical encoding** (LabelEncoder with unseen category handling)
5. **Numerical scaling** (StandardScaler)
6. **Train/Test split** (80/20)
7. **Train/Validation split** (80/20 of training data)
8. **SMOTE oversampling** (applied ONLY to training data)

### Dataset Statistics
- **Original**: ~18K samples (94% non-fraudulent, 6% fraudulent)
- **Training**: 18,730 samples (perfectly balanced with SMOTE)
- **Validation**: 2,859 samples (original distribution)
- **Test**: 3,573 samples (original distribution)

## Challenges Solved

1. **Class Imbalance**: Resolved using SMOTE on training data only
2. **Data Leakage**: Prevented with proper train/validation/test separation
3. **MPS Compatibility**: Resolved by switching to CPU training
4. **Binary Label Conversion**: Handled non-binary processed values
5. **Zero Division Warnings**: Resolved with proper metric handling

## Next Steps

1. **API Development**: Create REST API for model inference
2. **UI Development**: Build user interface for job posting analysis
3. **Deployment**: Deploy model for production use
4. **Monitoring**: Implement performance monitoring and model updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Microsoft for the DeBERTa model
- Hugging Face for the transformers library
- The original dataset providers 