# Fake Job Posting Detection

A machine learning project that uses DeBERTa-v3-base to detect fraudulent job postings using natural language processing techniques.

## Project Overview

This project implements a complete pipeline for detecting fake job postings:
1. **Data Processing**: Comprehensive data cleaning, feature engineering, and class balancing
2. **Model Training**: Fine-tuned DeBERTa-v3-base transformer model
3. **API Development**: REST API for model inference (**implemented**)
4. **UI Development**: User interface for job posting analysis (**implemented**)

## Features

- **Advanced NLP**: Uses DeBERTa-v3-base transformer model
- **Class Imbalance Handling**: SMOTE oversampling for balanced training
- **Comprehensive Data Processing**: Missing value handling, feature engineering, encoding
- **Proper Evaluation**: Train/validation/test separation with no data leakage
- **Production Ready**: Model saved and ready for deployment
- **REST API**: FastAPI backend for real-time inference
- **Modern UI**: React frontend for user-friendly predictions

## Model Performance

### Final Results
- **F1-Score**: 35.92% on test set
- **Recall**: 97.63% (catches almost all fraudulent jobs)
- **Precision**: 22.01% (some false positives, requires human review)
- **Accuracy**: 34.28% (expected for imbalanced dataset)

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
├── api/
│   ├── main.py                 # FastAPI backend
│   └── ...                     # API code and logs
├── ui/
│   ├── src/                    # React frontend source code
│   └── ...                     # Frontend assets
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fake-job-posting-prediction
   ```

2. **Set up Python environment (recommended: pyenv + virtualenv)**
   ```bash
   pyenv install 3.10.14
   pyenv virtualenv 3.10.14 hf310env
   pyenv activate hf310env
   ```

3. **Install backend dependencies**
   ```bash
   pip install -r requirements.txt
   # or
   pip install fastapi uvicorn transformers torch sentencepiece pandas numpy scikit-learn imbalanced-learn pydantic loguru datasets kaggle
   ```

4. **Install frontend dependencies**
   ```bash
   cd ui
   npm install
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

### Start the API Backend
```bash
cd api
uvicorn main:app --reload
```

### Start the Frontend UI
```bash
cd ui
npm start
```

### API Usage
- **POST /predict**: Send a job posting text and receive a prediction (fraudulent/legitimate) and probability.
- **GET /**: Health check endpoint.

### Frontend Usage
- Access the React app at [http://localhost:3000](http://localhost:3000)
- Enter a job posting and click "Predict" to see results.

## Demo
- The app provides real-time predictions for job postings via a modern web UI.
- Example: Enter "this is a high paying job, no experience needed and it is remote." and see the prediction.

## Troubleshooting & Tips
- **CORS errors**: Ensure CORS middleware is enabled in FastAPI and only one app instance is created.
- **Python version**: Use Python 3.10.x for best compatibility with ML libraries.
- **Environment issues**: Always activate your virtualenv before running backend commands.
- **Model always predicts one class**: Check your model, data balance, and retrain if needed.

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
6. **CORS/API Integration**: Fixed CORS and API integration issues for seamless frontend-backend communication

## Next Steps

- [x] Data processing pipeline
- [x] Model training and evaluation
- [x] API development and deployment
- [x] Frontend UI integration
- [x] Troubleshooting and bugfixes
- [x] Project documentation

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