[![Try the Interactive EV Sales Dashboard!](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?logo=streamlit)](https://kedarnath-bhakta-ev-market-analysis-aosu3srtf52xaaycglcn3n.streamlit.app)

Electric Vehicle Sales by State in India

🚗 Project Overview

This project analyzes electric vehicle (EV) sales data across different states in India. It provides:
- Data cleaning and preprocessing
- Exploratory data analysis and visualization
- Feature engineering
- Machine learning models for sales prediction
- An interactive Streamlit dashboard for data exploration and prediction

---

📁 Project Structure

```
.
├── app/                # Streamlit dashboard app
├── data/
│   ├── raw/            # Original, immutable data
│   └── processed/      # Cleaned and processed data
├── models/             # Saved model pipelines and artifacts
├── notebooks/          # Jupyter notebooks for exploration
├── reports/            # Generated analysis reports and figures
├── src/
│   ├── data_processing/
│   ├── feature_engineering/
│   ├── modeling/
│   └── visualization/
├── requirements.txt
└── README.md
```

---
⚡️ Setup and Installation

1.Clone this repository
   ```sh
   git clone <your-repo-url>
   cd Electric\ vehicle\ sales
   ```

2.(Recommended) Create a conda environment
   ```sh
   conda create -n ev-gpu python=3.10
   conda activate ev-gpu
   ```

3.Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

4.Optional, for LightGBM GPU support)
   ```sh
   conda install -c conda-forge lightgbm
   ```

---

🚀 GPU Support

This project can leverage GPU acceleration for faster model training and inference if you have a compatible NVIDIA GPU and the necessary CUDA drivers/toolkit installed.

-XGBoost:
  The default `pip install xgboost` supports GPU if CUDA is available.
- LightGBM:  
  For GPU support, install via conda:  
  `conda install -c conda-forge lightgbm`

- PyTorch/TensorFlow:
  See their official docs for CUDA-specific install commands.

>Note:
> You must have the appropriate NVIDIA drivers and CUDA toolkit installed for GPU acceleration.  
> If you do not have a GPU, the packages will run on CPU by default.

---

🛠️ Usage

Data Processing & Model Training

1.Process raw data and train the model pipeline:
   ```sh
   python src/modeling/model_trainer.py
   ```
   This will:
   - Clean and preprocess the raw data
   - Train a machine learning pipeline (with preprocessing + model)
   - Save the pipeline to `models/best_pipeline.pkl`

2. (Optional) Explore data in Jupyter notebooks:
   ```sh
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

Run the Streamlit Dashboard

1. Start the app:
   ```sh
   streamlit run app/app.py
   ```

2. Features:
   - Overview: Key metrics and summary
   - Data Explorer: Explore the dataset
   - Sales Analysis: Visualize sales trends and distributions
   - State Comparison: Compare sales across states
   - Predictions: Predict EV sales for custom input (state, vehicle type, month, etc.)

---

 📊 Example Prediction

- Select a state, vehicle category, vehicle type, vehicle class, month, and year in the dashboard.
- Click "Predict Sales" to get the predicted EV sales quantity for your chosen scenario.

---

📦 Requirements

See [`requirements.txt`](requirements.txt) for all dependencies.

Key packages:
- numpy, pandas, matplotlib, seaborn, plotly
- scikit-learn, xgboost, lightgbm
- streamlit

---

 

 🧠 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn, XGBoost, LightGBM
- Streamlit
- Jupyter Notebooks

---

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
Follow thsi page for more Real time Tech Projects 
