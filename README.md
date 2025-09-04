# 📚 Multi-stock-with-TGCNs 
This project focuses on building a deep learning model to predict the price trend of a single stock on the next day using a small simulated dataset

# Overview
This project explores stock trend prediction using Graph Neural Networks (GNNs), focusing on the Temporal Graph Convolutional Network (TGCN) and its enhanced variants:
- **Baseline TGCN:** Combines Graph Convolutional Networks (GCN) with Gated Recurrent Units (GRU).
- **Attention-TGCN:** Incorporates self-attention to learn the influence weights between stocks.
- **Feature-Enhanced TGCN:** Adds simulated financial features (P/E, ROE, Beta) to improve prediction performance.

The models are applied to a simplified dataset of 5 representative stocks (AAPL, MSFT, AMZN, GOOGL, META). An specifically, the mainpoint of this project is focusing on predict the APPL stocks on the next day and then make the decision that we shall make a investment

# Contribution
Contributions and feedback are welcomed! If you have any suggestions, improvements, or discover any issues, feel free to submit a pull request or open an issue in the GitHub repository.

# Notebook
I do train those 3 models within in just one notebook. The notebook provides a step-by-step walkthrough of the project:

**1. Data Preprocessing**: Importing stock data (AAPL, MSFT, AMZN, GOOGL, META) and normalization and feature engineering (P/E, ROE, Beta).
**2. Graph Construction**: Building a fully-connected stock relationship graph and creating edge indices for PyTorch Geometric.
**3. Model Implementation**:
- TGCN (Temporal Graph Convolutional Network).
- Attention-TGCN (self-attention mechanism).
- Feature-Enhanced TGCN (with financial indicators).
**4. Training & Evaluation:** Training models with Adam optimizer, then tracking loss and accuracy across epochs.
**5. Prediction:** Predicting AAPL stock trend (up/down) andd generating probability outputs for investment recommendation.

You can open the notebook directly in Jupyter, Google Colab or VS Code to see the workflow, visualizations, and model performance results step by step.

# Dependencies
- Python 3.9+
- PyTorch
- PyTorch Geometric
- Pandas, Numpy, Matplotlib, Scikit-learn
