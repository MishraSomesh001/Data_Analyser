from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Use a non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return redirect(url_for('analyze', filename=file.filename))

@app.route('/analyze/<filename>', methods=['POST', 'GET'])
def analyze(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    data = pd.read_csv(filepath)
    
    # Performing initial analysis
    summary_stats = data.describe(include='all').to_html()
    
    # Handling missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].fillna('None', inplace=True)
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        else:
            data[col].fillna(data[col].mean(), inplace=True)
    
    # Generate correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    heatmap_path = os.path.join('static', 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    
    # Generate histograms
    data.hist(bins=30, figsize=(20, 15))
    hist_path = os.path.join('static', 'histograms.png')
    plt.savefig(hist_path)
    plt.close()
    
    # Feature selection based on correlation with target
    target_column = data.columns[-1]  # Assuming the target column is the last one
    correlations = data.corr()[target_column].sort_values(ascending=False)
    
    # Select features with a correlation threshold
    correlation_threshold = 0.1
    selected_features = correlations[abs(correlations) > correlation_threshold].index.tolist()
    
    # Ensure target column is included
    if target_column not in selected_features:
        selected_features.append(target_column)
    
    if len(selected_features) > 1:
        data = data[selected_features]
        X = data.drop(columns=[target_column]).values
        y = data[target_column].values
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted (MSE: {mse:.2f})")
        plt.grid(True)
        plot_path = os.path.join('static', 'actual_vs_pred.png')
        plt.savefig(plot_path)
        plt.close()
        
        return render_template('analysis.html', summary_stats=summary_stats, heatmap_path=heatmap_path, hist_path=hist_path, plot_path=plot_path)
    else:
        return render_template('analysis.html', summary_stats=summary_stats, heatmap_path=heatmap_path, hist_path=hist_path, error_message="No valid features selected for modeling.")

if __name__ == '__main__':
    app.run(debug=True)
   
# from flask import Flask, request, render_template, redirect, url_for
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Use a non-interactive backend for matplotlib
# import matplotlib
# matplotlib.use('Agg')

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(filepath)
#         return redirect(url_for('analyze', filename=file.filename))

# @app.route('/analyze/<filename>')
# def analyze(filename):
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     data = pd.read_csv(filepath)
    
#     # Perform initial analysis
#     summary_stats = data.describe(include='all').to_html()
    
#     # Handle missing values
#     for col in data.columns:
#         if data[col].dtype == 'object':
#             # Fill missing values in categorical columns with 'None'
#             data[col].fillna('None', inplace=True)
#             # Encode categorical columns
#             le = LabelEncoder()
#             data[col] = le.fit_transform(data[col])
#         else:
#             # Fill missing values in numerical columns with the mean
#             data[col].fillna(data[col].mean(), inplace=True)
    
#     # Generate correlation heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
#     heatmap_path = os.path.join('static', 'heatmap.png')
#     plt.savefig(heatmap_path)
#     plt.close()
    
#     # Generate histograms for each column
#     data.hist(bins=30, figsize=(20, 15))
#     hist_path = os.path.join('static', 'histograms.png')
#     plt.savefig(hist_path)
#     plt.close()
    
#     # Prepare data for PyTorch model
#     target_column = data.columns[-1]  # Assuming the target column is the last one
#     X = data.drop(columns=[target_column]).values
#     y = data[target_column].values
    
#     # Standardize features
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
    
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Convert to PyTorch tensors
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     X_test = torch.tensor(X_test, dtype=torch.float32)
#     y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
#     y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
#     # Define PyTorch model
#     model = nn.Sequential(
#         nn.Linear(X_train.shape[1], 128),
#         nn.ReLU(),
#         nn.Linear(128, 64),
#         nn.ReLU(),
#         nn.Linear(64, 1)
#     )
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
    
#     # Train the model
#     num_epochs = 50  # Reduce the number of epochs to speed up training
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()
        
#     # Evaluate the model
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(X_test)
#         mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
    
#     # Plot actual vs predicted values
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.5)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
#     plt.xlabel("Actual")
#     plt.ylabel("Predicted")
#     plt.title(f"Actual vs Predicted (MSE: {mse:.2f})")
#     plt.grid(True)
#     plot_path = os.path.join('static', 'actual_vs_pred.png')
#     plt.savefig(plot_path)
#     plt.close()
    
#     return render_template('analysis.html', summary_stats=summary_stats, heatmap_path=heatmap_path, hist_path=hist_path, plot_path=plot_path)

# if __name__ == '__main__':
#     app.run(debug=True)
