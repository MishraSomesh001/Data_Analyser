# Data Analyzer Project

This project is a web-based application for analyzing and visualizing data. It allows users to upload datasets, perform initial data analysis, preprocess the data, generate visualizations, and train a simple machine learning model.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

The main goal of this project is to provide an easy-to-use interface for data analysis and visualization. Users can upload their datasets, explore summary statistics, visualize data distributions, and train a basic machine learning model.

## Features

- Upload and analyze datasets in CSV format.
- Perform initial data analysis and preprocessing.
- Generate summary statistics and visualizations.
- Train and evaluate a simple machine learning model.
- Visualize model predictions.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- Flask
- Pandas
- Matplotlib
- Seaborn
- Torch
- Scikit-learn
- Dask

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/data-analyzer.git
    ```
2. Change to the project directory:
    ```bash
    cd data-analyzer
    ```
3. Install the required packages:
    ```bash
    pip install Flask pandas matplotlib seaborn torch scikit-learn dask
    ```

## Usage

1. Run the Flask application:
    ```bash
    python app.py
    ```
2. Open your web browser and navigate to `http://127.0.0.1:5000`.
3. Upload a dataset in CSV format.
4. Analyze the dataset to view summary statistics and visualizations.
5. Train the machine learning model and view the results.

## File Structure
```data-analyzer/
├── app.py
├── templates/
│ ├── index.html
│ ├── analysis.html
├── static/
│ ├── actual_vs_pred.png
│ ├── heatmap.png
│ ├── hist.png
├── uploads/
│ ├── (uploaded files)
├── requirements.txt
└── README.md
```


## Technologies Used

- **Flask**: Web framework used to build the web interface.
- **Pandas**: Data manipulation and analysis library.
- **Matplotlib**: Plotting library for creating static, animated, and interactive visualizations.
- **Seaborn**: Data visualization library based on Matplotlib.
- **Torch**: Machine learning library for building and training neural networks.
- **Scikit-learn**: Machine learning library for model training and evaluation.
- **Dask**: Parallel computing library for larger-than-memory data collections.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact:

- Team Sharingan





