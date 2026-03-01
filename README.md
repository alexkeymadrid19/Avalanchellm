# Avalanche Risk Classification

## Project Overview
This project aims to classify the risk of avalanches using deep learning techniques, specifically leveraging the capabilities of PyTorch. The goal is to develop a model that can predict the likelihood of avalanches under varying conditions based on historical data and environmental factors.

## Installation Instructions
To set up the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/alexkeymadrid19/Avalanchellm.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Avalanchellm
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guidelines
To use the model for avalanche risk classification:
1. Prepare your dataset containing historical avalanche data and relevant features.
2. Train the model using:
   ```bash
   python train.py --data_path your_data.csv
   ```
3. Evaluate the model on new data:
   ```bash
   python evaluate.py --data_path new_data.csv
   ```

4. To make predictions:
   ```bash
   python predict.py --input your_input_data.csv
   ```

Ensure to modify the parameters in the scripts as per your needs. 

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.