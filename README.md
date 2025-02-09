# Entity Framing in Narratives: Multi-Label Role Classification of Named Entities 

## XLM-RoBERTa-base + Attention Layer + 2 Output Heads (with Focal Loss) Model
![Model Architecture](model_architecture.png)

* **Dataset_EN_PT:** This folder contains the text data for the articles. Each text file is named with the corresponding article ID.
* **train.csv:** Contains training data with columns `article_id`, `main_role`, `fine_grained_roles`.
* **test.csv:** Contains test data with the same columns as `train.csv`.
* **val.csv:** Contains validation data with the same columns as `train.csv`.
* **results:** Stores the trained model outputs, including metrics, predictions, and the model itself.
* **model.ipynb:** The Jupyter notebook containing the project code, including data loading, model training, evaluation, and visualization.

## Model and Approach

This project uses the following:

* **XLM-RoBERTa:** A powerful multilingual language model for text processing.
* **Focal Loss:** Addresses class imbalance during training.
* **Attention Mechanism:** Focuses on relevant parts of the text.
* **Evaluation Metrics:** Accuracy, precision, recall, F1-score, and exact match ratio (EMR) are used to assess model performance.

## Running the Project

1. **Set up Google Colab:** Create a new Colab notebook and upload the project files.
2. **Install Dependencies:** Install necessary libraries like `transformers`, `torch`, `sklearn`, and `pandas`.
3. **Mount Google Drive:** Mount your Google Drive to access the data.
4. **Run the Notebook:** Execute the cells in the `model.ipynb` file sequentially. This will load the data, train the model, evaluate its performance, and visualize the results.

## Results

The results are saved in the `results` folder. You can find the evaluation metrics and predicted roles there. The notebook also displays plots of metrics during training.

## Future Work

Possible improvements and extensions for the project:

* **Hyperparameter Tuning:** Experiment with different hyperparameters to optimize model performance.
* **Data Augmentation:** Increase the training data size to improve robustness.
* **Ensemble Methods:** Combine multiple models for better generalization.
* **Interpretability:** Investigate methods to understand the model's predictions.

## Acknowledgements

* XLM-RoBERTa: Developed by Facebook AI
* Hugging Face Transformers: A library for working with transformer models
