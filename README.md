# Luna AI

Luna AI is an open-source AI model developed by Luna OpenLabs for text classification tasks. Leveraging the BERT architecture, this model is designed to classify text into predefined categories efficiently and accurately.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Saving and Loading the Model](#saving-and-loading-the-model)
  - [Testing the Model](#testing-the-model)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Text Classification**: Classify text data into various categories.
- **Built on BERT**: Utilizes the powerful BERT architecture for natural language understanding.
- **Easy Integration**: Works seamlessly with Hugging Face Transformers library.
- **Open Source**: Available for anyone to use, modify, and distribute.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Clone the Repository

To clone the repository, run the following command:

bash
git clone https://github.com/LunaOpenLabs/Luna-Ai.git

### Install Requirements
To install the required packages, use:

bash
pip install -r requirements.txt

### Dataset
Luna AI requires a dataset in CSV format with two columns: text and label. An example dataset is provided in the data/ directory.

### Example Dataset Structure
Here’s an example of how the dataset should be structured:

csv
text,label
"I love this product!",1
"This is the worst experience.",0

### Usage
Training the Model

To train the model, execute the following command:

bash
python training/train.py

This command will load the dataset from data/dataset.csv and initiate the training process.

### Saving and Loading the Model
After training, save the trained model using:

bash
python save_model.py

This will save the model and its tokenizer in the luna_ai_model directory.

### Testing the Model
To test the model with sample inputs, you can use the test_model.py script. Modify the sample_text variable in the script as needed.

### Run the test script with:

bash
python test_model.py

### Example Output
The model will output the predicted class for the provided sample text.

### Contributing
Contributions are welcome! If you have suggestions, improvements, or bug fixes, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m 'Add some feature').
Push to the branch (git push origin feature-branch).
Open a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Contact
For questions, suggestions, or feedback, feel free to contact the Luna OpenLabs team at [lunaopenlabs@outlook.com].
