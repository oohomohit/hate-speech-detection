# Hate Speech Detection in Code-Mixed Hindi-English Content

This repository contains a machine learning model designed to detect hate speech in **code-mixed Hindi-English** content. The model uses a **pretrained transformer-based model** from Hugging Face and is fine-tuned on a custom dataset of social media posts. This project aims to identify hate speech and other offensive content in a mixture of Hindi and English, commonly found in social media platforms.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/hate-speech-detection.git
   cd hate-speech-detection

2. **Create a virtual environment (Optional but recommended):**:
   ```bash
   python -m venv myenv
   source myenv/bin/activate   # On Windows, use `myenv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

## Dataset

The dataset for training and testing the model consists of code-mixed Hindi-English social media posts labeled with either **Hate Speech** or **Non-Hate Speech**. This dataset is used for fine-tuning the model to identify and classify hate speech in the provided input text.

**Note:** If you're looking for the dataset, you may need to collect and preprocess it yourself. We recommend using platforms like Twitter and Reddit to scrape code-mixed posts using `snscrape` or other scraping tools.

## Model

The model is based on transformer-based architectures such as BERT, RoBERTa, or DistilBERT, which have been pretrained and fine-tuned for the task of hate speech detection in code-mixed content.

The model is fine-tuned on your own labeled dataset to accurately detect hate speech from mixed-language text (Hindi + English).

## Pretrained Model

The model utilizes a pretrained language model from Hugging Face and can be loaded through the `transformers` pipeline. The model is then fine-tuned on the collected dataset.

## Usage

To use the trained model for detecting hate speech in code-mixed Hindi-English text, follow these steps:

1.  **Run the model:**

    ```bash
    python detect_hate_speech.py "Your input text here"
    ```

    Replace "Your input text here" with any Hindi-English mixed sentence you want to classify. The model will output whether the sentence contains Hate Speech or not.

2.  **Example:**

    ```bash
    python detect_hate_speech.py "Yeh kya bakwaas hai, sabko maar do!"
    ```

    **Output:**

    ```plaintext
    Hate Speech: Yes
    ```

## Contributing

We welcome contributions! If you want to improve the model, dataset, or anything else in the project, feel free to fork the repository, make changes, and create a pull request.

**How to Contribute:**

1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature-name`).
3.  Make your changes and commit them (`git commit -a "Your commit message"`).
4.  Push your changes (`git push origin feature-name`).
5.  Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
