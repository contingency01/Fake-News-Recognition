# Fake News Recognition (R)

Multinomial Naive Bayes–based text classification project for detecting fake vs. real news claims, implemented in R using a Jupyter Notebook workflow.

Repository: https://github.com/contingency01/Fake-News-Recognition

---

## Overview

This project implements a Multinomial Naive Bayes (MNB) classifier for fake news recognition.  
It covers:

- Text preprocessing
- Train/validation split
- Model training with Laplace smoothing
- Log-probability scoring
- Prediction on validation and test data

The implementation is fully written in R and structured inside a Jupyter Notebook.

---

## Project Structure

```
Fake-News-Recognition/
│
├── FakeNewsRecognition.ipynb   # Main R notebook (full pipeline)
│
└── data/
    └── original/
        ├── train.csv           # Training data (Labels + Text)
        └── test.csv            # Test data (no Labels)
```

---

## Dataset

The training dataset includes at least the following columns:

- `Labels` – Class labels (e.g., False, True, Half-True, etc.)
- `Text` – The claim or statement text
- `Text_Tag` (if included)

The test dataset contains text without labels for prediction.

If replacing the dataset, ensure column names remain consistent or update the notebook accordingly.

---

## Methodology

### 1. Data Splitting
- 80/20 Train/Validation split
- Reproducible using:
  ```r
  set.seed(42)
  ```

### 2. Text Preprocessing
Typical NLP steps include:
- Lowercasing
- Tokenization
- Stopword removal
- Stemming (SnowballC)
- Term frequency computation

Common R packages used:
- tidytext
- dplyr
- tm
- SnowballC
- tidyr
- readr
- ggplot2

### 3. Multinomial Naive Bayes

The classifier:

- Computes prior probabilities:
  P(class)

- Computes likelihoods with Laplace smoothing:
  P(word | class)

- Uses log-space scoring to avoid numerical underflow:
  log P(class | document)

- Predicts the class with maximum posterior probability.

---

## Installation & Setup

### Option 1 — Run in Jupyter

1. Clone the repository:

   ```bash
   git clone https://github.com/contingency01/Fake-News-Recognition.git
   cd Fake-News-Recognition
   ```

2. Install required R packages:

   ```r
   install.packages(c(
     "ggplot2",
     "tidytext",
     "dplyr",
     "tm",
     "tidyr",
     "SnowballC",
     "readr"
   ))
   ```

3. Install and register IRkernel (if not already installed):

   ```r
   install.packages("IRkernel")
   IRkernel::installspec()
   ```

4. Open `FakeNewsRecognition.ipynb` in Jupyter and run all cells.

---

### Option 2 — Run in RStudio

- Open the notebook directly (if supported), or
- Convert to R script / RMarkdown and execute step by step.

---

## Reproducibility

- Fixed random seed (`set.seed(42)`)
- Results may vary slightly depending on R version and package versions.

For better reproducibility, consider adding:
- `renv` lockfile
- A dedicated `requirements` setup script

---

## Future Improvements

Possible extensions:

- Add evaluation metrics (accuracy, precision, recall, F1)
- Add confusion matrix visualization
- Create standalone `run.R` script
- Add cross-validation
- Package into reusable R module
- Add model comparison (e.g., Logistic Regression, SVM)

---

## License

No license file is currently included.

---

## Author

Barış Kalfa & Yusuf Kenan Şafak
Developed as an academic/ML text classification project using R and Multinomial Naive Bayes.
