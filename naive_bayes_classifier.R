# Multinomial Naive Bayes Fake News Classifier
#
# This script implements a simple multinomial naive Bayes classifier for
# the Kaggle fake news dataset used in this project.  It follows the
# pipeline that is partially demonstrated in the accompanying notebook
# and finishes the missing steps so that the entire workflow can be
# executed from a single script.

library(tidyverse)
library(tidytext)
library(tm)

# -------------------------------
# Utility functions
# -------------------------------

# Tokenize text, remove punctuation, numbers and stop words.
# Numbers are replaced by the token <number> unless they include a
# leading $ or trailing % sign.
tokenize_and_clean <- function(df) {
  df %>%
    mutate(Text = as.character(Text)) %>%
    unnest_tokens(word, Text, drop = FALSE) %>%
    mutate(word = gsub("[[:punct:]]", "", word)) %>%
    mutate(word = ifelse(grepl("^\\$[0-9]+|[0-9]+%$", word),
                         word,
                         gsub("[[:digit:]]+", "<number>", word))) %>%
    anti_join(stop_words, by = "word") %>%
    filter(word != "") %>%
    filter(nchar(word) > 1)
}

# Create a sparse document-term matrix from tokenised data
create_dtm <- function(tokens) {
  if (!all(c("doc_id", "word") %in% colnames(tokens))) {
    stop("Data must contain 'doc_id' and 'word' columns")
  }

  word_counts <- tokens %>%
    count(doc_id, word)

  word_counts %>%
    cast_dtm(document = doc_id, term = word, value = n)
}

# Ensure a document-term matrix has a specific set of columns.  Any
# missing terms are added as empty columns so that matrices can be
# aligned without subsetting errors when some terms only appear in the
# train or test set.
align_dtm <- function(dtm, terms) {
  cur_terms <- colnames(dtm)
  missing   <- setdiff(terms, cur_terms)

  if (length(missing) > 0) {
    # create an empty simple_triplet_matrix for the missing columns
    empty <- slam::simple_triplet_matrix(i = integer(0),
                                         j = integer(0),
                                         v = numeric(0),
                                         nrow = nrow(dtm),
                                         ncol = length(missing),
                                         dimnames = list(Docs = rownames(dtm),
                                                         Terms = missing))
    dtm <- cbind(dtm, empty)
  }

  dtm[, terms]
}

# Train a multinomial naive Bayes model.  The function also evaluates the
# model on the provided validation matrix.
train_naive_bayes <- function(train_matrix, train_labels,
                              validation_matrix, validation_labels) {
  train_labels <- as.factor(train_labels)
  validation_labels <- as.factor(validation_labels)
  classes <- levels(train_labels)

  # Prior probabilities
  priors <- table(train_labels) / length(train_labels)

  # Likelihoods with Laplace smoothing
  likelihoods <- list()
  for (class in classes) {
    class_rows <- train_matrix[train_labels == class, ]
    word_totals <- colSums(class_rows) + 1
    likelihoods[[class]] <- word_totals / sum(word_totals)
  }

  # Prediction helper
  predict_internal <- function(matrix) {
    preds <- character(nrow(matrix))
    for (i in seq_len(nrow(matrix))) {
      doc <- matrix[i, ]
      log_probs <- sapply(classes, function(class) {
        likelihood_vec <- likelihoods[[class]]
        epsilon <- 1e-10
        likelihood_safe <- sapply(colnames(matrix), function(w) {
          if (w %in% names(likelihood_vec)) likelihood_vec[[w]] else epsilon
        })
        sum(doc * log(likelihood_safe)) + log(priors[class])
      })
      preds[i] <- classes[which.max(log_probs)]
    }
    preds
  }

  validation_pred <- predict_internal(validation_matrix)
  conf <- table(Predicted = validation_pred, Actual = validation_labels)
  acc <- sum(diag(conf)) / sum(conf)

  print("Confusion Matrix:")
  print(conf)
  print(paste("Validation Accuracy:", round(acc * 100, 2), "%"))

  list(priors = priors, likelihoods = likelihoods,
       predict = predict_internal)
}

# -------------------------------
# Main workflow
# -------------------------------

# Load the Kaggle dataset
train_df <- read.csv("data/train.csv")
test_df  <- read.csv("data/test.csv")

# Train/validation split (80/20)
set.seed(123)
train_idx <- sample(seq_len(nrow(train_df)), size = 0.8 * nrow(train_df))
train_set <- train_df[train_idx, ]
validation_set <- train_df[-train_idx, ]

# Add document id column
train_set$doc_id <- seq_len(nrow(train_set))
validation_set$doc_id <- seq_len(nrow(validation_set))
test_df$doc_id <- seq_len(nrow(test_df))

# Tokenisation and cleaning
tokens_train <- tokenize_and_clean(train_set)
tokens_valid <- tokenize_and_clean(validation_set)
tokens_test  <- tokenize_and_clean(test_df)

# Build document-term matrices
train_matrix <- create_dtm(tokens_train)
validation_matrix <- create_dtm(tokens_valid)
test_matrix <- create_dtm(tokens_test)

# Ensure matrices have the same columns.  Terms appearing only in the
# test set need to be added to the training and validation matrices so
# that subsetting does not fail.
all_terms <- union(colnames(train_matrix), colnames(test_matrix))
train_matrix <- align_dtm(train_matrix, all_terms)
validation_matrix <- align_dtm(validation_matrix, all_terms)
test_matrix <- align_dtm(test_matrix, all_terms)

# Train the model
model <- train_naive_bayes(as.matrix(train_matrix),
                           train_set$Labels,
                           as.matrix(validation_matrix),
                           validation_set$Labels)

# Predict on the held-out Kaggle test set
predictions <- model$predict(as.matrix(test_matrix))

# Save the predictions
write.csv(data.frame(Id = test_df$doc_id, Label = predictions),
          file = "kaggle_test_predictions.csv",
          row.names = FALSE)

# --------------------------------------------------------------
# Optional: apply to a second dataset with binary labels
# --------------------------------------------------------------
if (file.exists("data/news2.csv")) {
  news2 <- read.csv("data/news2.csv")
  news2$doc_id <- seq_len(nrow(news2))
  tokens_news2 <- tokenize_and_clean(news2)
  news2_matrix <- create_dtm(tokens_news2)
  # Align terms with training matrix
  news2_matrix <- align_dtm(news2_matrix, all_terms)
  news2_pred <- model$predict(as.matrix(news2_matrix))
  conf2 <- table(Predicted = news2_pred, Actual = news2$label)
  acc2 <- sum(diag(conf2)) / sum(conf2)
  print("----- Results on second dataset -----")
  print(conf2)
  print(paste("Accuracy:", round(acc2 * 100, 2), "%"))
} else {
  message("Secondary dataset data/news2.csv not found; skipping")
}

