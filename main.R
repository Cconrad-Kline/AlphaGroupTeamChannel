# Title: Alpha Team Group Project
# Author: Naomi Bonnin (Alpha Team)
# University: University of Maryland Global Campus
# Course: DATA630 - Machine Learning
# Assignment: Group Project
# Professor: Dr. Ami Gates
# Due Date: August 3rd, 2022
# Description: This R script is used to perform sentiment analysis on movie 
#              summaries.

# Helper function to install and load packages
usePackage <- function(p) {
  if (!is.element(p, installed.packages()[, 1]))
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}

# Helper function to get the counts of the genres
genreCount <- function(string) {
  splits <- strsplit(string, ",")
  return(length(splits[[1]]))
}

# Install and load the appropriate packages
usePackage("keras")
usePackage("archive")
usePackage("readr")
usePackage("tidyr")
usePackage("tm")
usePackage('cld2')
usePackage('tokenizers')
usePackage('collections')
usePackage('pbapply')
usePackage('caret')
usePackage('textstem')
usePackage('wordcloud')
usePackage('lime')
usePackage("remotes")
remotes::install_github("aldomann/encoder")
usePackage('encoder')
# Install and bootstrap keras
# Change this line to your conda enviornment with keras/tensorflow
use_condaenv("tensorflow-metal")
# Use this line if you want to create a new temporary env with the needed packages
#install_keras()

# Set the working directory
setwd("~/Data630GroupProject")

download.file(
  "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz",
  "MovieSummaries.tar.gz"
)

# Read in the file
summaries <-
  read_tsv(
    archive_read("MovieSummaries.tar.gz", file = "MovieSummaries/plot_summaries.txt"),
    col_types = cols(),
    col_names = c(
      'wikipediaId',
      'summary'
    )
  )

metadata <-
  read_tsv(
    archive_read("MovieSummaries.tar.gz", file = "MovieSummaries/movie.metadata.tsv"),
    col_types = cols(),
    col_names = c(
      'wikipediaId',
      'frebaseId',
      'movieName',
      'releaseDate',
      'revenue',
      'runtime',
      'languages',
      'countries',
      'genres'
    )
  )

data <- merge(x=summaries, y=metadata, on='wikipediaId')

# Clean up languages
data$languages <- gsub("\\{|\\}", "", data$languages)
data$languages <- gsub('\\"', "", data$languages)
data$languages <- gsub('\\/m\\/.{3,8}:', '', data$languages)
data$languages <- gsub('\\\\u.{4}', '', data$languages)

# Clean up countries
data$countries <- gsub("\\{|\\}", "", data$countries)
data$countries <- gsub('\\"', "", data$countries)
data$countries <- gsub('\\/m\\/.{3,8}:', '', data$countries)
data$countries <- gsub('\\\\u.{4}', '', data$countries)

# Clean up the genres
data$genres <- gsub("\\{|\\}", "", data$genres)
data$genres <- gsub('\\"', "", data$genres)
data$genres <- gsub('\\/m\\/.{3,8}:', '', data$genres)
data$genres <- gsub('\\\\u.{4}', '', data$genres)
data$genres <- pbsapply(data$genres, FUN=tolower)

# Dropping useless columns
data$frebaseId <- NULL

# Clean up the summaries
data$summary <- pbsapply(data$summary, FUN = stripWhitespace)
data$summary <- pbsapply(data$summary, FUN = removeNumbers)
data$summary <- pbsapply(data$summary, FUN = tolower)
data$summary <-
  pbsapply(data$summary, FUN = removeWords, stopwords::stopwords('en'))
data$summary <-
  pbsapply(data$summary, FUN = textstem::lemmatize_strings)
data$summary <- pbsapply(data$summary, FUN = removePunctuation)

# Get counts of the genres and summarize and plot them
data$genreCounts <- pbsapply(data$genres, FUN = genreCount)
summary(data$genreCounts)
hist(
  data$genreCounts,
  xlab = "Number of Genres",
  main = "Histogram of Number of Genres",
  col = brewer.pal(8, "Dark2")
)
# Remove genreCounts as we dont need it
data$genreCounts <- NULL

# Get counts of the languages and summarize and plot them
data$languageCounts <- pbsapply(data$languages, FUN = genreCount)
summary(data$languageCounts)
hist(
  data$languageCounts,
  xlab = "Number of Genres",
  main = "Histogram of Number of Languages",
  col = brewer.pal(8, "Dark2")
)
# Remove genreCounts as we dont need it
data$languageCounts <- NULL

# Get counts of the languages and summarize and plot them
data$countryCounts <- pbsapply(data$languages, FUN = genreCount)
summary(data$languageCounts)
hist(
  data$languageCounts,
  xlab = "Number of Genres",
  main = "Histogram of Number of Languages",
  col = brewer.pal(8, "Dark2")
)
# Remove genreCounts as we dont need it
data$languageCounts <- NULL

# Add column for summary length, plot it, then remove it
data$summaryLength <- pbsapply(data$summary, FUN = count_words)
summary(data$summaryLength)
hist(
  data$summaryLength,
  xlab = "Word Count",
  main = "Histogram of Word Count for Summaries",
  col = brewer.pal(8, "Dark2")
)
# This is used to determine the optimal lengths of the inputs to the neural network (256 words)
plot(
  ecdf(data$summaryLength),
  xlab = "Word Count",
  ylab = "Percent",
  main = "CDF Plot for Word Count",
  col = brewer.pal(8, "Dark2")
)

# Drop books with summaries less than 10 words as they do not represent enough data
data <- data[data$summaryLength > 10, ]
data$summaryLength <- NULL

# Define a text tokenizer
summary_tokenizer <- text_tokenizer(num_words = 2 ** 10)

# Fit the tokenizer
fit_text_tokenizer(summary_tokenizer, data$summary)

# Apply the tokenizer to the summaries
data$tokens <- texts_to_sequences(summary_tokenizer, data$summary)

# Pad the sequences and limit the length
data$tokens <-
  pad_sequences(
    data$tokens,
    maxlen = 2 ** 8,
    padding = 'pre',
    truncating = 'post'
  )

# Get Word Frequency Counts
counts <- summary_tokenizer$word_counts
counts <- counts[order(unlist(counts), decreasing = TRUE)]

# Plot Word Frequencies
barplot(unlist(counts[1:10]), xlab="Word", ylab="Occurences", main="Top 10 Words by Frequency Across All Documents", col=brewer.pal(8, "Dark2"))

# Create a wordcloud
wordcloud(
  names(counts),
  unname(unlist(counts)),
  max.words = 100,
  random.order = FALSE,
  rot.per = 0.35,
  colors = brewer.pal(8, "Dark2"),
  scale = c(2.5, 0.25)
)

# Drop extra unneeded columns
data$wikipediaId <- NULL
data$frebaseId <- NULL
data$movieName <- NULL
data$releaseDate <- NULL
data$revenue <- NULL
data$runtime <- NULL
data$languages <- NULL
data$countries <- NULL

# Set the seed
set.seed(1993)

# Encode the tags (figure this guy out)
label_tokenizer <- text_tokenizer(split=", ")
fit_text_tokenizer(label_tokenizer, data$genres)

data$labels <- texts_to_matrix(label_tokenizer, data$genres, mode='binary')

# Perform the train test split
# Create a 80:20 train test split using stratified sampling
split_indicies <-
  createDataPartition(data$genres, p = 0.8, list = FALSE)
data.train <- data[split_indicies, ]
data.test <- data[-split_indicies, ]


# Define some hyperparameters
parameters <- ordered_dict()
parameters$set('filters', 2 ** 5)
parameters$set('kernel_size', 2 ** 2 + 1)
parameters$set('pool_size', 2 ** 2)
parameters$set('lstm_units', 2 ** 6)
parameters$set('input_dim', summary_tokenizer$num_words + 1)
parameters$set('output_dim', 2 ** 6)
parameters$set('input_length', 2 ** 8)
parameters$set('output_length', length(label_tokenizer$word_counts) + 1)
parameters$set('loss', loss_categorical_crossentropy())
parameters$set('optimizer', 'nadam')
parameters$set('metrics', c('accuracy', 'mse'))
parameters$set('batch_size', 2 ** 5)
parameters$set('epochs', 2 ** 4)
parameters$set('dropout', 0.5)
parameters$set('strides', 1)
parameters$set('validation_split', 0.2)
parameters$set(
  'callbacks',
  c(
    callback_early_stopping(
      monitor = "val_loss",
      min_delta = 0.0001,
      patience = 4,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      patience = 3,
      factor = 0.1,
      min_delta = 0.0001,
      monitor = "val_accuracy"
    ),
    callback_tensorboard(
      paste("logs/", Sys.time(), sep = ""),
      write_images = TRUE,
      histogram_freq = 1,
      embeddings_freq = 1,
      write_grads = TRUE
    )
  )
)

# Function to create and train the model
summaryLSTM <- function(x_train, y_train, params) {
  model <- keras_model_sequential()
  model %>%
    layer_embedding(
      params$get('input_dim'),
      params$get('output_dim'),
      input_length = params$get('input_length')
    ) %>%
    layer_dropout(params$get('dropout')) %>%
    layer_conv_1d(
      filters = params$get('filters'),
      kernel_size = params$get('kernel_size'),
      padding = 'valid',
      strides = params$get('strides'),
    ) %>%
    layer_activation_relu() %>%
    layer_max_pooling_1d(pool_size = params$get('pool_size')) %>%
    layer_lstm(params$get('lstm_units'), activation = NULL) %>%
    layer_activation('tanh') %>%
    layer_dense(params$get('output_length')) %>%
    layer_activation("sigmoid")
  
  model %>% compile(
    loss = params$get('loss'),
    optimizer = params$get('optimizer'),
    metrics = params$get('metrics')
  )
  
  summary(model)
  
  tensorboard("logs")
  
  model %>% keras::fit(
    x_train,
    y_train,
    batch_size = params$get('batch_size'),
    epochs = params$get('epochs'),
    validation_split = params$get('validation_split'),
    callbacks = params$get('callbacks')
  )
  
  return(model)
}

# Actually train the model
model <-
  summaryLSTM(data.train$tokens, data.train$labels, parameters)

# Save the model
save_model_hdf5(model, "model.hdf5")

# Evaluate the model
keras::evaluate(model, data.test$tokens, data.test$labels)

# Create a confusion matrix
preds <- model %>% predict(data.test$tokens) %>% round()
confusion_matrix <-
  caret::confusionMatrix(data = as.factor(preds),
                         reference = as.factor(data.test$labels))

# Plot the confusion matrix as a fourfoldplot
fourfoldplot(confusion_matrix$table, main = "Confusion Matrix on Test Data")

# Create a function to preprocess new text
clean_text <- function(text) {
  text <- stripWhitespace(text)
  text <- removeNumbers(text)
  text <- tolower(text)
  text <- removeWords(text, stopwords('en'))
  text <- textstem::lemmatize_strings(text)
  text <- removePunctuation(text)
  return(text)
}

# Create function to get tokens from new text
get_embedding_explination <- function(text) {
  text <- clean_text(text)
  summary_tokenizer %>% fit_text_tokenizer(text)
  text_to_seq <- texts_to_sequences(summary_tokenizer, text)
  sentences <- text_to_seq %>% pad_sequences(maxlen=2**9, padding='pre', truncating = 'post')
}

get_tokens <- function(text) {
  text <- clean_text(text)
  summary_tokenizer %>% fit_text_tokenizer(text)
  text_to_seq <- texts_to_sequences(summary_tokenizer, text)
  sequences <- text_to_seq %>% pad_sequences(maxlen=2**8, padding='pre', truncating = 'post')
  return(sequences)
}

# Function to predict based off a new string of text
predict_summary <- function(text) {
  tokens <- get_tokens(text)
  return(model %>% predict(tokens) %>% round())
}

# Explain some sentences (112 is an example of a science book)
sentence_to_explain <- data$summary[c(2,117)]
explainer <- lime(sentence_to_explain, model=model, preprocess=get_embedding_explination)
explination <- explain(sentence_to_explain, explainer, n_labels=1, n_features=25, n_permutations = 1e4)

# Get the actual explanations
plot_text_explanations(explination)
plot_features(explination)
interactive_text_explanations(explainer)


