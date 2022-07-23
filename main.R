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
# Install and bootstrap keras
use_condaenv("tensorflow-metal")

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
