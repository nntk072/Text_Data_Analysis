"""
Prune the vocabulary to remove stopwords, overly short and long words, the top 1% most frequent
words and words occurring less than 4 times. Report the top-100 words after pruning.
"""

import requests
import re
import os
import nltk
import bs4

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


# Function to get the top k books from the last 30 days
def get_top_k_books(k):
    # Get the top 100 books from the last 30 days
    url = "https://www.gutenberg.org/browse/scores/top"
    response = requests.get(url)
    mywebpage_parsed = bs4.BeautifulSoup(response.text, "html.parser")
    # Get the elements inside the tag
    h2_element = mywebpage_parsed.find(id="books-last30")
    ol_element = h2_element.find_next("ol")
    book_items = ol_element.find_all("a")

    #  Get the top k books
    top_k_links = []
    top_k_names = []
    for i in range(k):
        book = book_items[i]
        book_title = book.text.strip()
        book_link = book["href"]
        top_k_links.append(book_link)
        top_k_names.append(book_title)

    download_url = "https://www.gutenberg.org/cache/epub"

    # Split the link to get the book id
    book_ids = [link.split("/")[-1] for link in top_k_links]
    # Create the download link
    book_link = [f"{download_url}/{book_id}/pg{book_id}.txt" for book_id in book_ids]
    print(f"Downloading progress")
    # Uncomment the following lines to print the download progress
    # for i in range(k):
    #     print(f"Downloading {top_k_names[i]}: {book_link[i]}")

    return top_k_names, book_link


def get_book_content(book_link):
    response = requests.get(book_link)
    book_content = response.text
    return book_content


def tokenize_and_lemmatize(book_content):
    start_index = book_content.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end_index = book_content.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    book_content = book_content[start_index:end_index]
    book_content = book_content.strip()
    book_content = " ".join(book_content.split())
    # Tokenize the book content
    tokens = nltk.word_tokenize(book_content)
    # print(tokens)
    # Lemmatize the tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def remove_stopwords(vocabulary):
    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    vocabulary = [word for word in vocabulary if word not in stopwords]
    # print(f"Vocabulary size after removing stopwords: {len(vocabulary)}")
    return vocabulary


def remove_short_words(vocabulary):
    # Remove short words
    vocabulary = [word for word in vocabulary if len(word) > 2]
    # print(f"Vocabulary size after removing short words: {len(vocabulary)}")
    return vocabulary


def remove_long_words(vocabulary):
    # Remove long words
    vocabulary = [word for word in vocabulary if len(word) < 20]
    # print(f"Vocabulary size after removing long words: {len(vocabulary)}")
    return vocabulary


def remove_top_1_percent(vocabulary):
    # Remove the top 1% most frequent words
    word_frequencies = nltk.FreqDist(vocabulary)
    top_1_percent = int(len(word_frequencies) * 0.01)
    top_1_percent_words = word_frequencies.most_common(top_1_percent)
    top_1_percent_words = [word[0] for word in top_1_percent_words]
    vocabulary = [word for word in vocabulary if word not in top_1_percent_words]
    # print(
    #    f"Vocabulary size after removing top 1% most frequent words: {len(vocabulary)}"
    # )
    return vocabulary


def remove_less_than_4_times(vocabulary):
    # Remove words occurring less than 4 times
    word_frequencies = nltk.FreqDist(vocabulary)
    vocabulary = [word for word in vocabulary if word_frequencies[word] >= 4]
    # print(
    #    f"Vocabulary size after removing words occurring less than 4 times: {len(vocabulary)}"
    # )
    return vocabulary


def vocabulary_pruning(vocabulary):
    # Remove stopwords
    vocabulary = remove_stopwords(vocabulary)
    # Remove short words
    vocabulary = remove_short_words(vocabulary)
    # Remove long words
    vocabulary = remove_long_words(vocabulary)
    # Remove the top 1% most frequent words
    vocabulary = remove_top_1_percent(vocabulary)
    # Remove words occurring less than 4 times
    vocabulary = remove_less_than_4_times(vocabulary)
    return vocabulary


def main():
    # Get the top 20 books from the last 30 days
    k = 20
    top_k_names, book_link = get_top_k_books(k)
    book_content_list = []
    for i in range(k):
        # Get the book content
        book_content = get_book_content(book_link[i])
        # Add book content to a list
        book_content_list.append(book_content)
    # Combine the book content into a single string
    book_content = " ".join(book_content_list)
    # Tokenize and lemmatize the book content
    print("Tokenizing and lemmatizing the book content")
    lemmatized_tokens = tokenize_and_lemmatize(book_content)

    # Prune the vocabulary
    print("Pruning the vocabulary")
    vocabulary = vocabulary_pruning(lemmatized_tokens)

    # Get the top 100 words
    word_frequencies = nltk.FreqDist(vocabulary)
    # Print the top 100 words without their frequencies
    top_100_words_without_frequencies = [
        word[0] for word in word_frequencies.most_common(100)
    ]
    print(top_100_words_without_frequencies)


if __name__ == "__main__":
    main()
