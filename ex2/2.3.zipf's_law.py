import requests
import re
import os
import nltk
import bs4
import matplotlib.pyplot as plt
import numpy as np

nltk.download("punkt")
nltk.download("wordnet")


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

    for i in range(k):
        print(f"Downloading {top_k_names[i]}: {book_link[i]}")

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


def get_vocabulary(lemmatized_tokens):
    # Get the vocabulary
    vocabulary = list(set(lemmatized_tokens))
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
    lemmatized_tokens = tokenize_and_lemmatize(book_content)
    # Get the vocabulary
    vocabulary = get_vocabulary(lemmatized_tokens)

    # Get the word frequencies
    word_frequencies = nltk.FreqDist(lemmatized_tokens)

    # Get the total number of words
    total_words = len(lemmatized_tokens)

    # Calculate the frequency of each word divided by the total count of all words
    word_freq = {word: freq / total_words for word, freq in word_frequencies.items()}

    # Sort the frequencies in descending order
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Plot the frequencies
    plt.figure(figsize=(10, 5))
    plt.plot([freq for word, freq in sorted_word_freq], label="Real frequencies")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Word Frequencies in Top 20 Books from Project Gutenberg")

    # Plot the frequency distribution according to Zipf's law
    for a in [0.5, 1.0, 1.5, 2.0]:
        zipf = [
            sorted_word_freq[0][1] / (i**a)
            for i in range(1, len(sorted_word_freq) + 1)
        ]
        plt.plot(zipf, label=f"Zipf (a={a})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
