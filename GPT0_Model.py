
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_sequence, padded_everygram_pipeline
from nltk.lm import MLE, Vocabulary
from nltk.probability import FreqDist
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
import string
import PyPDF2
import pandas as pd
from io import StringIO
import requests
from bs4 import BeautifulSoup
import urllib.parse


#Load gpt2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

def split_text_into_chunks(text, chunk_size=512):
    tokens = tokenizer.encode(text)
    # Dividir el texto en fragmentos de `chunk_size` tokens
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]

def calculate_perplexity_for_chunks(text, chunk_size=512):
    total_perplexity = 0
    chunks = list(split_text_into_chunks(text, chunk_size))

    for chunk in chunks:
        input_ids = torch.tensor([chunk])

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Calcular la perplejidad del fragmento
        chunk_perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
        total_perplexity += chunk_perplexity.item()

    # Devolver la perplejidad promedio
    return total_perplexity / len(chunks) if chunks else 0

#def calculate_perplexity(text):
    encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    input_ids = encoded_input[0]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
    return perplexity.item()

def calculate_burstiness(text):
    tokens = preprocess_text(text)
    word_freq = nltk.FreqDist(tokens)

    avg_freq = sum(word_freq.values()) / len(word_freq)
    variance = sum((freq - avg_freq) ** 2 for freq in word_freq.values()) / len(word_freq)

    burstiness_score = variance / (avg_freq ** 2)
    return burstiness_score

def calculate_average_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences)
    total_words = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)
    
    if total_sentences == 0:
        return 0
    
    return total_words / total_sentences  # Average Sentence Length

def calculate_common_words_percentage(text):
    # List of common English words
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on", "with", 
        "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", 
        "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", 
        "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", 
        "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", 
        "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", 
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", 
        "new", "want", "because", "any", "these", "give", "day", "most", "us"
    ]
    tokens = nltk.word_tokenize(text.lower())
    common_word_count = sum(1 for word in tokens if word in common_words)
    total_words = len(tokens)
    if total_words == 0:
        return 0
    percentage_common_words = (common_word_count / total_words) * 100
    return percentage_common_words

def plot_top_repeated_words(text):
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

    # Count
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)

    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]

    # Plot
    fig = px.bar(x=words, y=counts, labels={'x':'Words', 'y':'Counts'}, title="Top 10 Most Repeated Words")
    st.plotly_chart(fig, use_container_width=True)


def define_possible_prompt(text):
    sentences = nltk.sent_tokenize(text)
    word_counts = Counter(preprocess_text(text))
    
    # Definir el tema principal por las palabras más comunes
    most_common_words = word_counts.most_common(3)
    common_words_str = ', '.join([word for word, count in most_common_words])
    
    # Definir el estilo del texto
    if len(sentences) > 5:
        style = "Write an essay"
    else:
        style = "Write a paragraph"
    
    # Longitud del texto
    if len(text.split()) > 100:
        detail = "detailed"
    else:
        detail = "brief"
    
    # Construir el posible prompt
    prompt = f"{style} {detail} on the subject of {common_words_str}. Make sure you hit the most important points and organize the content logically."
    
    return prompt

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_csv(data):
    output = StringIO()
    data.to_csv(output, index=False)
    processed_data = output.getvalue()
    return processed_data


st.set_page_config(layout="wide")
st.title("GPT Multi-Shield")

uploaded_files = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files is not None:
    results = []

    if st.button("Analyze All"):
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            

            if text:
                st.subheader(f"Analysis for {uploaded_file.name}")
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.info("Your Input Text")
                    st.success(text)

                with col2:
                    st.info("Calculated Score")
                    perplexity = calculate_perplexity_for_chunks(text)
                    burstiness_score = calculate_burstiness(text)
                    average_sentence_length = calculate_average_sentence_length(text)
                    common_words_percentage = calculate_common_words_percentage(text)
                    p_prompt = define_possible_prompt(text)
                    

                    st.success("Perplexity Score: " + str(perplexity / 1000))
                    st.success("Burstiness Score: " + str(burstiness_score))
                    st.success("Common Words    : " + str(common_words_percentage) + "%")
                    st.success("Average Sentence Length Score: " + str(average_sentence_length))


                    if (perplexity > 13000 and burstiness_score > 0.2) and common_words_percentage > 25:
                        st.error("Text Analysis Result: AI Generated Content")
                        
                    else:
                        st.success("Text Analysis Result: Likely not generated by AI")
                        

                    # Save the result in the results list
                    results.append({
                        "File Name": uploaded_file.name,
                        "Perplexity": perplexity / 1000,
                        "Burstiness": burstiness_score,
                        "Common Words (%)": common_words_percentage,
                        "Average Sentence Length": average_sentence_length,
                        "AI Generated": "Yes" if (perplexity > 13000 and burstiness_score > 0.2) and common_words_percentage > 25 else "No",
                        "Possible Prompt": p_prompt if (perplexity > 13000 and burstiness_score > 0.2) and common_words_percentage > 25 else "None"
                    })

                with col3:
                    if (perplexity > 13000 and burstiness_score > 0.2) and common_words_percentage > 25:
                        st.info("Possible Prompt")
                        p_prompt = define_possible_prompt(text)  # Placeholder for your function
                        st.success(p_prompt)
                    st.info("Basic Insights")
                    plot_top_repeated_words(text)

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Provide download button for CSV file
        st.markdown("### Download the analysis results")
        csv_data = generate_csv(results_df)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="analysis_results.csv",
            mime="text/csv"
        )