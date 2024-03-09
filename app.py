import torch
import wikipedia
import streamlit as st
from transformers import pipeline, Pipeline

def load_summary(topic: str) -> str:
    search_res = wikipedia.search(topic)
    print(search_res)
    if not search_res:
        return 'No results found'
    summary = wikipedia.summary(search_res[0], sentences=10)
    return summary


def load_pipeline() -> Pipeline:
    # question-answering pipeline from huggingface
    wiki_pipeline = pipeline('question-answering', model = "distilbert-base-uncased-distilled-squad")
    return wiki_pipeline

def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> dict:
    input = {
        "question": question,
        "context": paragraph
    }

    output = pipeline(input)

    return output


# If the script is run directly, then the below code will be executed.
if __name__ == '__main__':
    # Create a title for the app
    st.title('Wikipedia - Question Answering')
    st.write('Search a topic from Wikipedia, ask a question and get an answer')

    topic = st.text_input('Enter a topic to search on Wikipedia')

    article = st.empty()

    question = st.text_input('Enter a Question')

    if topic:
        summary = load_summary(topic)

        article.markdown(summary)

        if question != "":
            wiki_pipeline = load_pipeline()
            
            result = answer_question(wiki_pipeline, question, summary)
            answer = result['answer']

            st.write(answer)

