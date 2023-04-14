import streamlit as st
import easyocr
from PIL import Image
import io
from transformers import pipeline
from pprint import pprint
from tqdm.auto import tqdm
from haystack.nodes import QuestionGenerator, BM25Retriever, FARMReader
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines import (
    QuestionGenerationPipeline,
    RetrieverQuestionGenerationPipeline,
    QuestionAnswerGenerationPipeline,
)
from haystack.utils import launch_es, print_questions

st.set_page_config(page_title="Team 46", page_icon="📖")

with st.sidebar:
    st.title("Title")
    choices = ["Image", "Text"]
    choice = st.sidebar.selectbox("Source", choices)

# init states
if "summary_button_clicked" not in st.session_state:
    st.session_state.summary_button_clicked = False
if "quiz_button_clicked" not in st.session_state:
    st.session_state.quiz_button_clicked = False
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "current_question_temp" not in st.session_state:
    st.session_state.current_question_temp = 0


def summary_button_callback():
    st.session_state.summary_button_clicked = True


def quiz_button_callback():
    st.session_state.quiz_button_clicked = True


def display_buttons():
    # render summarize / quiz buttons
    col1, col2 = st.columns(2)
    with col1:
        summary_button = st.button(
            "Summarize", on_click=summary_button_callback, key="summary_button", use_container_width=True
        )
    with col2:
        quiz_button = st.button(
            "Pop Quiz", on_click=quiz_button_callback, key="quiz_button", use_container_width=True
        )


def update_sidebar():
    st.sidebar.write("Original image")
    st.sidebar.image(st.session_state.image)
    with st.expander("See original text"):
        st.write(st.session_state.extracted_text)


tab1, tab2 = st.tabs(["Main", "Analysis"])

with tab1:
    if choice == "Image":
        uploaded_file = st.sidebar.file_uploader(
            "", type=['jpg', 'png', 'jpeg'])

        if not st.session_state.summary_button_clicked and not st.session_state.quiz_button_clicked:
            image = None
            # get text from image using easyocr
            if uploaded_file is not None:
                with st.spinner("Extracting text..."):
                    image = Image.open(uploaded_file)
                    st.session_state.image = image
                    shown_image_element = st.image(image)
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG',
                               subsampling=0, quality=100)
                    img_byte_arr = img_byte_arr.getvalue()
                    reader = easyocr.Reader(['en'])
                    result = reader.readtext(img_byte_arr)

                    # prepare extracted text
                    text = [x[1] for x in result]
                    extracted_text_header = st.write("Extracted Text:")
                    st.session_state.text_length = len(text)
                    st.session_state.extracted_text = ' '.join(text)
                    st.write(st.session_state.extracted_text)
                    display_buttons()
                st.sidebar.write("Original image")
                st.sidebar.image(st.session_state.image)
                shown_image_element = st.empty()

        # summary option
        if st.session_state.summary_button_clicked:
            update_sidebar()
            summarizer = pipeline(
                "summarization", model="facebook/bart-large-cnn")
            with st.spinner("Summarizing text..."):
                display_buttons()
                summary = summarizer(st.session_state.extracted_text,
                                     max_length=150, min_length=30, do_sample=False)
                st.write(summary[0]["summary_text"])

        # quiz option
        if st.session_state.quiz_button_clicked:
            update_sidebar()
            display_buttons()
            st.write("QUIZ")

    elif choice == "Text":
        uploaded_file = st.sidebar.file_uploader(
            "", type=['txt', 'doc', 'pdf'])

    st.session_state.summary_button_clicked = False
    st.session_state.quiz_button_clicked = False

with tab2:
    pass
