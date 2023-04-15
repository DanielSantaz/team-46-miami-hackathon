import streamlit as st
import easyocr
from PIL import Image
import io
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import requests

st.set_page_config(page_title="Team 46", page_icon="📖")

# question generator setup
question_model = "allenai/t5-small-squad2-question-generation"
tokenizer = T5Tokenizer.from_pretrained(question_model)
model = T5ForConditionalGeneration.from_pretrained(question_model)


@st.cache_resource
def generate_question(text, **generator_args):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return output


# answer generator setup
answer_model = "consciousAI/question-answering-roberta-base-s"

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
if "new_source_summary" not in st.session_state:
    st.session_state.new_source_summary = False
if "new_source_quiz" not in st.session_state:
    st.session_state.new_source_quiz = False


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
                    st.sidebar.write("Original image")
                    st.sidebar.image(image)
                    st.session_state.image = image
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG',
                               subsampling=0, quality=100)
                    img_byte_arr = img_byte_arr.getvalue()
                    reader = easyocr.Reader(['en'])
                    result = reader.readtext(img_byte_arr)

                    # prepare extracted text
                    text = [x[1] for x in result]
                    st.session_state.sentence_list = text
                    extracted_text_header = st.write("Extracted Text:")
                    st.session_state.extracted_text = ' '.join(text)
                    st.write(st.session_state.extracted_text)
                    display_buttons()
                    st.session_state.new_source_summary = True
                    st.session_state.new_source_quiz = True

        # summary option
        if st.session_state.summary_button_clicked:
            update_sidebar()
            display_buttons()
            with st.spinner("Summarizing text..."):
                if st.session_state.new_source_summary:
                    querystring = {"sentences": "5",
                                   "txt": st.session_state.extracted_text}

                    headers = {
                        "Accept": "application/json",
                        "X-RapidAPI-Key": "5b64383f18msh37a79f12c4f83e3p1fc6e5jsn69b03780ea58",
                        "X-RapidAPI-Host": "meaningcloud-summarization-v1.p.rapidapi.com"
                    }

                    response = requests.request(
                        "GET", "https://meaningcloud-summarization-v1.p.rapidapi.com/summarization-1.0", headers=headers, params=querystring)
                    data = response.json()
                    st.session_state.summary = data['summary']
                    st.session_state.new_source_summary = False
                st.write(st.session_state.summary)

        # quiz option
        if st.session_state.quiz_button_clicked:
            update_sidebar()
            display_buttons()
            with st.spinner("Generating questions..."):
                if st.session_state.new_source_quiz:
                    paragraph = ""
                    paragraphs = []
                    for sentence in st.session_state.sentence_list:
                        paragraph += " " + sentence
                        if len(paragraph) > 200:
                            paragraphs.append(paragraph)
                            paragraph = ""
                    num_questions = min(len(paragraphs), 5)
                    questions_answers = []
                    question_answerer = pipeline(
                        "question-answering", model=answer_model)
                    for paragraph in paragraphs[:num_questions]:
                        question_list = generate_question(paragraph)
                        question = question_list[0]
                        answer = question_answerer(
                            question=question, context=paragraph)
                        questions_answers.append((question, answer['answer']))
                    st.session_state.questions_answers = questions_answers
                    st.session_state.new_source_quiz = False
                for qa in st.session_state.questions_answers:
                    "Question:"
                    qa[0]
                    "Answer:"
                    qa[1]

    elif choice == "Text":
        uploaded_file = st.sidebar.file_uploader(
            "", type=['txt', 'doc', 'pdf'])

    st.session_state.summary_button_clicked = False
    st.session_state.quiz_button_clicked = False

with tab2:
    pass
