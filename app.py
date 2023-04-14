import streamlit as st
import easyocr
from PIL import Image
import io
from transformers import pipeline, T5Config, T5ForConditionalGeneration, T5Tokenizer

# question generator setup
question_model = "allenai/t5-small-squad2-question-generation"
tokenizer = T5Tokenizer.from_pretrained(question_model)
model = T5ForConditionalGeneration.from_pretrained(question_model)

def generate_question(text, **generator_args):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return output

#answer generator setup
answer_model = "consciousAI/question-answering-roberta-base-s"

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
                    reader = easyocr.Reader(['en'],download_enabled=False)
                    result = reader.readtext(img_byte_arr)

                    # prepare extracted text
                    text = [x[1] for x in result]
                    st.session_state.sentence_list = text
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
            paragraph = ""
            paragraphs = []
            for sentence in st.session_state.sentence_list:
                paragraph += " " + sentence
                if len(paragraph) > 200:
                    paragraphs.append(paragraph)
                    paragraph = ""
            num_questions = min(len(paragraphs), 5)
            questions_answers = []
            question_answerer = pipeline("question-answering", model=answer_model)
            for paragraph in paragraphs[:num_questions]:
                question_list = generate_question(paragraph)
                question = question_list[0]
                answer = question_answerer(question=question, context=paragraph)
                questions_answers.append((question, answer['answer']))
            for qa in questions_answers:
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
