import streamlit as st
import easyocr
from PIL import Image
import io
from io import StringIO
import random
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Team 46", page_icon="📖")

# load in css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

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
    choice = st.sidebar.selectbox(
        "Source", choices, help="You can upload an image or text file")

# init states
if "summary_button_clicked" not in st.session_state:
    st.session_state.summary_button_clicked = False
if "quiz_button_clicked" not in st.session_state:
    st.session_state.quiz_button_clicked = False
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "current_question_temp" not in st.session_state:
    st.session_state.current_question_temp = -1
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
            "Quiz me!", on_click=quiz_button_callback, key="quiz_button", use_container_width=True
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
            "", type=['png'], help="Upload a file to begin!")

        if not st.session_state.summary_button_clicked and not st.session_state.quiz_button_clicked:
            image = None
            # get text from image using easyocr
            if uploaded_file is not None and not st.session_state.new_source_summary and not st.session_state.new_source_quiz:
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
                    summarizer = pipeline(
                        "summarization", model='jazzisfuture/new_summary_t5_small')
                    summary_raw = summarizer(st.session_state.extracted_text,
                                             max_length=250, min_length=30, do_sample=False)
                    st.session_state.summary = summary_raw[0]["summary_text"]
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

                while st.session_state.current_question == st.session_state.current_question_temp:
                    st.session_state.current_question = (random.randint(0, 4))
                question = st.session_state.questions_answers[st.session_state.current_question][0]
                answer = st.session_state.questions_answers[st.session_state.current_question][1]
                st.session_state.current_question_temp = st.session_state.current_question

                st.markdown(
                    f'<div class="blockquote-wrapper"><div class="blockquote"><h1><span style="color:#1e1e1e;">{question}</span></h1><h4>&mdash; Click "Quiz me!" for another question!</em></h4></div></div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='answer'><span style='font-weight: bold; color:#6d7284;'>Answer:</span><br><br>{answer}</div>",
                    unsafe_allow_html=True,
                )

    elif choice == "Text":
        uploaded_file = st.sidebar.file_uploader(
            "", type=['txt'])

        if not st.session_state.summary_button_clicked and not st.session_state.quiz_button_clicked:
            if uploaded_file is not None:
                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()
                st.write(bytes_data)

                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                st.write(stringio)

                # To read file as string:
                text = stringio.read()

    st.session_state.summary_button_clicked = False
    st.session_state.quiz_button_clicked = False

with tab2:
    # Mock data
    quiz_scores = [75, 90, 85, 78, 92, 88, 96, 81, 95, 89, 82, 91, 84, 77, 99]

    quiz_names = [
        "Introduction to AI",
        "Machine Learning Basics",
        "Deep Learning",
        "Neural Networks",
        "Computer Vision",
        "Natural Language Processing",
        "Reinforcement Learning",
        "Generative Models",
        "AI Ethics",
        "Robotics",
        "AI in Healthcare",
        "AI in Finance",
        "AI in Agriculture",
        "AI in Manufacturing",
        "AI in Transportation",
    ]

    quiz_data = pd.DataFrame({"Quiz": quiz_names, "Score": quiz_scores,
                             "Quiz Number": list(range(1, len(quiz_scores) + 1))})

    # Summary chart
    average_score = quiz_data["Score"].mean()
    min_score = quiz_data["Score"].min()
    max_score = quiz_data["Score"].max()

    st.header("Summary Statistics")
    summary_data = pd.DataFrame({
        "Statistic": ["Average Score", "Lowest Score", "Highest Score"],
        "Score": [average_score, min_score, max_score],
    })
    fig = px.bar(summary_data, x="Statistic", y="Score",
                 title="", color_discrete_sequence=["#9EE6CF"])
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Score")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # All quizes chart
    st.header("Quiz Scores by Topic")
    fig = px.bar(quiz_data, x="Quiz", y="Score", title="",
                 color_discrete_sequence=["#9EE6CF"])
    fig.update_xaxes(title_text="Quiz")
    fig.update_yaxes(title_text="Score")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Fit a linear regression model
    X = quiz_data["Quiz Number"].values.reshape(-1, 1)
    y = quiz_data["Score"].values.reshape(-1, 1)
    regression_model = LinearRegression()
    regression_model.fit(X, y)

    y_pred = regression_model.predict(X)

    # Quiz scores with linear regression chart
    st.header("Linear Regression of Quiz Scores")
    fig = px.scatter(quiz_data, x="Quiz Number", y="Score",
                     text="Quiz", color_discrete_sequence=["#9EE6CF"])
    fig.add_trace(px.line(
        x=quiz_data["Quiz Number"], y=y_pred.reshape(-1), markers=False).data[0])
    fig.update_xaxes(title_text="Quiz Number")
    fig.update_yaxes(title_text="Score")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Predicts next score, creates dataframe for previous score and predicted
    next_quiz_number = len(quiz_scores) + 1
    next_quiz_score = regression_model.predict(
        np.array([[next_quiz_number]]))[0][0]

    prev_and_projected_data = pd.DataFrame({
        "Type": ["Previous Quiz Score", "Projected Score for Next Quiz"],
        "Score": [quiz_scores[-1], next_quiz_score],
    })

    # Chart for previous and predicted
    st.header("Previous Quiz Score vs Projected Score for Next Quiz")
    fig = px.bar(prev_and_projected_data, x="Type", y="Score",
                 title="", color_discrete_sequence=["#9EE6CF"])
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Score")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
