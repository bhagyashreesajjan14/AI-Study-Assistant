import streamlit as st
import pandas as pd

from config import (
    NOTES_DIR,
    VECTOR_DIR,
    SUBJECTS
)

from database import (
    init_database,
    save_profile,
    get_profile,
    save_quiz_result,
    get_topic_performance,
    get_quiz_attempts
)

from ai import (
    ask_ai,
    generate_quiz,
    explain_mistake
)

from rag import (
    create_chunks,
    build_index,
    search,
    save_index
)

from quiz import (
    calculate_score,
    choose_difficulty
)

from ml_model import (
    load_performance_data,
    analyze_performance,
    train_model
)

from recommendations import (
    generate_recommendations,
    get_weak_topics
)

from study_planner import (
    generate_study_plan
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Study Assistant 2.0",
    page_icon="🎓",
    layout="wide"
)

init_database()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "chunks": [],
    "index": None,
    "quiz": None,
    "quiz_answers": {},
    "quiz_subject": "",
    "quiz_topic": "",
    "quiz_difficulty": "Easy"
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(
    "🎓 AI Study Assistant 2.0"
)

st.caption(
    "AI Tutor • RAG • Adaptive Quiz • ML • Recommendations"
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👤 Profile",
        "📚 Study Material",
        "🤖 AI Tutor",
        "📝 Quiz",
        "📊 Performance",
        "🎯 Recommendations"
    ]
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    profile = get_profile()

    if profile:

        name, course, semester = profile

        st.write(
            f"Welcome, **{name}**!"
        )

        st.caption(
            f"{course} • Semester {semester}"
        )

    else:

        st.info(
            "Create your student profile first."
        )

    performance = get_topic_performance()

    df = load_performance_data(
        performance
    )

    df = analyze_performance(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Topics",
            len(df)
        )

    with col2:

        if not df.empty:

            average = df[
                "average_score"
            ].mean()

            st.metric(
                "Average",
                f"{average:.1f}%"
            )

        else:

            st.metric(
                "Average",
                "—"
            )

    with col3:

        weak = get_weak_topics(df)

        st.metric(
            "Weak Topics",
            len(weak)
        )

    with col4:

        attempts = get_quiz_attempts()

        st.metric(
            "Quiz Attempts",
            len(attempts)
        )

    if not df.empty:

        st.divider()

        st.subheader(
            "📈 Topic Performance"
        )

        chart = df[
            ["topic", "average_score"]
        ].set_index("topic")

        st.bar_chart(chart)


# ==================================================
# PROFILE
# ==================================================

elif page == "👤 Profile":

    st.header("👤 Student Profile")

    profile = get_profile()

    current_name = (
        profile[0]
        if profile else ""
    )

    current_course = (
        profile[1]
        if profile else ""
    )

    current_semester = (
        profile[2]
        if profile else 1
    )

    name = st.text_input(
        "Name",
        current_name
    )

    course = st.text_input(
        "Course",
        current_course
    )

    semester = st.number_input(
        "Semester",
        min_value=1,
        max_value=12,
        value=int(current_semester)
    )

    if st.button(
        "Save Profile"
    ):

        if name.strip():

            save_profile(
                name,
                course,
                semester
            )

            st.success(
                "Profile saved!"
            )

        else:

            st.warning(
                "Please enter your name."
            )


# ==================================================
# STUDY MATERIAL
# ==================================================

elif page == "📚 Study Material":

    st.header(
        "📚 Study Material"
    )

    subject = st.selectbox(
        "Select subject",
        SUBJECTS
    )

    uploaded_file = st.file_uploader(
        "Upload PDF notes",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button(
            "Process Notes"
        ):

            with st.spinner(
                "Processing PDF..."
            ):

                try:

                    subject_dir = (
                        NOTES_DIR / subject
                    )

                    subject_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    pdf_path = (
                        subject_dir /
                        uploaded_file.name
                    )

                    with open(
                        pdf_path,
                        "wb"
                    ) as file:

                        file.write(
                            uploaded_file.getbuffer()
                        )

                    chunks = create_chunks(
                        pdf_path,
                        subject
                    )

                    index = build_index(
                        chunks
                    )

                    st.session_state.chunks = chunks
                    st.session_state.index = index

                    index_path = (
                        VECTOR_DIR /
                        f"{subject}.index"
                    )

                    metadata_path = (
                        VECTOR_DIR /
                        f"{subject}.json"
                    )

                    save_index(
                        index,
                        chunks,
                        index_path,
                        metadata_path
                    )

                    st.success(
                        f"Successfully processed "
                        f"{len(chunks)} chunks."
                    )

                except Exception as e:

                    st.error(
                        f"Processing error: {e}"
                    )

    if st.session_state.chunks:

        st.info(
            f"Loaded {len(st.session_state.chunks)} "
            "knowledge chunks."
        )


# ==================================================
# AI TUTOR
# ==================================================

elif page == "🤖 AI Tutor":

    st.header(
        "🤖 AI Tutor"
    )

    subject = st.selectbox(
        "Subject",
        ["General"] + SUBJECTS
    )

    mode = st.selectbox(
        "Tutor Mode",
        [
            "Explain",
            "Simple Explanation",
            "Example",
            "Exam Preparation"
        ]
    )

    question = st.text_area(
        "Ask your question",
        placeholder="Explain normalization..."
    )

    notes_only = st.checkbox(
        "Answer using my uploaded notes only"
    )

    if st.button(
        "Ask AI"
    ):

        if not question.strip():

            st.warning(
                "Enter a question first."
            )

        else:

            with st.spinner(
                "Thinking..."
            ):

                try:

                    context = ""

                    sources = []

                    if (
                        notes_only
                        and st.session_state.index
                    ):

                        results = search(
                            question,
                            st.session_state.chunks,
                            st.session_state.index,
                            subject=None
                            if subject == "General"
                            else subject
                        )

                        if results:

                            context = "\n\n".join(
                                item["text"]
                                for item in results
                            )

                            sources = results

                        else:

                            st.warning(
                                "I couldn't find relevant "
                                "information in your notes."
                            )

                    modified_question = question

                    if mode == "Simple Explanation":

                        modified_question = (
                            "Explain this for a beginner: "
                            + question
                        )

                    elif mode == "Example":

                        modified_question = (
                            "Explain this concept and "
                            "give a practical example: "
                            + question
                        )

                    elif mode == "Exam Preparation":

                        modified_question = (
                            "Explain this in an "
                            "exam-oriented format with "
                            "important points: "
                            + question
                        )

                    answer = ask_ai(
                        modified_question,
                        context
                    )

                    st.subheader(
                        "Answer"
                    )

                    st.write(answer)

                    if sources:

                        with st.expander(
                            "📚 Sources"
                        ):

                            for source in sources:

                                st.markdown(
                                    f"**{source['source']} "
                                    f"— Page {source['page']}**"
                                )

                                st.write(
                                    source["text"]
                                )

                except Exception as e:

                    st.error(
                        f"AI error: {e}"
                    )


# ==================================================
# QUIZ
# ==================================================

elif page == "📝 Quiz":

    st.header(
        "📝 Adaptive Quiz"
    )

    subject = st.selectbox(
        "Subject",
        SUBJECTS
    )

    topic = st.text_input(
        "Topic",
        "Transactions"
    )

    previous_score = st.number_input(
        "Previous score (optional)",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

    if previous_score == 0:

        difficulty = "Easy"

    else:

        difficulty = choose_difficulty(
            previous_score
        )

    st.info(
        f"Recommended difficulty: **{difficulty}**"
    )

    number = st.slider(
        "Questions",
        3,
        10,
        5
    )

    if st.button(
        "Generate Quiz"
    ):

        with st.spinner(
            "Generating quiz..."
        ):

            try:

                quiz_data = generate_quiz(
                    subject,
                    topic,
                    difficulty,
                    number
                )

                st.session_state.quiz = (
                    quiz_data["questions"]
                )

                st.session_state.quiz_subject = subject
                st.session_state.quiz_topic = topic
                st.session_state.quiz_difficulty = difficulty
                st.session_state.quiz_answers = {}

                st.success(
                    "Quiz generated!"
                )

            except Exception as e:

                st.error(
                    f"Quiz error: {e}"
                )

    if st.session_state.quiz:

        quiz = st.session_state.quiz

        for i, question in enumerate(
            quiz
        ):

            st.subheader(
                f"Question {i + 1}"
            )

            st.write(
                question["question"]
            )

            answer = st.radio(
                "Select answer",
                question["options"],
                key=f"q_{i}"
            )

            st.session_state.quiz_answers[i] = (
                question["options"].index(
                    answer
                )
            )

        if st.button(
            "Submit Quiz"
        ):

            correct, total, score, details = (
                calculate_score(
                    quiz,
                    st.session_state.quiz_answers
                )
            )

            answers_for_db = []

            for detail in details:

                answers_for_db.append({
                    "question":
                        detail["question"],
                    "selected_answer":
                        detail["selected_answer"],
                    "correct_answer":
                        detail["correct_answer"],
                    "is_correct":
                        detail["is_correct"]
                })

            save_quiz_result(
                st.session_state.quiz_subject,
                st.session_state.quiz_topic,
                st.session_state.quiz_difficulty,
                score,
                total,
                correct,
                answers_for_db
            )

            st.success(
                f"Score: {correct}/{total} "
                f"({score:.1f}%)"
            )

            st.divider()

            for detail in details:

                if detail["is_correct"]:

                    st.success(
                        f"✅ {detail['question']}"
                    )

                else:

                    st.error(
                        f"❌ {detail['question']}"
                    )

                    st.write(
                        f"Your answer: "
                        f"{detail['selected_answer']}"
                    )

                    st.write(
                        f"Correct answer: "
                        f"{detail['correct_answer']}"
                    )

                    with st.expander(
                        "🤖 Explain my mistake"
                    ):

                        explanation = (
                            explain_mistake(
                                detail["question"],
                                detail["selected_answer"],
                                detail["correct_answer"],
                                st.session_state.quiz_topic
                            )
                        )

                        st.write(
                            explanation
                        )


# ==================================================
# PERFORMANCE
# ==================================================

elif page == "📊 Performance":

    st.header(
        "📊 Performance Analytics"
    )

    rows = get_topic_performance()

    df = load_performance_data(
        rows
    )

    df = analyze_performance(
        df
    )

    if df.empty:

        st.info(
            "Take some quizzes first."
        )

    else:

        st.dataframe(
            df,
            use_container_width=True
        )

        st.subheader(
            "Performance by Topic"
        )

        chart = df[
            ["topic", "average_score"]
        ].set_index("topic")

        st.bar_chart(chart)

        st.subheader(
            "🧠 ML Analysis"
        )

        model, metrics = train_model(
            df
        )

        if metrics:

            st.metric(
                "Model Accuracy",
                f"{metrics['accuracy'] * 100:.1f}%"
            )

            with st.expander(
                "Classification Report"
            ):

                st.text(
                    metrics["report"]
                )

        else:

            st.info(
                "More performance data is needed "
                "before ML training becomes meaningful."
            )


# ==================================================
# RECOMMENDATIONS
# ==================================================

elif page == "🎯 Recommendations":

    st.header(
        "🎯 Personalized Recommendations"
    )

    rows = get_topic_performance()

    df = load_performance_data(
        rows
    )

    df = analyze_performance(
        df
    )

    if df.empty:

        st.info(
            "Complete some quizzes first."
        )

    else:

        recommendations = (
            generate_recommendations(
                df
            )
        )

        for item in recommendations:

            if item["priority"] == "HIGH":
                icon = "🔴"

            elif item["priority"] == "MEDIUM":
                icon = "🟠"

            else:
                icon = "🟢"

            st.markdown(
                f"{icon} **{item['topic']}** — "
                f"{item['score']:.1f}% — "
                f"{item['priority']}"
            )

            st.caption(
                item["recommendation"]
            )

        weak_topics = get_weak_topics(
            df
        )

        if weak_topics:

            st.divider()

            st.subheader(
                "📅 Personalized Study Plan"
            )

            days = st.slider(
                "Number of days",
                3,
                14,
                7
            )

            hours = st.slider(
                "Hours per day",
                1,
                6,
                2
            )

            if st.button(
                "Generate Study Plan"
            ):

                with st.spinner(
                    "Creating your plan..."
                ):

                    plan = generate_study_plan(
                        weak_topics,
                        days,
                        hours
                    )

                    st.write(plan)