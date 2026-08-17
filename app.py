import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import (
    NOTES_DIR,
    VECTOR_DIR,
    get_user_notes_dir,
    get_user_vector_dir
)

from database import (
    init_database,
    register_user,
    authenticate_user,
    get_all_subjects,
    add_subject,
    save_profile,
    get_profile,
    save_quiz_result,
    get_topic_performance,
    get_quiz_attempts,
    create_chat_session,
    get_chat_sessions,
    get_chat_messages,
    save_chat_message,
    delete_chat_session,
    rename_chat_session,
    get_user_document_jobs,
    get_user_documents,
    get_user_completed_documents,
    check_duplicate_document,
    save_study_plan,
    get_latest_study_plan,
    get_user_study_plans
)

from ai import (
    ask_ai,
    ask_ai_chat,
    generate_quiz,
    explain_mistake
)

from rag import (
    create_chunks,
    build_index,
    search,
    search_user_notes,
    save_index,
    load_index,
    start_background_indexing,
    load_user_subject_index
)

from quiz import (
    calculate_score,
    choose_difficulty,
    validate_all_answered
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
    generate_study_plan,
    generate_study_plan_pdf,
    format_duration
)

from utils import (
    clean_text,
    format_subject
)


# --------------------------------------------------
# CONFIG & INIT
# --------------------------------------------------

st.set_page_config(
    page_title="AI Study Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_database()


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

defaults = {
    "authenticated": False,
    "user": None,
    "quiz": None,
    "quiz_answers": {},
    "quiz_submitted": False,
    "quiz_score_details": None,
    "quiz_subject": "",
    "quiz_topic": "",
    "quiz_difficulty": "Intermediate",
    "active_chat_id": None,
    "pending_prompt": None,
    "current_study_plan": None,
    "current_plan_pdf": None,
    "study_material_qa_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# MODERN DESIGN SYSTEM & CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1e293b;
}

/* Minimalist Brand Bar */
.app-brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.5rem;
    margin-bottom: 1.5rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.app-brand-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0;
}

.app-brand-badge {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    background: #f1f5f9;
    color: #334155;
    border-radius: 9999px;
    border: 1px solid #cbd5e1;
}

/* Compact Centered Auth Container */
.auth-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 1.5rem 0;
}

.auth-box {
    width: 100%;
    max-width: 380px;
    margin: 0 auto;
    padding: 1.8rem 1.8rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
}

.auth-header {
    text-align: center;
    margin-bottom: 1.2rem;
}

.auth-header h2 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 1.45rem;
    color: #0f172a;
    margin: 0 0 0.3rem 0;
}

.auth-header p {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
}

/* Modern Card */
.ui-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}

.ui-card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f172a;
    margin: 0 0 0.4rem 0;
}

.ui-card-subtitle {
    font-size: 0.875rem;
    color: #64748b;
    margin: 0;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 14px 18px !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

div[data-testid="stMetric"] label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

/* ChatGPT Welcome Container */
.welcome-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    max-width: 720px;
    margin: 0 auto;
}

.welcome-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
}

.welcome-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 2rem;
}

/* Status Badges */
.badge-queued {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-processing {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-completed {
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-failed {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}

.status-badge-correct {
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85rem;
}

.status-badge-wrong {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85rem;
}

/* Citations */
.citation-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #6366f1;
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 0.85rem;
}

.citation-header {
    font-weight: 600;
    color: #334155;
    margin-bottom: 4px;
}

/* Disclaimer */
.chat-disclaimer {
    text-align: center;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.75rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    html, body, [class*="css"], .stApp {
        color: #f1f5f9;
    }
    .app-brand-bar, .auth-box, .ui-card, div[data-testid="stMetric"] {
        background: #0f172a;
        border-color: #1e293b;
    }
    .app-brand-title, .auth-header h2, .ui-card-title, .welcome-title {
        color: #f8fafc;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    .citation-card {
        background: #1e293b;
        border-color: #334155;
    }
    .citation-header {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1e293b;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# REQUIREMENT 1: COMPACT, CENTERED LOGIN & SIGNUP SCREEN
# ==================================================

if not st.session_state.authenticated:
    col_left, col_center, col_right = st.columns([1.1, 1.8, 1.1])

    with col_center:
        st.markdown("""
        <div class="auth-box">
            <div class="auth-header">
                <h2>AI Study Assistant</h2>
                <p>Welcome Back &bull; Sign in to continue</p>
            </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            login_user = st.text_input("Username / Email", key="login_username", placeholder="e.g. alex")
            login_pass = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("Sign In", type="primary", use_container_width=True):
                success, message, user_data = authenticate_user(login_user, login_pass)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = user_data
                    st.session_state.active_chat_id = None
                    st.session_state.quiz = None
                    st.rerun()
                else:
                    st.error(message)

        with tab_signup:
            reg_name = st.text_input("Full Name", key="reg_name", placeholder="e.g. Alex Johnson")
            reg_user = st.text_input("Username", key="reg_username", placeholder="e.g. alex")
            reg_pass = st.text_input("Password", type="password", key="reg_password", placeholder="Create a password")
            reg_course = st.text_input("Course / Major", key="reg_course", placeholder="e.g. Computer Science")
            reg_sem = st.number_input("Semester", min_value=1, max_value=12, value=1, key="reg_sem")

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("Create Account", type="primary", use_container_width=True):
                success, message, user_data = register_user(reg_user, reg_pass, reg_name, reg_course, reg_sem)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = user_data
                    st.session_state.active_chat_id = None
                    st.session_state.quiz = None
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ==================================================
# AUTHENTICATED APP SHELL & ISOLATION
# ==================================================

current_user = st.session_state.user or {}
user_id = current_user.get("id", 1)
user_display = current_user.get("name", "Student")

# Top Brand Bar
st.markdown(f"""
<div class="app-brand-bar">
    <div>
        <h1 class="app-brand-title">AI Study Assistant</h1>
    </div>
    <div>
        <span class="app-brand-badge">{user_display} &bull; {current_user.get('course', 'Student')}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Dynamic Subject Fetcher
available_subjects = get_all_subjects()


# Sidebar Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "AI Tutor",
        "Dashboard",
        "Study Material",
        "Quiz & Practice",
        "Performance & Analytics",
        "Recommendations & Study Plan",
        "Profile"
    ]
)

# User session info & Sign Out button in sidebar
st.sidebar.markdown("---")
st.sidebar.caption(f"Logged in as **{user_display}** (`@{current_user.get('username', '')}`)")
if st.sidebar.button("Sign Out", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state.authenticated = False
    st.rerun()


# ==================================================
# REQUIREMENT 3: AI TUTOR (NOTES ONLY CHECKBOX & DOCUMENT SELECTOR)
# ==================================================

if page == "AI Tutor":
    st.sidebar.markdown("---")
    st.sidebar.caption("CHAT SESSIONS")

    chat_subject = st.sidebar.selectbox(
        "Subject Focus",
        ["General"] + available_subjects,
        key="tutor_sidebar_subject"
    )

    chats = get_chat_sessions(chat_subject, user_id=user_id)
    if st.session_state.active_chat_id is None:
        if chats:
            st.session_state.active_chat_id = chats[0]["id"]
        else:
            new_id = create_chat_session(
                chat_subject,
                title=f"New {format_subject(chat_subject) if chat_subject != 'General' else 'General'} Chat",
                user_id=user_id
            )
            st.session_state.active_chat_id = new_id
            chats = get_chat_sessions(chat_subject, user_id=user_id)

    if st.sidebar.button("+ New Chat", use_container_width=True, type="primary"):
        new_id = create_chat_session(
            chat_subject,
            title=f"New {format_subject(chat_subject) if chat_subject != 'General' else 'General'} Chat",
            user_id=user_id
        )
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.caption("RECENT CHATS")

    for chat in chats:
        col_chat, col_del = st.sidebar.columns([0.84, 0.16])
        is_active = (st.session_state.active_chat_id == chat["id"])

        btn_label = chat["title"]
        if is_active:
            btn_label = f"• {chat['title']}"

        with col_chat:
            if st.button(
                btn_label,
                key=f"chat_btn_{chat['id']}",
                use_container_width=True
            ):
                st.session_state.active_chat_id = chat["id"]
                st.rerun()

        with col_del:
            if st.button(
                "x",
                key=f"chat_del_{chat['id']}",
                help="Delete Chat"
            ):
                delete_chat_session(chat["id"], user_id=user_id)
                if st.session_state.active_chat_id == chat["id"]:
                    st.session_state.active_chat_id = None
                st.rerun()

    active_chat_id = st.session_state.active_chat_id

    if active_chat_id is not None:
        messages = get_chat_messages(active_chat_id, user_id=user_id)

        # Controls Row
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([0.30, 0.35, 0.35])
        with ctrl_col1:
            mode = st.selectbox(
                "Tutor Mode",
                [
                    "Explain",
                    "Simple Explanation",
                    "Example",
                    "Exam Preparation"
                ],
                key="tutor_mode"
            )
        with ctrl_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            notes_only = st.checkbox(
                "Reference uploaded notes only",
                value=False,
                key="tutor_notes_only"
            )
        with ctrl_col3:
            if notes_only:
                completed_docs = get_user_completed_documents(user_id)
                doc_options = ["All my completed notes"] + [d["filename"] for d in completed_docs]
                selected_ref_doc = st.selectbox("Reference Document", doc_options, key="tutor_ref_doc")
            else:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                active_subject_name = format_subject(chat_subject) if chat_subject != 'General' else 'General Knowledge'
                st.caption(f"Context: **{active_subject_name}**")

        st.markdown("---")

        # Empty State Welcome & Prompt Suggestions
        if len(messages) == 0:
            st.markdown("""
            <div class="welcome-container">
                <h2 class="welcome-title">How can I help you study today?</h2>
                <p class="welcome-subtitle">Ask questions, request concept breakdowns, or explore your uploaded course notes.</p>
            </div>
            """, unsafe_allow_html=True)

            sugg_col1, sugg_col2 = st.columns(2)
            with sugg_col1:
                if st.button("Explain Normalization with 1NF, 2NF, 3NF examples", use_container_width=True):
                    st.session_state.pending_prompt = "Explain Database Normalization with 1NF, 2NF, and 3NF examples clearly."
                    st.rerun()
                if st.button("Compare Processes vs Threads in Operating Systems", use_container_width=True):
                    st.session_state.pending_prompt = "What is the difference between Processes and Threads in Operating Systems?"
                    st.rerun()

            with sugg_col2:
                if st.button("Break down ACID properties with real-world scenarios", use_container_width=True):
                    st.session_state.pending_prompt = "Explain ACID properties in DBMS with real-world scenarios and examples."
                    st.rerun()
                if st.button("Summarize key concepts from my study material", use_container_width=True):
                    st.session_state.pending_prompt = "Provide a comprehensive summary of the core concepts in this subject."
                    st.rerun()

        # Render Chat History
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["sources"]:
                    with st.expander("Reference Sources"):
                        for source in msg["sources"]:
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-header">Source: 📄 <b>{source['source']}</b> &bull; Page {source['page']} (Relevance: {source['score']:.2f})</div>
                                <div>{source['text']}</div>
                            </div>
                            """, unsafe_allow_html=True)

        prompt = st.chat_input("Ask a question, request an explanation, or explore a topic...")
        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None

        if prompt:
            with st.chat_message("user"):
                st.write(prompt)

            save_chat_message(active_chat_id, "user", prompt, user_id=user_id)

            context = ""
            sources = []

            if notes_only:
                with st.spinner("Searching your uploaded notes..."):
                    doc_filter = selected_ref_doc if selected_ref_doc != "All my completed notes" else None
                    results = search_user_notes(
                        question=prompt,
                        user_id=user_id,
                        document_name=doc_filter,
                        subject=None if chat_subject == "General" else chat_subject,
                        top_k=5
                    )

                    if results:
                        context = "\n\n".join(item["text"] for item in results)
                        sources = [{
                            "source": item["source"],
                            "page": item["page"],
                            "text": item["text"],
                            "score": item["score"]
                        } for item in results]
                    else:
                        answer = "I couldn't find this information in your uploaded notes."
                        with st.chat_message("assistant"):
                            st.write(answer)
                        save_chat_message(active_chat_id, "assistant", answer, sources=[], user_id=user_id)
                        st.rerun()

            with st.spinner("Generating response..."):
                try:
                    full_history = []
                    for msg in messages:
                        full_history.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    full_history.append({
                        "role": "user",
                        "content": prompt
                    })

                    answer = ask_ai_chat(full_history, context=context, mode=mode, notes_only=notes_only)

                    with st.chat_message("assistant"):
                        st.write(answer)
                        if sources:
                            with st.expander("Reference Sources"):
                                for source in sources:
                                    st.markdown(f"""
                                    <div class="citation-card">
                                        <div class="citation-header">Source: 📄 <b>{source['source']}</b> &bull; Page {source['page']} (Relevance: {source['score']:.2f})</div>
                                        <div>{source['text']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                    save_chat_message(active_chat_id, "assistant", answer, sources=sources, user_id=user_id)

                    if len(messages) == 0:
                        new_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
                        rename_chat_session(active_chat_id, new_title, user_id=user_id)

                    st.rerun()
                except Exception as e:
                    st.error(f"Response error: {e}")

        st.markdown(
            '<div class="chat-disclaimer">AI Study Assistant provides educational guidance. Verify important formulas and exam facts with your course syllabus.</div>',
            unsafe_allow_html=True
        )


# ==================================================
# DASHBOARD
# ==================================================

elif page == "Dashboard":
    profile = get_profile(user_id=user_id)

    if profile:
        name, course, semester = profile
        st.markdown(f"""
        <div class="ui-card">
            <h3 class="ui-card-title">Welcome back, {name}</h3>
            <p class="ui-card-subtitle">{course} &bull; Semester {semester}</p>
        </div>
        """, unsafe_allow_html=True)

    performance = get_topic_performance(user_id=user_id)
    df = load_performance_data(performance)
    df = analyze_performance(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Topics Studied", len(df))

    with col2:
        if not df.empty:
            average = df["average_score"].mean()
            st.metric("Average Score", f"{average:.1f}%")
        else:
            st.metric("Average Score", "—")

    with col3:
        weak = get_weak_topics(df)
        st.metric("Focus Areas", len(weak))

    with col4:
        attempts = get_quiz_attempts(user_id=user_id)
        st.metric("Quizzes Completed", len(attempts))

    st.markdown("<br>", unsafe_allow_html=True)

    # Active Background Jobs summary
    user_jobs = get_user_document_jobs(user_id=user_id)
    active_jobs = [j for j in user_jobs if j["status"] in ("queued", "processing")]
    if active_jobs:
        st.markdown("""
        <div class="ui-card" style="border-left: 4px solid #f59e0b;">
            <h4 class="ui-card-title">Active Background Indexing Tasks</h4>
            <p class="ui-card-subtitle">Your PDF notes are processing independently in the background.</p>
        </div>
        """, unsafe_allow_html=True)
        for job in active_jobs:
            st.write(f"**{job['filename']}** ({format_subject(job['subject'])}) — {job['status'].upper()} ({job['progress']}%)")
            st.progress(job["progress"] / 100.0)

    if not df.empty:
        st.markdown("""
        <div class="ui-card">
            <h4 class="ui-card-title">Topic Performance Overview</h4>
            <p class="ui-card-subtitle">Average score breakdown across practiced topics</p>
        </div>
        """, unsafe_allow_html=True)

        chart = df[["topic", "average_score"]].set_index("topic")
        st.bar_chart(chart)
    else:
        st.info("Complete practice quizzes to visualize your topic performance metrics here.")


# ==================================================
# REQUIREMENT 2: STUDY MATERIAL & PERSISTENT NOTES Q&A
# ==================================================

elif page == "Study Material":
    st.subheader("Study Material & Notes")
    st.caption("Upload course notes with background indexing, or directly ask questions from your completed study materials.")

    # Subject Selector and Add Subject Row
    sub_col1, sub_col2 = st.columns([0.7, 0.3])
    with sub_col1:
        subject = st.selectbox(
            "Select Subject",
            available_subjects,
            format_func=lambda s: format_subject(s)
        )
    with sub_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        with st.expander("+ Add New Subject"):
            new_subj_input = st.text_input("New Subject Name", placeholder="e.g. Cloud Computing", key="new_subj_input")
            if st.button("Add Subject", type="primary"):
                if new_subj_input.strip():
                    ok, msg = add_subject(new_subj_input)
                    if ok:
                        st.success(msg)
                        st.rerun()
                else:
                    st.warning("Please enter a subject name.")

    uploaded_file = st.file_uploader("Upload Notes (PDF)", type=["pdf"], key="pdf_uploader")

    if uploaded_file:
        existing_doc = check_duplicate_document(user_id, uploaded_file.name, subject)
        if existing_doc and existing_doc["status"] == "completed":
            st.info(f"ℹ️ '{uploaded_file.name}' is already indexed and ready in your library below. You can ask questions directly or re-index if updated.")

        if st.button("Process & Index Notes (Background)", type="primary"):
            try:
                # Save file immediately to user's isolated directory
                user_notes_dir = get_user_notes_dir(user_id, subject)
                pdf_path = user_notes_dir / uploaded_file.name
                file_bytes = uploaded_file.getbuffer()

                with open(pdf_path, "wb") as f:
                    f.write(file_bytes)

                # Start background indexing worker
                doc_id, job_id = start_background_indexing(
                    user_id=user_id,
                    filename=uploaded_file.name,
                    file_path=str(pdf_path),
                    subject=subject,
                    file_size=len(file_bytes)
                )

                st.success(f"Processing job started for '{uploaded_file.name}' in the background! You can navigate to other pages anytime.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start processing job: {e}")

    # --------------------------------------------------
    # REQUIREMENT 2: DEDICATED STUDY MATERIAL QUESTION-ANSWER AREA
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div class="ui-card" style="border-left: 4px solid #6366f1;">
        <h3 class="ui-card-title">📚 Ask Questions About Your Notes</h3>
        <p class="ui-card-subtitle">Select any of your uploaded completed documents and get grounded answers with page citations.</p>
    </div>
    """, unsafe_allow_html=True)

    completed_docs = get_user_completed_documents(user_id)

    if not completed_docs:
        st.info("No completed study materials yet. Upload and index a PDF above to ask questions.")
    else:
        doc_map = {f"📄 {d['filename']} ({format_subject(d['subject'])})": d for d in completed_docs}
        selected_label = st.selectbox(
            "Selected Document",
            list(doc_map.keys()),
            key="sm_selected_doc"
        )
        selected_doc = doc_map[selected_label]

        sm_q_col1, sm_q_col2 = st.columns([0.84, 0.16])
        with sm_q_col1:
            sm_question = st.text_input(
                "Question",
                placeholder=f"Ask a question about {selected_doc['filename']}...",
                key="sm_question_input",
                label_visibility="collapsed"
            )
        with sm_q_col2:
            sm_ask_btn = st.button("Ask Question", type="primary", use_container_width=True, key="sm_ask_btn")

        if sm_ask_btn:
            if not sm_question.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("Searching document notes..."):
                    results = search_user_notes(
                        question=sm_question.strip(),
                        user_id=user_id,
                        document_name=selected_doc["filename"],
                        subject=selected_doc["subject"],
                        top_k=5
                    )

                    if not results:
                        answer_text = "I couldn't find this information in the selected uploaded notes."
                        sources_list = []
                    else:
                        context = "\n\n".join(r["text"] for r in results)
                        sources_list = [{
                            "source": r["source"],
                            "page": r["page"],
                            "text": r["text"],
                            "score": r["score"]
                        } for r in results]
                        answer_text = ask_ai(sm_question.strip(), context=context)

                    st.session_state.study_material_qa_history.insert(0, {
                        "question": sm_question.strip(),
                        "answer": answer_text,
                        "document": selected_doc["filename"],
                        "sources": sources_list,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

        # Display Study Material Q&A Results
        if st.session_state.study_material_qa_history:
            for item in st.session_state.study_material_qa_history[:5]:
                st.markdown(f"""
                <div class="ui-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <strong style="color:#0f172a; font-size:1.02rem;">Q: {item['question']}</strong>
                        <span style="font-size:0.78rem; color:#64748b;">📄 {item['document']} &bull; {item['timestamp']}</span>
                    </div>
                    <div style="font-size:0.95rem; color:#334155; margin-bottom:8px; line-height:1.5;">{item['answer']}</div>
                """, unsafe_allow_html=True)

                if item["sources"]:
                    with st.expander("View Page Citations"):
                        for s in item["sources"]:
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-header">Source: 📄 <b>{s['source']}</b> &bull; Page {s['page']} (Relevance: {s['score']:.2f})</div>
                                <div>{s['text']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # PERSISTENT JOBS & DOCUMENT STATUS MONITOR
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("Document Processing Status")
    st.caption("Real-time persistent status of your uploaded documents and vector index generation.")

    col_ref1, col_ref2 = st.columns([0.85, 0.15])
    with col_ref2:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

    jobs = get_user_document_jobs(user_id=user_id)

    if not jobs:
        st.info("No documents uploaded yet. Upload a PDF above to create a vector index.")
    else:
        for job in jobs:
            st.markdown("""
            <div class="ui-card">
            """, unsafe_allow_html=True)

            status = job["status"]
            if status == "completed":
                badge_html = '<span class="badge-completed">🟢 COMPLETED</span>'
            elif status == "processing":
                badge_html = f'<span class="badge-processing">🟡 PROCESSING ({job["progress"]}%)</span>'
            elif status == "failed":
                badge_html = '<span class="badge-failed">🔴 FAILED</span>'
            else:
                badge_html = '<span class="badge-queued">⚪ QUEUED</span>'

            j_col1, j_col2 = st.columns([0.7, 0.3])
            with j_col1:
                st.markdown(f"**{job['filename']}** &nbsp; {badge_html}", unsafe_allow_html=True)
                st.caption(f"Subject: **{format_subject(job['subject'])}** &bull; Created: {job['created_at'][:19].replace('T', ' ')}")
            with j_col2:
                if job["completed_at"]:
                    st.caption(f"Completed: {job['completed_at'][:19].replace('T', ' ')}")

            if status == "processing":
                st.progress(job["progress"] / 100.0)

            if status == "failed" and job["error_message"]:
                st.error(f"Error: {job['error_message']}")

            st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# QUIZ & PRACTICE (MANUAL TOPIC, TARGET LEVEL, MIN 5 QUESTIONS)
# ==================================================

elif page == "Quiz & Practice":
    st.subheader("Adaptive Quiz & Practice")
    st.caption("Generate targeted multiple-choice practice questions specifically for your chosen topic and target level.")

    q_col1, q_col2 = st.columns(2)
    with q_col1:
        subject = st.selectbox(
            "Subject",
            available_subjects,
            format_func=lambda s: format_subject(s),
            key="quiz_sub_select"
        )
        topic = st.text_input("Topic", placeholder="e.g. Transaction, Normalization, Deadlock, TCP/IP...", key="quiz_top_input")

    with q_col2:
        target_level = st.selectbox(
            "Target Level",
            ["Beginner", "Intermediate", "Advanced"],
            index=1,
            key="quiz_target_level"
        )
        number = st.slider("Number of Questions", min_value=5, max_value=15, value=5, key="quiz_num_slider")

    if st.button("Generate Quiz", type="primary"):
        if not topic.strip():
            st.warning("⚠️ Please enter a topic before generating the quiz.")
        else:
            with st.spinner(f"Generating {target_level} practice questions for '{topic}'..."):
                try:
                    quiz_data = generate_quiz(subject, topic.strip(), target_level, number)
                    st.session_state.quiz = quiz_data["questions"]
                    st.session_state.quiz_subject = subject
                    st.session_state.quiz_topic = topic.strip()
                    st.session_state.quiz_difficulty = target_level
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score_details = None
                    st.success(f"Quiz generated successfully for topic: **{topic.strip()}** ({target_level} Level).")
                except Exception as e:
                    st.error(f"Quiz generation error: {e}")

    if st.session_state.quiz:
        st.markdown("---")
        quiz = st.session_state.quiz

        for i, question in enumerate(quiz):
            st.markdown(f"**Question {i + 1}:** {question['question']}")

            current_choice = st.session_state.quiz_answers.get(i)

            answer = st.radio(
                f"Select answer for Question {i + 1}",
                options=question["options"],
                index=current_choice,
                key=f"mcq_q_{i}_{len(quiz)}",
                label_visibility="collapsed"
            )

            if answer is not None and answer in question["options"]:
                st.session_state.quiz_answers[i] = question["options"].index(answer)

            st.markdown("<br>", unsafe_allow_html=True)

        if not st.session_state.quiz_submitted:
            if st.button("Submit Quiz", type="primary"):
                is_complete, missing_questions = validate_all_answered(quiz, st.session_state.quiz_answers)

                if not is_complete:
                    missing_str = ", ".join([f"Question {q}" for q in missing_questions])
                    st.warning(f"⚠️ Please answer all questions before submitting. Unanswered: **{missing_str}**")
                else:
                    correct, total, score, details = calculate_score(
                        quiz,
                        st.session_state.quiz_answers
                    )

                    answers_for_db = []
                    for detail in details:
                        answers_for_db.append({
                            "question": detail["question"],
                            "selected_answer": detail["selected_answer"],
                            "correct_answer": detail["correct_answer"],
                            "is_correct": detail["is_correct"]
                        })

                    save_quiz_result(
                        st.session_state.quiz_subject,
                        st.session_state.quiz_topic,
                        st.session_state.quiz_difficulty,
                        score,
                        total,
                        correct,
                        answers_for_db,
                        user_id=user_id
                    )

                    st.session_state.quiz_submitted = True
                    st.session_state.quiz_score_details = (correct, total, score, details)
                    st.rerun()

        if st.session_state.quiz_submitted and st.session_state.quiz_score_details:
            correct, total, score, details = st.session_state.quiz_score_details

            st.markdown(f"""
            <div class="ui-card">
                <h3 class="ui-card-title">Quiz Results: {st.session_state.quiz_topic} ({st.session_state.quiz_difficulty})</h3>
                <p class="ui-card-subtitle">Score: <strong>{correct}/{total}</strong> ({score:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)

            for idx, detail in enumerate(details):
                st.markdown("---")
                if detail["is_correct"]:
                    st.markdown(f'<span class="status-badge-correct">CORRECT</span> &nbsp; **{detail["question"]}**', unsafe_allow_html=True)
                    st.write(f"Your answer: {detail['selected_answer']}")
                else:
                    st.markdown(f'<span class="status-badge-wrong">INCORRECT</span> &nbsp; **{detail["question"]}**', unsafe_allow_html=True)
                    st.write(f"Your answer: {detail['selected_answer']}")
                    st.write(f"Correct answer: {detail['correct_answer']}")

                    with st.expander(f"Explain Solution for Question {idx + 1}"):
                        explanation = explain_mistake(
                            detail["question"],
                            detail["selected_answer"],
                            detail["correct_answer"],
                            st.session_state.quiz_topic
                        )
                        st.write(explanation)


# ==================================================
# PERFORMANCE & ANALYTICS
# ==================================================

elif page == "Performance & Analytics":
    st.subheader("Performance & Analytics")
    st.caption("Track your learning progress, mastery metrics, and predictive insights.")

    rows = get_topic_performance(user_id=user_id)
    df = load_performance_data(rows)
    df = analyze_performance(df)

    if df.empty:
        st.info("No quiz attempts recorded yet. Complete quizzes to generate performance analytics.")
    else:
        st.dataframe(df, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ui-card">
            <h4 class="ui-card-title">Performance by Topic</h4>
            <p class="ui-card-subtitle">Average scores by topic</p>
        </div>
        """, unsafe_allow_html=True)

        chart = df[["topic", "average_score"]].set_index("topic")
        st.bar_chart(chart)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ui-card">
            <h4 class="ui-card-title">Topic Mastery Analytics</h4>
            <p class="ui-card-subtitle">Machine learning model evaluation on study retention</p>
        </div>
        """, unsafe_allow_html=True)

        model, metrics = train_model(df)
        if metrics:
            st.metric("Mastery Classification Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
            with st.expander("Detailed Evaluation Report"):
                st.text(metrics["report"])
        else:
            st.info("More quiz attempts are needed to train the predictive retention model.")


# ==================================================
# RECOMMENDATIONS & PERSONALIZED STUDY PLANNER
# ==================================================

elif page == "Recommendations & Study Plan":
    st.subheader("Personalized Study Planner & Recommendations")
    st.caption("Design custom study schedules with exact day allocations, minute-level precision, and PDF export.")

    rows = get_topic_performance(user_id=user_id)
    df = load_performance_data(rows)
    df = analyze_performance(df)

    if not df.empty:
        recommendations = generate_recommendations(df)
        with st.expander("📊 View Your Personalized Topic Recommendations", expanded=False):
            for item in recommendations:
                priority = item["priority"]
                if priority == "HIGH":
                    pill = '<span class="badge-failed">HIGH PRIORITY</span>'
                elif priority == "MEDIUM":
                    pill = '<span class="badge-processing">MEDIUM PRIORITY</span>'
                else:
                    pill = '<span class="badge-completed">MASTERY MAINTAINED</span>'

                st.markdown(f"""
                <div class="ui-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <strong style="font-size:1.05rem;">{item['topic']}</strong>
                        <div>{pill} &nbsp; <span style="font-weight:600;">{item['score']:.1f}%</span></div>
                    </div>
                    <div style="font-size:0.9rem; color:#64748b;">{item['recommendation']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="ui-card">
        <h3 class="ui-card-title">Create Personalized Study Schedule</h3>
        <p class="ui-card-subtitle">Set your target topic, proficiency level, available study days, and exact daily study duration.</p>
    </div>
    """, unsafe_allow_html=True)

    p_col1, p_col2 = st.columns([0.65, 0.35])
    with p_col1:
        planner_topic = st.text_input(
            "Study Topic",
            placeholder="e.g. Transaction, Operating Systems Memory Management, Machine Learning Regression...",
            key="planner_topic_input"
        )
    with p_col2:
        planner_level = st.selectbox(
            "Target Level",
            ["Beginner", "Intermediate", "Advanced"],
            index=1,
            key="planner_level_select"
        )

    st.markdown("##### 📅 Available Study Days")
    st.caption("Select the specific days of the week you plan to study.")

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_cols = st.columns(7)
    selected_days = []

    for i, day in enumerate(weekdays):
        with day_cols[i]:
            is_def = day in ["Monday", "Wednesday", "Saturday"]
            checked = st.checkbox(day[:3], value=is_def, key=f"chk_day_{day}")
            if checked:
                selected_days.append(day)

    st.markdown("<br>", unsafe_allow_html=True)

    day_schedules = []
    total_weekly_minutes = 0

    if selected_days:
        st.markdown("##### ⏱️ Daily Study Duration (Hours & Minutes)")
        st.caption("Specify your available study time for each selected day.")

        minute_options = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

        for day in selected_days:
            dur_col1, dur_col2, dur_col3, dur_col4 = st.columns([0.25, 0.25, 0.25, 0.25])
            with dur_col1:
                st.markdown(f"<div style='padding-top:8px;'><b>{day}</b></div>", unsafe_allow_html=True)
            with dur_col2:
                d_hours = st.number_input(f"Hours ({day})", min_value=0, max_value=12, value=2 if day != "Saturday" else 3, key=f"hrs_{day}", label_visibility="collapsed")
            with dur_col3:
                def_min_idx = 6 if day == "Monday" else 0  # 30 mins for Monday
                d_mins = st.selectbox(f"Minutes ({day})", options=minute_options, index=def_min_idx, key=f"mins_{day}", label_visibility="collapsed")
            with dur_col4:
                d_total_mins = (d_hours * 60) + d_mins
                total_weekly_minutes += d_total_mins
                st.markdown(f"<div style='padding-top:8px; color:#64748b;'>{format_duration(d_hours, d_mins)} ({d_total_mins}m)</div>", unsafe_allow_html=True)

            day_schedules.append({
                "day": day,
                "hours": int(d_hours),
                "minutes": int(d_mins),
                "total_minutes": int(d_total_mins)
            })

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate Study Plan", type="primary"):
        if not planner_topic.strip():
            st.warning("⚠️ Please enter a study topic.")
        elif not selected_days:
            st.warning("⚠️ Please select at least one available study day.")
        elif total_weekly_minutes <= 0:
            st.warning("⚠️ Please enter a valid study duration greater than 0 minutes.")
        else:
            with st.spinner(f"Generating customized {planner_level} study schedule for '{planner_topic}'..."):
                try:
                    plan_content = generate_study_plan(
                        topic=planner_topic.strip(),
                        target_level=planner_level,
                        day_schedules=day_schedules
                    )

                    summary_items = [f"{s['day']}: {format_duration(s['hours'], s['minutes'])}" for s in day_schedules]
                    schedule_summary = ", ".join(summary_items)

                    save_study_plan(
                        user_id=user_id,
                        topic=planner_topic.strip(),
                        target_level=planner_level,
                        schedule_summary=schedule_summary,
                        plan_content=plan_content
                    )

                    student_name = current_user.get("name", "Student")
                    pdf_bytes = generate_study_plan_pdf(
                        student_name=student_name,
                        topic=planner_topic.strip(),
                        target_level=planner_level,
                        day_schedules=day_schedules,
                        plan_content=plan_content
                    )

                    st.session_state.current_study_plan = {
                        "topic": planner_topic.strip(),
                        "target_level": planner_level,
                        "day_schedules": day_schedules,
                        "schedule_summary": schedule_summary,
                        "plan_content": plan_content
                    }
                    st.session_state.current_plan_pdf = pdf_bytes
                    st.success("Personalized study plan created successfully!")
                except Exception as e:
                    st.error(f"Failed to generate study plan: {e}")

    if st.session_state.current_study_plan:
        plan_data = st.session_state.current_study_plan
        st.markdown("---")

        plan_head_col1, plan_head_col2 = st.columns([0.7, 0.3])
        with plan_head_col1:
            st.markdown(f"### Study Plan: {plan_data['topic']} ({plan_data['target_level']})")
            st.caption(f"Schedule: {plan_data['schedule_summary']}")
        with plan_head_col2:
            if st.session_state.current_plan_pdf:
                st.download_button(
                    label="📥 Download Study Plan as PDF",
                    data=st.session_state.current_plan_pdf,
                    file_name=f"Study_Plan_{plan_data['topic'].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )

        st.markdown("""
        <div class="ui-card">
        """, unsafe_allow_html=True)
        st.markdown(plan_data["plan_content"])
        st.markdown("</div>", unsafe_allow_html=True)

    past_plans = get_user_study_plans(user_id=user_id)
    if past_plans:
        with st.expander("📚 View Previously Generated Study Plans"):
            for p in past_plans:
                st.markdown(f"**{p['topic']}** ({p['target_level']}) — *{p['created_at'][:19].replace('T', ' ')}*")
                st.caption(f"Schedule: {p['schedule_summary']}")
                with st.expander(f"Read Plan: {p['topic']}"):
                    st.markdown(p["plan_content"])
                st.markdown("---")


# ==================================================
# PROFILE
# ==================================================

elif page == "Profile":
    st.subheader("Student Profile")

    profile = get_profile(user_id=user_id)
    current_name = profile[0] if profile else ""
    current_course = profile[1] if profile else ""
    current_semester = profile[2] if profile else 1

    with st.container():
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        name = st.text_input("Full Name", current_name, placeholder="e.g. Alex Johnson")
        course = st.text_input("Degree / Course", current_course, placeholder="e.g. Computer Science & Engineering")
        semester = st.number_input("Current Semester", min_value=1, max_value=12, value=int(current_semester))

        if st.button("Save Profile", type="primary"):
            if name.strip():
                save_profile(name, course, semester, user_id=user_id)
                if st.session_state.user:
                    st.session_state.user["name"] = name
                    st.session_state.user["course"] = course
                    st.session_state.user["semester"] = semester
                st.success("Profile saved successfully.")
            else:
                st.warning("Please enter your name.")
        st.markdown('</div>', unsafe_allow_html=True)