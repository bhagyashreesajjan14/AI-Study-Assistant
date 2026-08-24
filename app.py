import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

from config import (
    NOTES_DIR,
    VECTOR_DIR,
    SUPPORTED_FILE_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
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
    get_document_by_id,
    get_latest_user_document,
    check_duplicate_document,
    save_study_plan,
    get_latest_study_plan,
    get_user_study_plans,
    create_note,
    update_note,
    delete_note,
    get_user_notes,
    get_note_by_id,
    create_quiz_job,
    update_quiz_job_status,
    get_user_quiz_jobs,
    get_quiz_job_by_id,
    get_latest_quiz_job,
    delete_quiz_job,
    delete_user_document
)

from ai import (
    ask_ai,
    ask_ai_chat,
    ask_ai_chat_stream,
    generate_quiz,
    generate_quiz_from_material,
    generate_summary,
    generate_flashcards,
    generate_notes_explanation,
    explain_mistake,
    start_background_quiz_generation,
    fix_mermaid_syntax
)

from rag import (
    create_chunks,
    build_index,
    search,
    search_user_notes,
    save_index,
    load_index,
    start_background_indexing,
    load_user_subject_index,
    get_full_document_text,
    delete_document_data
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

from pdf_generator import (
    generate_response_pdf,
    generate_chat_pdf,
    derive_pdf_filename
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
    "active_quiz_job_id": None,
    "active_chat_id": None,
    "pending_prompt": None,
    "current_study_plan": None,
    "current_plan_pdf": None,
    "study_material_qa_history": [],
    "active_document_id": None,
    "active_doc_summary": None,
    "active_doc_flashcards": None,
    "active_doc_notes": None,
    "active_card_idx": 0,
    "show_card_back": False,
    "active_note_id": None,
    "note_search_query": "",
    "material_uploader_version": 0,
    "pending_delete_doc_id": None
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

.chat-disclaimer {
    text-align: center;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.75rem;
}

.flashcard-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.75rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    text-align: center;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.flashcard-front {
    font-size: 1.15rem;
    font-weight: 600;
    color: #0f172a;
    line-height: 1.5;
}

.flashcard-back {
    font-size: 1.05rem;
    color: #334155;
    line-height: 1.6;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px dashed #cbd5e1;
}

.flashcard-tag {
    display: inline-block;
    background: #e0e7ff;
    color: #4338ca;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

.doc-info-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.25rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    margin-bottom: 1.2rem;
}

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
# 1. AUTHENTICATION (LOGIN & SIGNUP)
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
            if st.button("Sign In", type="primary", width="stretch"):
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
            if st.button("Create Account", type="primary", width="stretch"):
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
# APP SHELL & ISOLATION
# ==================================================

current_user = st.session_state.user or {}
user_id = current_user.get("id", 1)

profile_record = get_profile(user_id=user_id)
if profile_record and profile_record[0]:
    user_display = profile_record[0]
    user_course = profile_record[1] if profile_record[1] else current_user.get("course", "")
    user_semester = profile_record[2] if profile_record[2] else current_user.get("semester", 1)
else:
    user_display = current_user.get("name", "Student")
    user_course = current_user.get("course", "")
    user_semester = current_user.get("semester", 1)

badge_text = f"{user_display} &bull; {user_course}" if user_course else user_display
st.markdown(f"""
<div class="app-brand-bar">
    <div>
        <h1 class="app-brand-title">AI Study Assistant</h1>
    </div>
    <div>
        <span class="app-brand-badge">{badge_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)

available_subjects = get_all_subjects()

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

st.sidebar.markdown("---")
st.sidebar.caption(f"Logged in as **{user_display}** (`@{current_user.get('username', '')}`)")
if st.sidebar.button("Sign Out", width="stretch"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state.authenticated = False
    st.rerun()


# ==================================================
# AI TUTOR
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

    if st.sidebar.button("+ New Chat", width="stretch", type="primary"):
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

        btn_label = f"• {chat['title']}" if is_active else chat["title"]

        with col_chat:
            if st.button(btn_label, key=f"chat_btn_{chat['id']}", width="stretch"):
                st.session_state.active_chat_id = chat["id"]
                st.rerun()

        with col_del:
            if st.button("x", key=f"chat_del_{chat['id']}", help="Delete Chat"):
                delete_chat_session(chat["id"], user_id=user_id)
                if st.session_state.active_chat_id == chat["id"]:
                    st.session_state.active_chat_id = None
                st.rerun()

    active_chat_id = st.session_state.active_chat_id

    if active_chat_id is not None:
        messages = get_chat_messages(active_chat_id, user_id=user_id)

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([0.25, 0.25, 0.25, 0.25])
        with ctrl_col1:
            mode = st.selectbox(
                "Tutor Mode",
                ["Explain", "Simple Explanation", "Example", "Exam Preparation"],
                key="tutor_mode"
            )
        with ctrl_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            notes_only = st.checkbox("Reference uploaded notes only", value=False, key="tutor_notes_only")
        with ctrl_col3:
            if notes_only:
                completed_docs = get_user_completed_documents(user_id)
                doc_options = ["All my completed notes"] + [d["filename"] for d in completed_docs]
                
                # Check valid selection string
                if "tutor_ref_doc" in st.session_state and st.session_state.tutor_ref_doc not in doc_options:
                    del st.session_state["tutor_ref_doc"]
                
                selected_ref_doc = st.selectbox("Reference Document", doc_options, key="tutor_ref_doc")
            else:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                active_subject_name = format_subject(chat_subject) if chat_subject != 'General' else 'General Knowledge'
                st.caption(f"Context: **{active_subject_name}**")

        with ctrl_col4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if messages:
                try:
                    active_chat_obj = next((c for c in chats if c["id"] == active_chat_id), None)
                    chat_title_str = active_chat_obj["title"] if active_chat_obj else "AI Tutor Chat"
                    full_chat_pdf = generate_chat_pdf(
                        session_title=chat_title_str,
                        subject=chat_subject,
                        student_name=user_display,
                        messages=messages
                    )
                    st.download_button(
                        label="📥 Download Full Chat as PDF",
                        data=full_chat_pdf,
                        file_name=derive_pdf_filename(chat_title_str, prefix="Chat"),
                        mime="application/pdf",
                        key=f"dl_full_chat_{active_chat_id}",
                        width="stretch"
                    )
                except Exception as pdf_err:
                    st.caption(f"PDF export unavailable: {pdf_err}")

        st.markdown("---")

        if len(messages) == 0:
            st.markdown("""
            <div class="welcome-container">
                <h2 class="welcome-title">How can I help you study today?</h2>
                <p class="welcome-subtitle">Ask questions, request concept breakdowns, or explore your uploaded course notes.</p>
            </div>
            """, unsafe_allow_html=True)

            sugg_col1, sugg_col2 = st.columns(2)
            with sugg_col1:
                if st.button("Explain Normalization with 1NF, 2NF, 3NF examples", width="stretch"):
                    st.session_state.pending_prompt = "Explain Database Normalization with 1NF, 2NF, and 3NF examples clearly."
                    st.rerun()
                if st.button("Compare Processes vs Threads in Operating Systems", width="stretch"):
                    st.session_state.pending_prompt = "What is the difference between Processes and Threads in Operating Systems?"
                    st.rerun()

            with sugg_col2:
                if st.button("Break down ACID properties with real-world scenarios", width="stretch"):
                    st.session_state.pending_prompt = "Explain ACID properties in DBMS with real-world scenarios and examples."
                    st.rerun()
                if st.button("Summarize key concepts from my study material", width="stretch"):
                    st.session_state.pending_prompt = "Provide a comprehensive summary of the core concepts in this subject."
                    st.rerun()

        for msg_idx, msg in enumerate(messages):
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(fix_mermaid_syntax(msg["content"]))
                
                if role == "assistant":
                    act_c1, act_c2 = st.columns([0.72, 0.28])
                    with act_c2:
                        try:
                            response_title = f"{format_subject(chat_subject)} Study Notes"
                            if msg_idx > 0 and messages[msg_idx - 1]["role"] == "user":
                                response_title = messages[msg_idx - 1]["content"][:40]
                                
                            single_pdf_bytes = generate_response_pdf(
                                title=response_title,
                                content=msg["content"],
                                student_name=user_display,
                                subject=chat_subject
                            )
                            pdf_fname = derive_pdf_filename(response_title)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=single_pdf_bytes,
                                file_name=pdf_fname,
                                mime="application/pdf",
                                key=f"dl_resp_pdf_{msg.get('id', msg_idx)}",
                                width="stretch"
                            )
                        except Exception:
                            pass

                if msg.get("sources"):
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
                st.markdown(prompt)

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
                            st.markdown(answer)
                        save_chat_message(active_chat_id, "assistant", answer, sources=[], user_id=user_id)
                        st.rerun()

            with st.chat_message("assistant"):
                full_history = []
                for m in messages:
                    full_history.append({
                        "role": m["role"],
                        "content": m["content"]
                    })
                full_history.append({
                    "role": "user",
                    "content": prompt
                })

                try:
                    stream_gen = ask_ai_chat_stream(
                        messages=full_history,
                        context=context,
                        mode=mode,
                        notes_only=notes_only
                    )
                    full_answer = st.write_stream(stream_gen)

                    if sources:
                        with st.expander("Reference Sources"):
                            for source in sources:
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">Source: 📄 <b>{source['source']}</b> &bull; Page {source['page']} (Relevance: {source['score']:.2f})</div>
                                    <div>{source['text']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    save_chat_message(active_chat_id, "assistant", fix_mermaid_syntax(full_answer), sources=sources, user_id=user_id)

                    if len(messages) == 0:
                        new_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
                        rename_chat_session(active_chat_id, new_title, user_id=user_id)

                    # Smooth finish without jarring rerun
                except Exception as e:
                    st.error(f"Response generation error: {e}")

        st.markdown(
            '<div class="chat-disclaimer">AI Study Assistant provides verified educational guidance. Responses can be saved or exported as real PDF notes at any time.</div>',
            unsafe_allow_html=True
        )


# ==================================================
# DASHBOARD
# ==================================================

elif page == "Dashboard":
    profile = get_profile(user_id=user_id)
    dash_name = profile[0] if profile and profile[0] else user_display
    dash_course = profile[1] if profile and profile[1] else user_course
    dash_sem = profile[2] if profile and profile[2] else user_semester

    st.markdown(f"""
    <div class="ui-card">
        <h3 class="ui-card-title">Welcome back, {dash_name}</h3>
        <p class="ui-card-subtitle">{dash_course} &bull; Semester {dash_sem}</p>
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

    user_jobs = get_user_document_jobs(user_id=user_id)
    active_doc_jobs = [j for j in user_jobs if j["status"] in ("queued", "processing")]
    user_quiz_jobs = get_user_quiz_jobs(user_id=user_id)
    active_quiz_jobs = [q for q in user_quiz_jobs if q["status"] in ("pending", "processing")]

    if active_doc_jobs or active_quiz_jobs:
        st.markdown("""
        <div class="ui-card" style="border-left: 4px solid #6366f1;">
            <h4 class="ui-card-title">⚡ Active Background Tasks</h4>
            <p class="ui-card-subtitle">Your document indexing and quiz generation tasks continue running independently in the background.</p>
        </div>
        """, unsafe_allow_html=True)

        for job in active_doc_jobs:
            st.write(f"📄 **Document Indexing:** {job['filename']} ({format_subject(job['subject'])}) — {job['status'].upper()} ({job['progress']}%)")
            st.progress(job["progress"] / 100.0)

        for qjob in active_quiz_jobs:
            st.write(f"🎯 **Quiz Generation:** {qjob['topic']} ({qjob['difficulty']}) — {qjob['status'].upper()} ({qjob['progress']}%)")
            st.progress(qjob["progress"] / 100.0)

        time.sleep(2)
        st.rerun()

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
# STUDY MATERIAL & MULTI-NOTES STUDIO
# ==================================================

elif page == "Study Material":
    st.subheader("Study Material & Multi-Notes Studio")
    st.caption("Create unlimited persistent notes, upload learning materials across all formats, and use them seamlessly.")

    tab_library, tab_notes_studio = st.tabs([
        "📚 Uploaded Materials & Learning Hub",
        "📝 Multi-Notes Studio"
    ])

    with tab_library:
        st.markdown("""
        <div class="ui-card">
            <h4 class="ui-card-title">📤 Upload New Learning Material</h4>
            <p class="ui-card-subtitle">Supported formats: PDF (with OCR fallback), Word DOCX, DOC, Images (PNG, JPG, WEBP), Plaintext, Markdown (Max 25MB).</p>
        </div>
        """, unsafe_allow_html=True)

        up_col1, up_col2 = st.columns([0.65, 0.35])
        with up_col1:
            subject = st.selectbox(
                "Select Subject",
                available_subjects,
                format_func=lambda s: format_subject(s),
                key="sm_subject_select"
            )
        with up_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            with st.expander("+ Add New Subject"):
                new_subj_input = st.text_input("New Subject Name", placeholder="e.g. Cloud Computing", key="new_subj_input")
                if st.button("Add Subject", type="primary", key="btn_add_subj"):
                    if new_subj_input.strip():
                        ok, msg = add_subject(new_subj_input)
                        if ok:
                            st.success(msg)
                            st.rerun()
                    else:
                        st.warning("Please enter a subject name.")

        uploader_version = st.session_state.get("material_uploader_version", 0)
        uploaded_file = st.file_uploader(
            "Choose Document or Screenshot/Image File",
            type=SUPPORTED_FILE_EXTENSIONS,
            key=f"multi_material_uploader_{uploader_version}",
            help=f"Upload PDF documents, slides, images, or notes (Max {MAX_UPLOAD_SIZE_MB}MB)."
        )

        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)
            ext = Path(uploaded_file.name).suffix.lower()

            if len(file_bytes) == 0:
                st.error("⚠️ The uploaded file is empty. Please upload a valid document or image.")
            elif file_size_mb > MAX_UPLOAD_SIZE_MB:
                st.error(f"⚠️ File size ({file_size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_MB} MB.")
            elif ext[1:] not in SUPPORTED_FILE_EXTENSIONS:
                st.error(f"⚠️ Unsupported file type '{ext}'. Supported formats: {', '.join(SUPPORTED_FILE_EXTENSIONS)}")
            else:
                is_img = ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
                file_desc = "Screenshot / Image" if is_img else "Document"

                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 14px 18px; margin: 12px 0;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <span style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">📄 {uploaded_file.name}</span>
                            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">
                                Type: <b>{file_desc} ({ext.upper()})</b> &bull; Size: <b>{file_size_mb:.2f} MB</b> &bull; Target Subject: <b>{format_subject(subject)}</b>
                            </div>
                        </div>
                        <span style="background: #e0f2fe; color: #0369a1; font-weight: 600; font-size: 0.8rem; padding: 4px 10px; border-radius: 20px; border: 1px solid #bae6fd;">
                            Ready to Process
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                existing_doc = check_duplicate_document(user_id, uploaded_file.name, subject)
                if existing_doc and existing_doc["status"] == "completed":
                    st.info(f"ℹ️ '{uploaded_file.name}' is already indexed in your library. Re-processing will refresh its index.")

                up_btn_col1, up_btn_col2 = st.columns([0.72, 0.28])
                force_ocr = False
                if ext[1:] == "pdf":
                    force_ocr = st.checkbox("Force OCR on PDF (Check this if your PDF contains handwritten notes on slides)", value=False)
                
                with up_btn_col1:
                    if st.button("🚀 Process & Extract Content (OCR & Indexing)", type="primary", width="stretch", key="btn_process_upload"):
                        try:
                            user_notes_dir = get_user_notes_dir(user_id, subject)
                            saved_path = user_notes_dir / uploaded_file.name

                            with open(saved_path, "wb") as f:
                                f.write(file_bytes)

                            doc_id, job_id = start_background_indexing(
                                user_id=user_id,
                                filename=uploaded_file.name,
                                file_path=str(saved_path),
                                subject=subject,
                                file_size=len(file_bytes),
                                file_type=ext[1:],
                                force_ocr=force_ocr
                            )

                            st.session_state.active_document_id = doc_id
                            st.session_state.active_doc_summary = None
                            st.session_state.active_doc_flashcards = None
                            st.session_state.active_doc_notes = None

                            st.success("Processing started: OCR and indexing running in background.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to start processing: {e}")

                with up_btn_col2:
                    if st.button("🗑️ Discard / Remove", width="stretch", key="btn_discard_upload"):
                        st.session_state.material_uploader_version = uploader_version + 1
                        st.rerun()

        st.markdown("---")
        user_docs = get_user_documents(user_id)

        if user_docs:
            with st.expander(f"📁 Manage Uploaded Files ({len(user_docs)} {'file' if len(user_docs) == 1 else 'files'})", expanded=False):
                st.caption("View and manage all uploaded materials. You can independently delete any file to remove its physical data, extracted text, and AI embeddings.")
                for d in user_docs:
                    d_id = d["id"]
                    d_name = d["filename"]
                    d_subj = d["subject"]
                    d_size_kb = d["file_size"] / 1024
                    d_status = d.get("status", "processing")
                    
                    status_badge = (
                        '<span class="badge-completed">🟢 Completed</span>' if d_status == "completed"
                        else '<span class="badge-processing">🟡 Processing</span>' if d_status == "processing"
                        else '<span class="badge-failed">🔴 Failed</span>' if d_status == "failed"
                        else '<span class="badge-queued">⚪ Queued</span>'
                    )

                    row_c1, row_c2, row_c3 = st.columns([0.62, 0.20, 0.18])
                    with row_c1:
                        st.markdown(f"**📄 {d_name}** &bull; <small>{format_subject(d_subj)} &bull; {d_size_kb:.1f} KB</small>", unsafe_allow_html=True)
                    with row_c2:
                        st.markdown(status_badge, unsafe_allow_html=True)
                    with row_c3:
                        if st.button("🗑️ Delete", key=f"btn_del_row_{d_id}", width="stretch"):
                            st.session_state.pending_delete_doc_id = d_id
                            st.rerun()

                    if st.session_state.get("pending_delete_doc_id") == d_id:
                        st.warning(f"⚠️ **Confirm Deletion**: Are you sure you want to permanently delete '**{d_name}**'?\n\nThis will remove the file, extracted text, AI vector embeddings, and search index.")
                        conf_c1, conf_c2, _ = st.columns([0.25, 0.25, 0.5])
                        with conf_c1:
                            if st.button("🗑️ Confirm Delete", type="primary", key=f"conf_del_btn_{d_id}", width="stretch"):
                                ok, msg = delete_document_data(
                                    user_id=user_id,
                                    document_id=d_id,
                                    filename=d_name,
                                    subject=d_subj,
                                    file_path=d.get("file_path")
                                )
                                st.session_state.pending_delete_doc_id = None
                                if ok:
                                    if st.session_state.active_document_id == d_id:
                                        st.session_state.active_document_id = None
                                        st.session_state.active_doc_summary = None
                                        st.session_state.active_doc_flashcards = None
                                        st.session_state.active_doc_notes = None
                                    if "active_doc_picker" in st.session_state:
                                        del st.session_state["active_doc_picker"]
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        with conf_c2:
                            if st.button("Cancel", key=f"cancel_del_btn_{d_id}", width="stretch"):
                                st.session_state.pending_delete_doc_id = None
                                st.rerun()
                        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

        if not user_docs:
            st.session_state.active_document_id = None
            if "active_doc_picker" in st.session_state:
                del st.session_state["active_doc_picker"]
            st.info("💡 No learning materials uploaded yet. Upload a document or screenshot above to activate the learning hub.")
        else:
            st.markdown("### 📚 Active Study Material Hub")
            st.caption("Select any uploaded document or screenshot to ask questions, view AI summaries, practice flashcards, or generate notes.")

            doc_dict = {f"📄 {d['filename']} ({format_subject(d['subject'])})": d for d in user_docs}
            doc_labels = list(doc_dict.keys())

            selected_idx = 0
            matching_indices = [i for i, d in enumerate(user_docs) if d["id"] == st.session_state.active_document_id]
            
            if matching_indices:
                selected_idx = matching_indices[0]
            else:
                selected_idx = 0
                st.session_state.active_document_id = user_docs[0]["id"]
                if "active_doc_picker" in st.session_state and st.session_state.active_doc_picker not in doc_labels:
                    del st.session_state["active_doc_picker"]

            selected_idx = max(0, min(selected_idx, len(doc_labels) - 1))

            active_doc_label = st.selectbox(
                "Selected Active Material",
                doc_labels,
                index=selected_idx,
                key="active_doc_picker"
            )
            active_doc = doc_dict[active_doc_label]
            st.session_state.active_document_id = active_doc["id"]

            user_jobs = get_user_document_jobs(user_id)
            active_job = next((j for j in user_jobs if j.get("document_id") == active_doc["id"]), None)
            current_status = active_job["status"] if active_job else active_doc.get("status", "completed")
            current_progress = active_job["progress"] if active_job else 100
            error_msg = active_job.get("error_message") if active_job else None

            if current_status == "completed":
                badge_html = '<span class="badge-completed">🟢 COMPLETED & READY</span>'
            elif current_status == "processing":
                if current_progress < 30:
                    step_name = "INITIALIZING"
                elif current_progress < 50:
                    step_name = "EXTRACTING TEXT & OCR"
                elif current_progress < 70:
                    step_name = "CHUNKING DOCUMENT"
                elif current_progress < 90:
                    step_name = "GENERATING AI EMBEDDINGS"
                else:
                    step_name = "BUILDING SEARCH INDEX"
                badge_html = f'<span class="badge-processing">🟡 {step_name} ({current_progress}%)</span>'
            elif current_status == "failed":
                badge_html = '<span class="badge-failed">🔴 PROCESSING FAILED</span>'
            else:
                badge_html = '<span class="badge-queued">⚪ QUEUED</span>'

            banner_col1, banner_col2 = st.columns([0.80, 0.20])
            with banner_col1:
                st.markdown(f"""
                <div class="doc-info-banner">
                    <div>
                        <strong>{active_doc['filename']}</strong> &bull; Subject: <em>{format_subject(active_doc['subject'])}</em> &bull; Size: {active_doc['file_size'] / 1024:.1f} KB
                    </div>
                    <div>
                        {badge_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with banner_col2:
                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Delete File", key=f"btn_del_active_hub_{active_doc['id']}", width="stretch"):
                    st.session_state.pending_delete_doc_id = active_doc["id"]
                    st.rerun()

            if st.session_state.get("pending_delete_doc_id") == active_doc["id"]:
                st.warning(f"⚠️ **Confirm Deletion**: Are you sure you want to permanently delete active material '**{active_doc['filename']}**'? All extracted text and AI search indexes will be cleared.")
                c_hub1, c_hub2, _ = st.columns([0.25, 0.25, 0.5])
                with c_hub1:
                    if st.button("🗑️ Confirm Delete", type="primary", key=f"conf_del_hub_{active_doc['id']}", width="stretch"):
                        ok, msg = delete_document_data(
                            user_id=user_id,
                            document_id=active_doc["id"],
                            filename=active_doc["filename"],
                            subject=active_doc["subject"],
                            file_path=active_doc.get("file_path")
                        )
                        st.session_state.pending_delete_doc_id = None
                        if ok:
                            st.session_state.active_document_id = None
                            st.session_state.active_doc_summary = None
                            st.session_state.active_doc_flashcards = None
                            st.session_state.active_doc_notes = None
                            if "active_doc_picker" in st.session_state:
                                del st.session_state["active_doc_picker"]
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                with c_hub2:
                    if st.button("Cancel", key=f"cancel_del_hub_{active_doc['id']}", width="stretch"):
                        st.session_state.pending_delete_doc_id = None
                        st.rerun()

            if current_status == "processing":
                st.progress(current_progress / 100.0)
                p_col1, p_col2, p_col3 = st.columns([0.6, 0.2, 0.2])
                with p_col1:
                    st.info(f"⏳ Processing in background ({current_progress}%)...")
                with p_col2:
                    if st.button("🔄 Refresh Status", width="stretch", key="btn_refresh_proc"):
                        st.rerun()
                with p_col3:
                    if st.button("⚡ Force Re-Index", width="stretch", key="btn_force_reindex"):
                        try:
                            start_background_indexing(
                                user_id=user_id,
                                filename=active_doc["filename"],
                                file_path=active_doc["file_path"],
                                subject=active_doc["subject"],
                                file_size=active_doc["file_size"],
                                file_type=active_doc.get("file_type", "pdf")
                            )
                            st.success("Re-indexing started!")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Re-index failed: {ex}")
                
                time.sleep(2)
                st.rerun()

            elif current_status == "failed":
                st.error(f"❌ Processing failed for '{active_doc['filename']}': {error_msg or 'Unknown error'}")
                if st.button("🔄 Retry Processing", type="primary", key="btn_retry_proc"):
                    try:
                        start_background_indexing(
                            user_id=user_id,
                            filename=active_doc["filename"],
                            file_path=active_doc["file_path"],
                            subject=active_doc["subject"],
                            file_size=active_doc["file_size"],
                            file_type=active_doc.get("file_type", "pdf")
                        )
                        st.success("Retry job initiated!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Retry failed: {ex}")

            elif current_status == "completed":
                cached_text = active_doc.get("extracted_text", "")
                if not cached_text and Path(active_doc["file_path"]).exists():
                    cached_text = get_full_document_text(active_doc["file_path"])

                tab_qa, tab_summary, tab_flashcards, tab_notes = st.tabs([
                    "💬 Ask Questions & Q&A",
                    "📝 Summarize Document",
                    "🗂️ Interactive Flashcards",
                    "💡 Key Notes & Concept Explanation"
                ])

                with tab_qa:
                    st.markdown(f"#### Ask Questions About *{active_doc['filename']}*")
                    st.caption("Ask questions grounded strictly in your selected notes.")

                    qa_col1, qa_col2 = st.columns([0.84, 0.16])
                    with qa_col1:
                        sm_question = st.text_input(
                            "Question",
                            placeholder=f"Ask a question about {active_doc['filename']}...",
                            key="sm_question_input",
                            label_visibility="collapsed"
                        )
                    with qa_col2:
                        sm_ask_btn = st.button("Ask Question", type="primary", width="stretch", key="sm_ask_btn")

                    if sm_ask_btn:
                        if not sm_question.strip():
                            st.warning("⚠️ Please enter a question.")
                        else:
                            with st.spinner("Searching document notes..."):
                                results = search_user_notes(
                                    question=sm_question.strip(),
                                    user_id=user_id,
                                    document_name=active_doc["filename"],
                                    subject=active_doc["subject"],
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
                                    "document": active_doc["filename"],
                                    "sources": sources_list,
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })

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

                with tab_summary:
                    st.markdown(f"#### AI Summary for *{active_doc['filename']}*")
                    sum_style = st.radio("Summary Style", ["Detailed", "Executive / Bulleted"], horizontal=True, key="sum_style_radio")

                    if st.button("Generate Summary", type="primary", key="btn_gen_sum"):
                        if not cached_text:
                            st.warning("No readable text found in document to summarize.")
                        else:
                            with st.spinner("Analyzing document and crafting comprehensive summary..."):
                                try:
                                    summary_res = generate_summary(cached_text, summary_type=sum_style)
                                    st.session_state.active_doc_summary = summary_res
                                except Exception as ex:
                                    st.error(f"Summarization error: {ex}")

                    if st.session_state.active_doc_summary:
                        st.markdown("""
                        <div class="ui-card">
                        """, unsafe_allow_html=True)
                        st.markdown(st.session_state.active_doc_summary)
                        st.markdown("</div>", unsafe_allow_html=True)

                        try:
                            sum_pdf_bytes = generate_response_pdf(
                                title=f"Summary: {active_doc['filename']}",
                                content=st.session_state.active_doc_summary,
                                student_name=user_display,
                                subject=active_doc["subject"]
                            )
                            st.download_button(
                                label="📥 Download Summary as PDF",
                                data=sum_pdf_bytes,
                                file_name=f"Summary_{Path(active_doc['filename']).stem}.pdf",
                                mime="application/pdf",
                                key="dl_summary_pdf"
                            )
                        except Exception:
                            pass

                with tab_flashcards:
                    st.markdown(f"#### Interactive Flashcards for *{active_doc['filename']}*")
                    fc_col1, fc_col2 = st.columns([0.6, 0.4])
                    with fc_col1:
                        num_cards = st.slider("Number of Flashcards", min_value=3, max_value=12, value=6, key="fc_slider")
                    with fc_col2:
                        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                        gen_fc_btn = st.button("Generate Flashcards", type="primary", width="stretch", key="btn_gen_fc")

                    if gen_fc_btn:
                        if not cached_text:
                            st.warning("No readable text found in document to generate flashcards.")
                        else:
                            with st.spinner("Synthesizing interactive flashcards..."):
                                try:
                                    fc_list = generate_flashcards(cached_text, num_cards=num_cards)
                                    st.session_state.active_doc_flashcards = fc_list
                                    st.session_state.active_card_idx = 0
                                    st.session_state.show_card_back = False
                                except Exception as ex:
                                    st.error(f"Flashcard generation error: {ex}")

                    if st.session_state.active_doc_flashcards:
                        cards = st.session_state.active_doc_flashcards
                        card_idx = st.session_state.active_card_idx % len(cards)
                        curr_card = cards[card_idx]

                        st.markdown(f"**Card {card_idx + 1} of {len(cards)}**")
                        st.markdown(f"""
                        <div class="flashcard-box">
                            <span class="flashcard-tag">{curr_card.get('tag', 'Concept')}</span>
                            <div class="flashcard-front">{curr_card.get('front', '')}</div>
                            {f'<div class="flashcard-back">{curr_card.get("back", "")}</div>' if st.session_state.show_card_back else '<div style="color:#94a3b8; font-size:0.85rem; margin-top:0.8rem;">(Click Reveal Answer below)</div>'}
                        </div>
                        """, unsafe_allow_html=True)

                        btn_c1, btn_c2, btn_c3 = st.columns([0.33, 0.34, 0.33])
                        with btn_c1:
                            if st.button("⬅️ Previous Card", width="stretch", key="btn_prev_card"):
                                st.session_state.active_card_idx = (card_idx - 1) % len(cards)
                                st.session_state.show_card_back = False
                                st.rerun()
                        with btn_c2:
                            toggle_lbl = "🙈 Hide Answer" if st.session_state.show_card_back else "👁️ Reveal Answer"
                            if st.button(toggle_lbl, type="primary", width="stretch", key="btn_flip_card"):
                                st.session_state.show_card_back = not st.session_state.show_card_back
                                st.rerun()
                        with btn_c3:
                            if st.button("Next Card ➡️", width="stretch", key="btn_next_card"):
                                st.session_state.active_card_idx = (card_idx + 1) % len(cards)
                                st.session_state.show_card_back = False
                                st.rerun()

                with tab_notes:
                    st.markdown(f"#### Structured Concept Explanation for *{active_doc['filename']}*")
                    focus_concept = st.text_input("Specific Concept to Focus On (Optional)", placeholder="e.g. Memory Hierarchy, Deadlocks...", key="notes_focus_input")

                    if st.button("Generate Detailed Concept Breakdown", type="primary", key="btn_gen_notes"):
                        if not cached_text:
                            st.warning("No readable text found in document to explain.")
                        else:
                            with st.spinner("Generating explanation and analogies..."):
                                try:
                                    notes_res = generate_notes_explanation(cached_text, focus_area=focus_concept)
                                    st.session_state.active_doc_notes = notes_res
                                except Exception as ex:
                                    st.error(f"Explanation error: {ex}")

                    if st.session_state.active_doc_notes:
                        st.markdown("""
                        <div class="ui-card">
                        """, unsafe_allow_html=True)
                        st.markdown(st.session_state.active_doc_notes)
                        st.markdown("</div>", unsafe_allow_html=True)

                        try:
                            breakdown_pdf_bytes = generate_response_pdf(
                                title=f"Explanation: {focus_concept or active_doc['filename']}",
                                content=st.session_state.active_doc_notes,
                                student_name=user_display,
                                subject=active_doc["subject"]
                            )
                            st.download_button(
                                label="📥 Download Breakdown as PDF",
                                data=breakdown_pdf_bytes,
                                file_name=derive_pdf_filename(focus_concept or active_doc['filename'], prefix="Explanation"),
                                mime="application/pdf",
                                key="dl_breakdown_pdf"
                            )
                        except Exception:
                            pass

    with tab_notes_studio:
        st.markdown("""
        <div class="ui-card">
            <h4 class="ui-card-title">📝 Multi-Notes Studio</h4>
            <p class="ui-card-subtitle">Create, organize, and edit unlimited study notes with rich Markdown formatting.</p>
        </div>
        """, unsafe_allow_html=True)

        note_top_col1, note_top_col2, note_top_col3 = st.columns([0.45, 0.35, 0.20])
        with note_top_col1:
            note_search = st.text_input("🔍 Search Notes", placeholder="Search by title or topic...", key="note_search_box")
        with note_top_col2:
            note_subj_filter = st.selectbox("Filter by Subject", ["All Subjects"] + available_subjects, format_func=lambda s: "All Subjects" if s == "All Subjects" else format_subject(s), key="note_subj_filter")
        with note_top_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("+ New Note", type="primary", width="stretch", key="btn_create_new_note"):
                new_note_id = create_note(
                    user_id=user_id,
                    title="New Study Note",
                    content="# Notes\n\n- Key concepts:\n- Important definitions:\n- Formulas/Examples:",
                    subject=note_subj_filter if note_subj_filter != "All Subjects" else "General"
                )
                st.session_state.active_note_id = new_note_id
                st.success("New note created!")
                st.rerun()

        filter_subject = None if note_subj_filter == "All Subjects" else note_subj_filter
        user_notes = get_user_notes(user_id=user_id, subject=filter_subject)
        if note_search.strip():
            user_notes = [n for n in user_notes if note_search.lower() in n["title"].lower() or note_search.lower() in n["content"].lower()]

        if st.session_state.active_note_id is not None:
            active_note = get_note_by_id(st.session_state.active_note_id, user_id=user_id)
            if active_note:
                st.markdown("---")
                st.markdown(f"### ✏️ Editing Note: *{active_note['title']}*")
                
                ed_c1, ed_c2 = st.columns([0.65, 0.35])
                with ed_c1:
                    edit_title = st.text_input("Note Title", value=active_note["title"], key=f"note_title_{active_note['id']}")
                with ed_c2:
                    curr_subj = active_note["subject"]
                    subj_idx = available_subjects.index(curr_subj) if curr_subj in available_subjects else 0
                    subj_idx = max(0, min(subj_idx, len(available_subjects) - 1)) if available_subjects else 0
                    edit_subject = st.selectbox("Subject", available_subjects, index=subj_idx, format_func=lambda s: format_subject(s), key=f"note_subj_{active_note['id']}")

                completed_docs = get_user_completed_documents(user_id)
                available_doc_names = [d["filename"] for d in completed_docs]
                curr_attached = [f for f in active_note.get("associated_files", []) if f in available_doc_names]
                edit_attached = st.multiselect("Associated Uploads / Screenshots", available_doc_names, default=curr_attached, key=f"note_files_{active_note['id']}")

                edit_content = st.text_area(
                    "Note Content (Markdown supported)",
                    value=active_note["content"],
                    height=280,
                    key=f"note_content_{active_note['id']}"
                )

                btn_s1, btn_s2, btn_s3, btn_s4 = st.columns([0.25, 0.25, 0.25, 0.25])
                with btn_s1:
                    if st.button("💾 Save Note", type="primary", width="stretch", key=f"btn_save_note_{active_note['id']}"):
                        update_note(
                            note_id=active_note["id"],
                            user_id=user_id,
                            title=edit_title,
                            content=edit_content,
                            subject=edit_subject,
                            associated_files=edit_attached
                        )
                        st.success("Note saved successfully!")
                        st.rerun()

                with btn_s2:
                    if st.button("❌ Close Editor", width="stretch", key="btn_close_note_ed"):
                        st.session_state.active_note_id = None
                        st.rerun()

                with btn_s3:
                    try:
                        note_pdf = generate_response_pdf(
                            title=edit_title,
                            content=edit_content,
                            student_name=user_display,
                            subject=edit_subject
                        )
                        st.download_button(
                            label="📥 Export as PDF",
                            data=note_pdf,
                            file_name=derive_pdf_filename(edit_title, prefix="Note"),
                            mime="application/pdf",
                            key=f"dl_note_pdf_{active_note['id']}",
                            width="stretch"
                        )
                    except Exception:
                        pass

                with btn_s4:
                    if st.button("🗑️ Delete Note", width="stretch", key=f"btn_del_note_{active_note['id']}"):
                        delete_note(active_note["id"], user_id=user_id)
                        st.session_state.active_note_id = None
                        st.success("Note deleted.")
                        st.rerun()

                st.markdown("---")

        if not user_notes:
            st.info("No notes found. Click **+ New Note** above to create your first note.")
        else:
            st.markdown(f"#### 📋 Your Notes ({len(user_notes)})")
            for note in user_notes:
                is_current = (st.session_state.active_note_id == note["id"])
                border_color = "#6366f1" if is_current else "#e2e8f0"

                st.markdown(f"""
                <div class="ui-card" style="border-color: {border_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <strong style="font-size:1.1rem; color:#0f172a;">📝 {note['title']}</strong>
                        <span class="badge-completed">{format_subject(note['subject'])}</span>
                    </div>
                    <div style="font-size:0.82rem; color:#64748b; margin-bottom:8px;">
                        Created: {note['created_at'][:19].replace('T', ' ')} &bull; Updated: {note['updated_at'][:19].replace('T', ' ')}
                        {f" &bull; 📎 {len(note['associated_files'])} file(s)" if note.get('associated_files') else ""}
                    </div>
                """, unsafe_allow_html=True)

                with st.expander(f"Preview: {note['title']}"):
                    st.markdown(note["content"])
                    if note.get("associated_files"):
                        st.caption(f"Attached files: {', '.join(note['associated_files'])}")

                nc_1, nc_2, nc_3 = st.columns([0.33, 0.33, 0.34])
                with nc_1:
                    if st.button("✏️ Edit Note", key=f"open_note_{note['id']}", width="stretch"):
                        st.session_state.active_note_id = note["id"]
                        st.rerun()
                with nc_2:
                    try:
                        n_pdf = generate_response_pdf(
                            title=note["title"],
                            content=note["content"],
                            student_name=user_display,
                            subject=note["subject"]
                        )
                        st.download_button(
                            label="📥 Download PDF",
                            data=n_pdf,
                            file_name=derive_pdf_filename(note["title"], prefix="Note"),
                            mime="application/pdf",
                            key=f"dl_card_pdf_{note['id']}",
                            width="stretch"
                        )
                    except Exception:
                        pass
                with nc_3:
                    if st.button("🗑️ Delete", key=f"card_del_note_{note['id']}", width="stretch"):
                        delete_note(note["id"], user_id=user_id)
                        if st.session_state.active_note_id == note["id"]:
                            st.session_state.active_note_id = None
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# QUIZ & PRACTICE
# ==================================================

elif page == "Quiz & Practice":
    st.subheader("Adaptive Background Quiz & Practice Hub")
    st.caption("Generate targeted quizzes that process independently in the background.")

    completed_docs = get_user_completed_documents(user_id)
    user_quiz_jobs = get_user_quiz_jobs(user_id=user_id)

    active_qjobs = [q for q in user_quiz_jobs if q["status"] in ("pending", "processing")]
    latest_qjob = user_quiz_jobs[0] if user_quiz_jobs else None

    if active_qjobs:
        curr_active_qjob = active_qjobs[0]
        st.markdown(f"""
        <div class="ui-card" style="border-left: 4px solid #f59e0b; background: #fffbeb;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h4 style="color:#b45309; margin:0 0 4px 0;">⏳ Quiz Generation in Progress...</h4>
                    <p style="color:#92400e; margin:0; font-size:0.9rem;">
                        Generating <strong>{curr_active_qjob['topic']}</strong> ({curr_active_qjob['difficulty']} Level &bull; {curr_active_qjob['number_of_questions']} questions).
                    </p>
                </div>
                <div>
                    <span class="badge-processing">PROCESSING ({curr_active_qjob['progress']}%)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(curr_active_qjob["progress"] / 100.0)

        q_ref1, q_ref2, q_ref3 = st.columns([0.6, 0.2, 0.2])
        with q_ref2:
            if st.button("🔄 Refresh Status", width="stretch", key="btn_ref_q_job"):
                st.rerun()
        with q_ref3:
            if st.button("❌ Dismiss Job", width="stretch", key=f"btn_cancel_q_job_{curr_active_qjob['id']}"):
                delete_quiz_job(curr_active_qjob["id"], user_id)
                st.rerun()

        time.sleep(2)
        st.rerun()

    elif latest_qjob and latest_qjob["status"] == "completed" and (st.session_state.quiz is None or st.session_state.quiz_submitted):
        st.markdown(f"""
        <div class="ui-card" style="border-left: 4px solid #10b981; background: #f0fdf4;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h4 style="color:#065f46; margin:0 0 4px 0;">✅ Quiz Generated Successfully!</h4>
                    <p style="color:#047857; margin:0; font-size:0.9rem;">
                        <strong>{latest_qjob['topic']}</strong> ({latest_qjob['difficulty']} Level &bull; {latest_qjob['number_of_questions']} questions &bull; Created: {latest_qjob['created_at'][:19].replace('T', ' ')})
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if latest_qjob.get("quiz_data") and latest_qjob["quiz_data"].get("questions"):
            if st.button("🎯 Start Practicing Generated Quiz", type="primary", width="stretch", key=f"btn_start_latest_q_{latest_qjob['id']}"):
                st.session_state.quiz = latest_qjob["quiz_data"]["questions"]
                st.session_state.quiz_subject = latest_qjob["subject"]
                st.session_state.quiz_topic = latest_qjob["topic"]
                st.session_state.quiz_difficulty = latest_qjob["difficulty"]
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score_details = None
                st.rerun()

    elif latest_qjob and latest_qjob["status"] == "failed":
        st.error(f"❌ Previous quiz generation failed: {latest_qjob.get('error_message', 'Unknown error')}")

    st.markdown("---")

    st.markdown("### 🚀 Create New Practice Quiz")
    quiz_mode = st.radio(
        "Quiz Source Mode",
        [
            "📄 Generate Quiz from Uploaded Material (PDF / Image / DOCX / Notes)",
            "✍️ Generate Quiz from Custom Topic"
        ],
        horizontal=True,
        key="quiz_source_mode_radio"
    )

    if "Uploaded Material" in quiz_mode:
        if not completed_docs:
            st.info("💡 You have no completed uploaded documents yet. Upload notes in 'Study Material' or switch to 'Custom Topic' mode.")
        else:
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                doc_map = {f"📄 {d['filename']} ({format_subject(d['subject'])})": d for d in completed_docs}
                doc_choices = list(doc_map.keys())

                def_doc_idx = 0
                matching_q_indices = [i for i, d in enumerate(completed_docs) if d["id"] == st.session_state.active_document_id]
                if matching_q_indices:
                    def_doc_idx = matching_q_indices[0]
                
                def_doc_idx = max(0, min(def_doc_idx, len(doc_choices) - 1))
                
                if "quiz_doc_select" in st.session_state and st.session_state.quiz_doc_select not in doc_choices:
                    del st.session_state["quiz_doc_select"]

                chosen_label = st.selectbox(
                    "Select Source Document / Notes",
                    doc_choices,
                    index=def_doc_idx,
                    key="quiz_doc_select"
                )
                chosen_doc = doc_map[chosen_label]

            with q_col2:
                target_level = st.selectbox("Target Level", ["Beginner", "Intermediate", "Advanced"], index=1, key="quiz_mat_target_level")
                number = st.slider("Number of Questions", min_value=5, max_value=15, value=5, key="quiz_mat_num_slider")

            if st.button("Start Quiz Generation", type="primary", width="stretch", key="btn_start_bg_quiz_mat"):
                doc_text = chosen_doc.get("extracted_text", "")
                if not doc_text and Path(chosen_doc["file_path"]).exists():
                    doc_text = get_full_document_text(chosen_doc["file_path"])

                if not doc_text.strip():
                    st.warning(f"⚠️ No readable text found in '{chosen_doc['filename']}' to generate questions.")
                else:
                    job_id = start_background_quiz_generation(
                        user_id=user_id,
                        subject=chosen_doc["subject"],
                        topic=f"Material: {chosen_doc['filename']}",
                        difficulty=target_level,
                        number_of_questions=number,
                        source_type="material",
                        source_id=chosen_doc["id"],
                        source_name=chosen_doc["filename"],
                        content=doc_text
                    )
                    st.session_state.active_quiz_job_id = job_id
                    st.session_state.quiz = None
                    st.session_state.quiz_submitted = False
                    st.success(f"Quiz job #{job_id} started in background!")
                    st.rerun()

    else:
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            subject = st.selectbox("Subject", available_subjects, format_func=lambda s: format_subject(s), key="quiz_sub_select")
            topic = st.text_input("Topic", placeholder="e.g. Normalization, Process Synchronization...", key="quiz_top_input")

        with q_col2:
            target_level = st.selectbox("Target Level", ["Beginner", "Intermediate", "Advanced"], index=1, key="quiz_target_level")
            number = st.slider("Number of Questions", min_value=5, max_value=15, value=5, key="quiz_num_slider")

        if st.button("🚀 Start Background Quiz Generation", type="primary", width="stretch", key="btn_start_bg_quiz_top"):
            if not topic.strip():
                st.warning("⚠️ Please enter a study topic.")
            else:
                job_id = start_background_quiz_generation(
                    user_id=user_id,
                    subject=subject,
                    topic=topic.strip(),
                    difficulty=target_level,
                    number_of_questions=number,
                    source_type="topic",
                    source_name=topic.strip()
                )
                st.session_state.active_quiz_job_id = job_id
                st.session_state.quiz = None
                st.session_state.quiz_submitted = False
                st.success(f"Quiz job #{job_id} for '{topic.strip()}' started in background!")
                st.rerun()

    # RENDER ACTIVE QUIZ INSIDE ST.FORM
    if st.session_state.quiz:
        st.markdown("---")
        quiz = st.session_state.quiz

        st.markdown(f"""
        <div class="ui-card">
            <h4 class="ui-card-title">📝 Active Practice Quiz: {st.session_state.quiz_topic}</h4>
            <p class="ui-card-subtitle">Difficulty: <strong>{st.session_state.quiz_difficulty}</strong> &bull; Total Questions: {len(quiz)}</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.quiz_submitted:
            with st.form(key="active_quiz_form"):
                form_answers = {}
                for i, question in enumerate(quiz):
                    st.markdown(f"**Question {i + 1}:** {question['question']}")

                    current_choice = st.session_state.quiz_answers.get(i)
                    safe_index = current_choice if (current_choice is not None and 0 <= current_choice < len(question["options"])) else None

                    selected = st.radio(
                        f"Select answer for Question {i + 1}",
                        options=question["options"],
                        index=safe_index,
                        key=f"form_mcq_{i}",
                        label_visibility="collapsed"
                    )
                    if selected in question["options"]:
                        form_answers[i] = question["options"].index(selected)
                    st.markdown("<br>", unsafe_allow_html=True)

                submit_btn = st.form_submit_button("Submit Quiz", type="primary", width="stretch")

                if submit_btn:
                    st.session_state.quiz_answers = form_answers
                    is_complete, missing_questions = validate_all_answered(quiz, form_answers)

                    if not is_complete:
                        missing_str = ", ".join([f"Question {q}" for q in missing_questions])
                        st.warning(f"⚠️ Please answer all questions before submitting. Unanswered: **{missing_str}**")
                    else:
                        correct, total, score, details = calculate_score(quiz, form_answers)

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

        else:
            if st.session_state.quiz_score_details:
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

    if user_quiz_jobs:
        st.markdown("---")
        with st.expander("📚 Saved Generated Quizzes & Generation History"):
            for qj in user_quiz_jobs:
                q_status = qj["status"]
                badge = '<span class="badge-completed">COMPLETED</span>' if q_status == "completed" else f'<span class="badge-processing">{q_status.upper()}</span>'
                
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <strong>{qj['topic']}</strong> ({qj['difficulty']} &bull; {qj['number_of_questions']} questions) &nbsp; {badge}
                    <span style="font-size:0.8rem; color:#64748b;">{qj['created_at'][:19].replace('T', ' ')}</span>
                </div>
                """, unsafe_allow_html=True)

                if q_status == "completed" and qj.get("quiz_data") and qj["quiz_data"].get("questions"):
                    col_q1, col_q2 = st.columns([0.3, 0.7])
                    with col_q1:
                        if st.button("🎯 Practice This Quiz", key=f"retake_q_{qj['id']}"):
                            st.session_state.quiz = qj["quiz_data"]["questions"]
                            st.session_state.quiz_subject = qj["subject"]
                            st.session_state.quiz_topic = qj["topic"]
                            st.session_state.quiz_difficulty = qj["difficulty"]
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_score_details = None
                            st.rerun()
                st.markdown("---")


# ==================================================
# PERFORMANCE & ANALYTICS
# ==================================================

elif page == "Performance & Analytics":
    st.subheader("Performance & Analytics")
    st.caption("Track learning progress, mastery metrics, and predictive insights.")

    rows = get_topic_performance(user_id=user_id)
    df = load_performance_data(rows)
    df = analyze_performance(df)

    if df.empty:
        st.info("No quiz attempts recorded yet. Complete quizzes to generate performance analytics.")
    else:
        st.dataframe(df, width="stretch")

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
# RECOMMENDATIONS & STUDY PLANNER
# ==================================================

elif page == "Recommendations & Study Plan":
    st.subheader("Personalized Study Planner & Recommendations")
    st.caption("Design custom study schedules with exact day allocations and PDF export.")

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
        <p class="ui-card-subtitle">Set your target topic, proficiency level, available study days, and exact duration.</p>
    </div>
    """, unsafe_allow_html=True)

    p_col1, p_col2 = st.columns([0.65, 0.35])
    with p_col1:
        planner_topic = st.text_input(
            "Study Topic",
            placeholder="e.g. Transaction, Operating Systems Memory Management...",
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
        st.markdown("##### Daily Study Duration")
        minute_options = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

        for day in selected_days:
            dur_col1, dur_col2, dur_col3, dur_col4 = st.columns([0.25, 0.25, 0.25, 0.25])
            with dur_col1:
                st.markdown(f"<div style='padding-top:8px;'><b>{day}</b></div>", unsafe_allow_html=True)
            with dur_col2:
                d_hours = st.number_input(f"Hours ({day})", min_value=0, max_value=12, value=2 if day != "Saturday" else 3, key=f"hrs_{day}", label_visibility="collapsed")
            with dur_col3:
                def_min_idx = minute_options.index(30) if day == "Monday" and 30 in minute_options else 0
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
                    width="stretch"
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
    current_name = profile[0] if (profile and profile[0]) else user_display
    current_course = profile[1] if (profile and profile[1]) else user_course
    current_semester = profile[2] if (profile and profile[2]) else user_semester

    with st.container():
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        name = st.text_input("Full Name", current_name, placeholder="e.g. Alex Johnson")
        course = st.text_input("Degree / Course", current_course, placeholder="e.g. Computer Science & Engineering")
        semester = st.number_input("Current Semester", min_value=1, max_value=12, value=int(current_semester))

        if st.button("Save Profile", type="primary"):
            if name.strip():
                save_profile(name.strip(), course.strip(), int(semester), user_id=user_id)
                if st.session_state.user:
                    st.session_state.user["name"] = name.strip()
                    st.session_state.user["course"] = course.strip()
                    st.session_state.user["semester"] = int(semester)
                st.success("Profile saved successfully.")
                st.rerun()
            else:
                st.warning("Please enter your name.")
        st.markdown('</div>', unsafe_allow_html=True)