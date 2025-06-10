import streamlit as st
import PyPDF2
import time
from llama_cpp import Llama

st.set_page_config(page_title="Resume + Career Advisor Bot", layout="wide")
st.title("🤖 Resume + Career Advisor Bot (GPU-Powered LLaMA)")
st.markdown("Upload your resume and chat about your career — now blazing fast with CUDA.")

# ---------------- Session state setup ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "user_input_reset" not in st.session_state:
    st.session_state.user_input_reset = False

# ---------------- Callback to reset input ----------------
def reset_input():
    st.session_state["user_input_reset"] = True

# ---------------- LLaMA model load ----------------
llm = Llama(
    model_path="llama.cpp/models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=40,
    n_threads=8,
    use_mlock=False,
    n_batch=512
)

# ---------------- Resume Upload ----------------
pdf_file = st.file_uploader("📄 Upload your resume (PDF only)", type=["pdf"])

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join([page.extract_text() for page in reader.pages])

if pdf_file and not st.session_state.resume_text:
    st.session_state.resume_text = extract_text_from_pdf(pdf_file)
    st.success("✅ Resume uploaded and processed!")

# ---------------- Input reset if flagged ----------------
if st.session_state.user_input_reset:
    st.session_state.user_input = ""
    st.session_state.user_input_reset = False

# ---------------- Chat bubble CSS styling (ChatGPT style + paragraph fix) ----------------
chat_styles = """
<style>
.user-msg {
    background-color: #444654;
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 6px;
    max-width: 75%;
    align-self: flex-end;
    text-align: right;
    margin-left: auto;
    font-family: monospace;
    white-space: pre-wrap;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}
.ai-msg {
    background-color: #343541;
    color: #e0e0e0;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 6px;
    max-width: 75%;
    align-self: flex-start;
    text-align: left;
    margin-right: auto;
    font-family: monospace;
    white-space: pre-wrap;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}
.chat-container {
    display: flex;
    flex-direction: column;
}
</style>
"""

st.markdown(chat_styles, unsafe_allow_html=True)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# ---------------- Chat history display ----------------
for role, msg in st.session_state.chat_history:
    css_class = "user-msg" if role == "User" else "ai-msg"
    st.markdown(f'<div class="{css_class}">{msg}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- User Input Box ----------------
user_input = st.text_input("You:", key="user_input")

# ---------------- Handle User Message ----------------
if user_input and st.session_state.resume_text:
    st.session_state.chat_history.append(("User", user_input))

    chat_context = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])

    full_prompt = f"""
You are a friendly, casual, and knowledgeable AI career advisor. DO not use bulletins or numbers as points. Speak like a helpful buddy who knows resumes, tech, and job hunting really well. Keep it real, clear, and useful — no stiff corporate language unless the user asks for it. Maybe try to chat like chatgpt from openai.
RESUME:
{st.session_state.resume_text}

CONVERSATION:
{chat_context}
AI:
"""

    with st.spinner("Thinking..."):
        full_response = ""
        response_placeholder = st.empty()
        token_counter = 0
        max_tokens = 512  # safety cap

        response_stream = llm.create_completion(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
            stop=["</s>", "\n\n", "End of response."]
        )

        for chunk in response_stream:
            token_counter += 1
            full_response += chunk["choices"][0]["text"]
            if token_counter >= max_tokens:
                break
            response_placeholder.markdown(full_response.strip() + "▌")

        time.sleep(0.1)
        response_placeholder.empty()
        response_placeholder.markdown(full_response.strip())

    st.session_state.chat_history.append(("AI", full_response.strip()))
    reset_input()
    st.rerun()

# ---------------- Clear Chat Button ----------------
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.resume_text = ""
    st.rerun()
