#  Resume + Career Advisor Bot (GPU-Powered LLaMA with CUDA)

This project is a **local, offline chatbot** that acts like a friendly AI career advisor. Upload your resume and chat in real-time — all powered by a quantized [LLaMA](https://github.com/ggerganov/llama.cpp) model running on your GPU.

⚙️ Built using:
- 🦙 [`llama.cpp`](https://github.com/ggerganov/llama.cpp) with the [`CapybaraHermes-2.5-Mistral-7B-GGUF`](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF) model (`capybarahermes-2.5-mistral-7b.Q4_K_M.gguf`)
-  Streamlit for UI
-  CUDA for blazing-fast local inference
-  PyPDF2 for resume parsing

---

##  Features
-  Real-time AI chat about your career path
-  Upload and process your own resume (PDF)
-  Responses are friendly, helpful, and resume-aware
-  100% local — no APIs, no internet needed

---

##  Demo
> Since this app runs entirely offline with CUDA, it's not deployed on Streamlit Cloud.  
> Here's a short screen recording of the full experience:  
> [Video](https://www.linkedin.com/posts/mkashok_ai-llms-llama-activity-7338256337706864641-bLAL?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADWbMxwBo8KzXbeny91WV269EG6u-6CR0WQ)

---

##  Setup Instructions

> Prerequisites:
- GPU with CUDA support
- Python 3.8+
- [`llama.cpp`](https://github.com/ggerganov/llama.cpp) compiled with CUDA backend
- Download a `.gguf` model (e.g., `capybarahermes-2.5-mistral-7b.Q4_K_M.gguf`)

### 1. Clone this repo

```bash
git clone https://github.com/MK-github03/resume-career-advisor-bot-gpu-local-llama.git
cd resume-career-advisor-bot-gpu-local-llama
