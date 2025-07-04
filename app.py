import streamlit as st
import google.generativeai as genai
from PIL import Image
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# -------------------- 페이지 설정 --------------------
st.set_page_config(
    page_title="AI 도우미",
    page_icon="🚀",
    layout="wide"
)

# -------------------- 함수 정의 --------------------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# -------------------- 사이드바 (설정) --------------------
with st.sidebar:
    st.header("1. 기본 설정")
    
    # [수정된 부분] Streamlit Cloud의 Secrets에서 API 키를 가져옵니다.
    # 이 부분은 배포 시에만 정상 작동하며, 로컬 테스트 시에는 에러가 발생합니다.
    if 'api_key' not in st.session_state:
        try:
            st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=st.session_state.api_key)
            st.success("API 키가 준비되었습니다!")
        except (KeyError, Exception):
            # 로컬 테스트를 위해 경고 메시지를 표시하거나, 직접 키를 입력하는 로직을 임시로 추가할 수 있습니다.
            st.warning("배포 환경이 아니거나 Secrets에 API 키가 없습니다.")

    st.divider()
    st.header("2. 챗봇 페르소나 설정")
    system_prompt = st.text_area("AI 비서의 역할을 지정해주세요.", "당신은 유능한 AI 비서입니다.")

    st.divider()
    st.header("3. 파일 학습 (RAG)")
    knowledge_file = st.file_uploader("지식 베이스로 사용할 PDF 파일을 업로드하세요.", type="pdf")

    if st.button("파일 학습 시작"):
        # API 키가 세션 상태에 있는지 확인
        if 'api_key' in st.session_state and st.session_state.api_key:
            if knowledge_file:
                with st.spinner("파일을 읽고 학습하는 중..."):
                    try:
                        raw_text = extract_text_from_pdf(knowledge_file)
                        text_chunks = split_text_into_chunks(raw_text)
                        
                        embedding_model = 'models/text-embedding-004'
                        result = genai.embed_content(model=embedding_model, content=text_chunks, task_type="RETRIEVAL_DOCUMENT")
                        st.session_state.rag_embeddings = np.array(result['embedding'])
                        st.session_state.rag_chunks = text_chunks
                        st.success(f"{len(text_chunks)}개의 텍스트 조각 학습 완료!")
                    except Exception as e:
                        st.error(f"파일 학습 중 오류 발생: {e}")
            else:
                st.warning("학습할 PDF 파일을 먼저 업로드해주세요.")
        else:
            st.warning("API 키가 준비되지 않았습니다.")

# -------------------- 메인 화면 --------------------
st.title("🚀 AI 도우미")
st.caption("음성으로 질문하거나, 학습된 파일에 대해 질문해보세요.")

# 모델 및 대화 기록 초기화
try:
    if 'api_key' in st.session_state and st.session_state.api_key:
        if "model" not in st.session_state or st.session_state.system_prompt != system_prompt:
            st.session_state.system_prompt = system_prompt
            st.session_state.model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_prompt)
            st.session_state.messages = []
except Exception as e:
    st.error(f"모델 초기화 오류: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 입력 처리 (음성, 텍스트) ---
prompt = st.chat_input("무엇이든 물어보세요!")
audio_prompt = None

st.write("또는, 음성으로 질문하세요:")
audio_data = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key='recorder')
if audio_data:
    if 'api_key' in st.session_state and st.session_state.api_key:
        with st.spinner("음성을 텍스트로 변환 중..."):
            try:
                audio_path = "temp_audio.wav"
                with open(audio_path, "wb") as f:
                    f.write(audio_data['bytes'])
                
                audio_file = genai.upload_file(path=audio_path)
                response = genai.GenerativeModel('gemini-2.5-flash').generate_content(
                    ["이 오디오 파일의 내용을 텍스트로 적어줘.", audio_file]
                )
                audio_prompt = response.text
                st.info(f"음성 인식 결과: {audio_prompt}")
            except Exception as e:
                st.error(f"음성 변환 오류: {e}")
    else:
        st.warning("API 키가 설정되지 않아 음성 인식을 사용할 수 없습니다.")

final_prompt = prompt or audio_prompt

if final_prompt:
    if 'model' not in st.session_state:
        st.error("API 키를 먼저 설정하고 모델을 초기화해주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("AI가 생각 중입니다..."):
            try:
                augmented_prompt = final_prompt
                if "rag_embeddings" in st.session_state:
                    embedding_model = 'models/text-embedding-004'
                    query_embedding = genai.embed_content(model=embedding_model, content=final_prompt, task_type="RETRIEVAL_QUERY")['embedding']
                    
                    similarities = cosine_similarity(np.array(query_embedding).reshape(1, -1), st.session_state.rag_embeddings).flatten()
                    top_indices = similarities.argsort()[-3:][::-1]
                    
                    context = "\n---\n".join([st.session_state.rag_chunks[i] for i in top_indices])
                    augmented_prompt = f"아래 컨텍스트를 참고해서 질문에 답해줘.\n\n[컨텍스트]\n{context}\n\n[질문]\n{final_prompt}"
                    st.info("ℹ️ 학습된 파일 내용을 참고하여 답변합니다.")

                response = st.session_state.model.generate_content(augmented_prompt)
                assistant_response = response.text
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"응답 생성 오류: {e}")
