import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# Streamlit 제목 설정
st.title("Document Summarization")

# 파일 업로드 위젯
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # 파일 내용 읽기
    pages = uploaded_file.read()

    # Map 단계에서 처리할 프롬프트 정의
    map_template = """다음은 문서 중 일부 내용입니다
    {pages}
    이 문서 목록을 기반으로 주요 내용을 요약해 주세요.
    답변:"""

    # Map 프롬프트 완성
    map_prompt = PromptTemplate.from_template(map_template)

    # Map에서 수행할 LLMChain 정의
    llm = ChatOpenAI(temperature=0, 
                     model_name='gpt-3.5-turbo-16k',
                     max_tokens=2000)

    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # 문서 요약
    summary = map_chain.run(pages=pages)

    # 요약 결과 출력
    st.subheader("Summary")
    st.write(summary)

else:
    st.write("Please upload a PDF file.")

