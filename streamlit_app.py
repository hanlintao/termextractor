import streamlit as st
import pandas as pd
from io import BytesIO
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os

class Term(BaseModel):
    """Information about a term."""
    term_cn: Optional[str] = Field(default=None, description="The Chinese term")
    term_en: Optional[str] = Field(default=None, description="must provide an english translation of the Chinese term")
    term_type: Optional[str] = Field(default=None, description="only choose from the following:【专业术语】、【通用术语】、【活动名称】、【人名】、【地名】、【机构名】、【其他】")
    term_explanation: Optional[str] = Field(default=None, description="The explanation or definition of the term")

class Data(BaseModel):
    """Extracted data about terms."""
    terms: List[Term]

def extract_terms(api_key, source_text, target_text, source_lang, target_lang):
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(model="gpt-4o")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是一个资深的术语提取专家，提取术语时term_type严格遵循以下术语分类标准:【专业术语】、【通用术语】、【活动名称】、【人名】、【地名】、【机构名】、【其他】。请从给定的文本中提取术语信息，所有术语务必提供英文译文，并以以下JSON格式返回结果。如果没有抽取到术语则也输出一个空值的JSON输出:"),
        ("user", f"```json\n{{{{\"terms\": [\n {{{{ \"term_cn\": \"\",\n \"term_en\": \"\",\n \"term_type\": \"\",\n \"term_explanation\": \"\"\n }}}}\n]}}}}\n```\n请从以下文本中提取术语信息。源语言：{source_lang}, 目标语言：{target_lang}\n{source_text}\n{target_text}")
    ])

    output_parser = PydanticOutputParser(pydantic_object=Data)
    chain = prompt | llm | output_parser

    extracted_data = chain.invoke({"input": f"{source_text}\n{target_text}"})

    terms_data = []
    for term in extracted_data.terms:
        terms_data.append({
            'term_cn': term.term_cn,
            'term_en': term.term_en,
            'term_type': term.term_type,
            'term_explanation': term.term_explanation,
            'term_source': source_text,
            'term_target': target_text
        })

    return terms_data

st.title('术语提取工具')

api_key = st.text_input('请输入OpenAI API密钥', type='password')
source_text = st.text_area('请输入源语言文本')
target_text = st.text_area('请输入目标语言文本（可选）')

source_lang = st.selectbox('选择源语言', ['中文', '英文', '其他'])
target_lang = st.selectbox('选择目标语言', ['英文', '中文', '其他'])

if st.button('提取术语'):
    if not api_key or not source_text:
        st.error('请填写所有字段')
    elif not target_text and (not source_lang or not target_lang):
        st.error('如果没有目标语言文本，请选择源语言和目标语言')
    else:
        terms_data = extract_terms(api_key, source_text, target_text, source_lang, target_lang)

        if terms_data:
            df = pd.DataFrame(terms_data)
            st.write(df)

            # 将DataFrame导出为Excel文件
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Terms')
            output.seek(0)

            st.download_button(
                label="下载术语表格",
                data=output,
                file_name='extracted_terms.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.write('没有提取到术语')
