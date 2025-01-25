import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
from dotenv import load_dotenv
from langchain import callbacks
from langchain_core.ducuments import Document
from langchain_openai import ChatOpenAI
from pypdf import PdfReader


model = "gpt-4o-mini"
pdf_file_urls = [...]

def load_pdf(pdf_url: str) -> Document:
    response = requests.get([pdf_url])
    response.raise_for_status()

    with BytesIO(response.content) as file:
        reader = PdfReader(file)
        pdf_title = re.split(
            r"契約|契約書",
            re.sub(
                r"\s*1\s*",
                "",
                reader.pages[0].extract_text().replace("\n", "").replace(" ", ""),
            ),
            1,
        )[0]
        pdf_text = "".join(
            page.extract_text().replace("\n", "") for page in reader.pages
        )
        return Document(
            page_content=pdf_text,
            metadata={"title": pdf_title, "source": pdf_url},
        )

def retrieve_documents(question: str) -> list[Document]:
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_pdf, pdf_file_urls))
    titles = [doc.metadata["title"] for doc in results]

    openai = ChatOpenAI(model=model, temperature=0.0)
    messages = [
        (
            "system",
            (
                "あなたは法律の専門家です。質問に関連する契約書のタイトルを候補の中から全て選び、リストで出力してください。\n"
                "出力は、リストのタイトルを','で区切ったものを出力してください。\n"
                "契約書1,契約書2,契約書3"
            ),
        ),
        (
            "user",
            (
                "【質問】{question}\n"
                "【契約書のタイトル候補】\n"
                f"{','.join(titles)}"
            ),
        ),
    ]
    related_doc_titles = openai.invoke(messages).content
    related_doc_titles = related_doc_titles.split(",")
    return [doc for doc in results if doc.metadata["title"] in related_doc_titles]

# 回答を生成する関数
def generate_answer(question: str, context_text: str) -> str:
    openai = ChatOpenAI(model=model, temperature=0.0)
    prompt = r"【コンテキスト】\n{context_text}\n\n【質問】{question}"
    messages = [
        (
            "system",
            (
                "あなたは法律の専門家です。"
                "ユーザーの質問には、可能な限り具体的かつ簡潔で、必要な情報を網羅した回答を提供してください。"
                "端的に答えられる質問には端的に答えてください。"
                "ユーザーへの質問が必要な質問には、まずコンテキストにかかれている事実を説明し、必要に応じて..."
                "また、与えられたコンテキストのみでは回答の正確性に不安がある場合は、その旨を提示してください。"
                "コンテキスト外の情報を推測や想像で補わないでください。法律的な助言を提供する際は、正確で最新の..."
                "あなたの回答は、correctness, helpfulness, conciseness, harmlessnessで評価される..."
                "回答の例を以下に示します。\n"
                "-----\n"
                "【質問】ソフトウェア開発業務委託契約について、委託料の金額はいくら？\n"
                "【回答】委託料の金額は金五百万円（税別）です。"
                "-----\n"
                "-----\n"
            )
        ),
        ("user", prompt),
    ]
    response = openai.invoke(messages)
    return response.content


def rag_impletention(question: str) -> str:
    docs = load_pdf(pdf_file_urls)
    retrieve_docs = retrieve_documents(question)
    ###?###
    answer = generate_answer(question, retrieve_docs)