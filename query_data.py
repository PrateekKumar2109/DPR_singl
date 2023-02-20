from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.llms import Cohere

from langchain.chains import ChatVectorDBChain
import cohere
import time
import altair as alt
import numpy as np
import pandas as pd
api_key = 'vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg'

co = cohere.Client(api_key)

cohere = Cohere(model="command-xlarge-nightly", cohere_api_key="api_key")
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI DPR assistant for answering questions about the DPR related drilling data.
You are given the following extracted parts of a long document and a question. Provide a conversational answer from the data:

If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = Cohere(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    
    )
    return qa_chain
