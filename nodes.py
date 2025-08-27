import os
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_model_type(state):
    """Prompt user to select model type (classification or regression)."""

    modeltype = st.text_input(
        "Enter the type of model you are working on (press Enter):",
        placeholder="C for Classification | R for Regression",
        help="Type 'C' or 'Classification' for classification tasks. "
             "Type 'R' or 'Regression' for regression tasks."
    )

    # Validate input
    valid_inputs = {"c": "classification", "classification": "classification",
                    "r": "regression", "regression": "regression"}
    
    if not modeltype:
        st.stop()  # wait until user enters something
    
    if modeltype.lower() not in valid_inputs:
        st.info("⚠️ Please enter a valid option: C for Classification or R for Regression.")
        st.stop()

    # Update state and return
    state["model_type"] = valid_inputs[modeltype.lower()]
    return state

def llm_node_classification(state):
    """Processes classification metrics and returns suggestions."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    # Create prompt
    messages = ChatPromptTemplate.from_messages([
        ("system", dedent("""\
            You are a seasoned data scientist specialized in classification models. 
            You will get the user's classification metrics and provide actionable suggestions 
            to improve the model in bullet points.
            Be concise and avoid unnecessary details.
            If the question is not about classification, say 'Please input classification model metrics.'.
        """)),
        MessagesPlaceholder(variable_name="messages"),
        ("user", state["metrics_to_tune"])
    ])
    
    # Run chain
    chain = messages | llm
    response = chain.invoke(state)
    state["final_answer"] = [response]
    return state

def llm_node_regression(state):
    """Processes regression metrics and returns suggestions."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    # Create prompt
    messages = ChatPromptTemplate.from_messages([
        ("system", dedent("""\
            You are a seasoned data scientist specialized in regression models. 
            You will get the user's regression metrics and provide actionable suggestions 
            to improve the model in bullet points.
            Be concise and avoid unnecessary details.
            If the question is not about regression, say 'Please input regression model metrics.'.
        """)),
        MessagesPlaceholder(variable_name="messages"),
        ("user", state["metrics_to_tune"])
    ])
    
    # Run chain
    chain = messages | llm
    response = chain.invoke(state)
    state["final_answer"] = [response]
    return state
