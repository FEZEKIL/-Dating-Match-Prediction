
# Speed Dating Match Prediction Agent with LangGraph

## Project Overview
This project aims to predict romantic chemistry in speed dating scenarios using machine learning and an AI agent built with LangGraph. By analyzing a rich dataset of speed dating interactions, including ratings on various attributes, the agent will learn to predict the likelihood of a "match" between participants.

## Problem Statement
Speed dating provides a unique dataset for exploring human interaction and attraction. Can machine learning algorithms, orchestrated by an intelligent agent, effectively predict match outcomes from limited initial impressions? This project seeks to answer that question by building an autonomous data science pipeline.

## Key Technologies/Libraries
*   **LangGraph**: Core framework for building AI workflows and stateful multi-agent applications.
*   **LangChain**: Base library for managing prompts, agents, and tool integrations.
*   **LangChain-OpenAI**: Provides access to OpenAI's GPT-based models for intelligent reasoning.
*   **Pandas**: Essential for data manipulation and analysis.
*   **Scikit-learn**: Used for preprocessing, feature selection (e.g., `LabelEncoder`, `SimpleImputer`, `RFE`, `RandomForestClassifier`), and model evaluation.
*   **XGBoost**: A powerful gradient boosting framework used for the match prediction classification model.
*   **Numpy**, **SciPy**, **Tqdm**, **Joblib**, **Requests**, **PyYAML**: Supporting libraries for numerical operations, progress bars, serialization, HTTP requests, and configuration.

## Setup and Installation
To run this project, follow these steps:

1.  **Install Dependencies**: Execute the following `pip install` commands (already present in the notebook):
    ```bash
    %pip install langchain==1.2.10
    %pip install langchain-openai==1.1.9
    %pip install langgraph==1.0.8
    %pip install openai==2.20.0
    %pip install numpy==2.4.2
    %pip install pandas==3.0.0
    %pip install scikit-learn==1.8.0
    %pip install scipy==1.17.0
    %pip install xgboost==3.2.0
    %pip install tqdm==4.67.3
    %pip install joblib==1.5.3
    %pip install requests==2.32.5
    %pip install PyYAML==6.0.3
    ```

2.  **Obtain OpenAI API Key**: You will need an API key from OpenAI. Set it as an environment variable named `OPENAI_API_KEY` or use Colab's secrets manager. The notebook code expects this key to be accessible via `os.environ.get('OPENAI_API_KEY')`.

3.  **Download Dataset**: The `speeddating.csv` dataset is fetched automatically by the provided `!wget` command in the notebook:
    ```bash
    !wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/sqiJ_CW9x_k2T6C2KgPf6Q/speeddating.csv
    ```

## Agent Workflow and Components

This project uses LangGraph to build a sequential agent that performs data cleaning, feature selection, and model training. The core components are:

*   **Agent State (`AgentState`)**: Defines the state of our agent, primarily holding a list of `messages` to track the conversation and reasoning process.
    ```python
    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], add]
    ```

*   **Tools**: Three custom tools are defined for the agent to interact with the data:
    *   `clean_speed_dating_data(file_name: str)`: Reads the raw CSV, drops leakage columns (`decision`, `decision_o`), handles byte-string values, and imputes missing values using `SimpleImputer`. Saves the cleaned data to `cleaned_data.csv`.
    *   `select_top_features(n_features: int)`: Reads `cleaned_data.csv`, applies `Recursive Feature Elimination (RFE)` with a `RandomForestClassifier` to identify the most impactful features for predicting a match. Returns a list of selected feature names.
    *   `train_probability_model(features: List[str])`: Reads `cleaned_data.csv`, trains an `XGBoostClassifier` on the specified features, and evaluates its performance using the `ROC AUC score`. Returns the score to indicate the model's predictive quality.

*   **Agent Reasoning (`agent_reasoning`)**: This node uses an OpenAI `ChatOpenAI` model (`gpt-4o-mini`) to decide which tool to call next, based on a system prompt that guides it through the data science pipeline (clean -> select features -> train model). It orchestrates the process and provides analysis.

*   **LangGraph Workflow**: The `StateGraph` defines the flow:
    1.  Starts at the `scientist` node (which executes `agent_reasoning`).
    2.  If the `scientist` decides to call a tool, it transitions to the `tools` node (which executes the selected tool).
    3.  After a tool is executed, it transitions back to the `scientist` node for further reasoning or to determine if the task is complete.
    4.  The process continues until the agent determines its task is `END`.

## How to Run

1.  Ensure all dependencies are installed and your OpenAI API key is set.
2.  Run all the code cells in the notebook sequentially.
3.  The agent's execution is triggered by the `app.stream` call with a human message specifying the task:
    ```python
    query = "Clean 'speeddating.csv', select top 10 features, and predict match probability."

    for output in app.stream({"messages": [HumanMessage(content=query)]}, stream_mode="updates"):
        # ... (logging and output display logic)
    ```

## Expected Output
The output will show the agent's step-by-step reasoning (`🤔 THOUGHT:`), the actions it takes (`🛠️ ACTION:`), the observations from tool execution (`👁️ OBSERVATION:`), and finally, its `✅ FINAL ANALYSIS:` including the reported ROC AUC score for the trained model.

## Experimental Agent (`experimental_agent_reasoning`)
An additional `experimental_agent_reasoning` function is provided to demonstrate how the agent's behavior can be modified. This version guides the agent to experiment with different numbers of features ([5, 10, 15]) and report the best performing feature count.

```python
def experimental_agent_reasoning(state: AgentState):
    system_prompt = SystemMessage(content=(
        "You are a Match Prediction Expert conducting feature selection experiments. "
        "First, clean the data. Then, for each of [5, 10, 15] features: "
        "1. Select that number of features "
        "2. Train a model "
        "3. Record the ROC AUC score "
        "Finally, report which feature count gave the best performance."
    ))
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}
