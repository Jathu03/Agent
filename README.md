# Agent

![Python](https://img.shields.io/badge/Language-Python-blue)

This repository currently houses a collection of **Python-based scripts** designed to implement various types of AI agent architectures and functionalities.

> **History:** This repository was initially designated as a GitHub Desktop tutorial repository. It has since evolved to focus entirely on mechanisms used for building intelligent agents.

## 📂 Repository Contents

The project is written entirely in **Python**. Below is a breakdown of the key files and their architectures:

| File Name | Functionality |
| :--- | :--- |
| **`RAG_Agent.py`** | Implements **Retrieval-Augmented Generation (RAG)** architecture, allowing the agent to retrieve external data to augment its responses. |
| **`ReAct.py`** | Implements the **Reasoning and Acting (ReAct)** framework, enabling the agent to reason through steps and utilize tools. |
| `Memory_Agent.py` | An agent component focused on managing conversation history, context, and information retrieval (Memory). |
| `Drafter.py` | A specialized agent designed for drafting content, responses, or structured text. |
| `Agent_Bot.py` | The core agent implementation or general bot logic. |

## 🛠️ Tech Stack

*   **Language:** Python (100%)

## 🚀 Getting Started

To explore these agents:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Jathu03/Agent.git
    cd Agent
    ```

2.  **Run the specific agent script:**
    ```bash
    python RAG_Agent.py
    # or
    python ReAct.py
    ```

*(Note: Ensure you have the necessary environment variables, such as API keys, set up depending on the libraries used within the scripts).*
