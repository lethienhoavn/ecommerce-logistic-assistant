# ✨ AI Assistant for Parcel Delivery Operations

This project is an intelligent **AI assistant** tailored for **e-commerce logistics and parcel delivery companies**, enabling natural, context-aware interactions to support operations, customer service, and real-time data queries.

It features a web-based interface powered by **Gradio**, and integrates seamlessly with leading large language models (LLMs) such as **OpenAI**, **Ollama**, and **HuggingFace**,...

### Key Features

- **🧏 Multimodal Interaction Support**:
  - **🖼️ Visual Generation Capabilities**: Ability to generate personalized greeting cards or visuals using user input — e.g., “create a thank-you card for a delivery driver” — perfect for customer appreciation, holidays, or custom requests.
  - **🎤 Audio-Based Conversations**: Supports speech-to-text and text-to-speech for full-duplex audio conversations, enabling natural spoken dialogue in environments where typing is impractical.

- **🔗 RAG System (Retrieval-Augmented Generation)**: 
  Integrates vector search (e.g., FAISS, Chroma) with LLMs to retrieve and generate accurate, context-aware answers based on internal documentation and delivery data.

- **🧠 AI Assistant with Memory**: 
  Uses memory to maintain conversation context, enabling multi-turn dialogue and more human-like interactions.

- **🛠️ Assistant Tools**: Enhanced with custom **LangChain** tools and agents to:
  - Query internal **SQL/NoSQL databases**
  - Call external/internal **parcel APIs** (e.g., ETA/EDT estimation)
  - Answer questions related to **delivery status, time, routes**, and more

- **📦 Use Case Focus**: Built specifically to serve **parcel delivery operations**, providing smart automation and real-time Q&A for operations teams, customer service, or end users.


By combining RAG, tool usage, API calls, and memory-based agents, this chatbot acts as a domain-specific AI copilot for modern logistics workflows.

<br>

### 🛠️ 1. Install Dependencies

This project needs these tools:
- Ollama (address: `http://localhost:11434/`). Some basic commands:
  - Run `ollama serve` to start Ollama Server on local
  - Run `ollama run llama2` to pull and run Llama2
  - Run `ollama stop llama2` to unload model from RAM
  - Run `ollama rm llama2` to remove llama2 model from the disk

And Python libraries:
  ```bash
  pip install -r requirements.txt
  ````

### ▶️ 2. Run the Application

```bash
python app.py
```

### 🌐 3. Access the Agent

Open your browser and navigate to:

```
http://localhost:7860
```

<br>
<br>

## MORE 


### 🔄 Finetuning LLMs

* **LoRA / QLoRA / PEFT** – lightweight finetuning LLM
* **DeepSpeed / Hugging Face PEFT / BitsAndBytes** – helpers to optimize training cost and memory (RAM/GPU) usage
* **DPO / PPO / RLHF** – align LLMs using human feedback



### 🔍 LLM Monitoring

* **LangSmith, Weights & Biases, MLflow, Arize AI, WhyLabs** - monitor LLM performance, detect drift, and track input/output data
* **Prompt Tracking & Versioning** - keep records of prompt history and associated responses



### 🚀 Deployment

* **Hugging Face Transformers & Hub** – access, finetune, or deploy open-source LLMs.
* **vLLM / TGI (Text Generation Inference)** – high-performance LLM serving solutions.
* **Ray Serve / FastAPI / Triton Inference Server** – frameworks for deploying GenAI inference APIs.



### ✅ Evaluation & Quality Control

* **TruLens / Ragas / Promptfoo** – automated evaluation tools for RAG and prompt-based applications.
* **Guardrails AI / Rebuff / LMQL** – control LLM output and mitigating risks (e.g., toxicity, hallucinations).
* **LLM-as-a-judge** – use LLMs to automatically evaluate the quality of model-generated outputs.

