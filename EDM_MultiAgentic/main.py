import os
import re
import time
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader, PDFMinerLoader

# --- 1. System Setup & RAG Integration ---

print("--- Initializing Agentic AI System for Enterprise Document Management ---")

AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2025-04-01-preview"

# Initialize local LLMs via Ollama. This setup is private and free.
llm = AzureChatOpenAI(
    model="gpt-4o-mini",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
    )

# --- RAG PIPELINE SETUP ---
# Simulate a diverse enterprise document repository in-memory.
# In a real application, this would load from a directory of files.
print("--- Loading Enterprise Knowledge Base ---")
def load_enterprise_documents(directory_path: str = "data/") -> List[Document]:
    """
    Reads and loads various enterprise documents from a specified directory,
    processes them, and returns a list of LangChain Document objects.
    """
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return []

    print(f"--- Loading documents from '{directory_path}' ---")

    # Define loaders for different file types
    # TextLoader for .txt files
    txt_loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    # UnstructuredMarkdownLoader for .md files
    md_loader = DirectoryLoader(
        directory_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    # PDFMinerLoader for .pdf files (requires 'pypdf' installed: pip install pypdf)
    pdf_loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PDFMinerLoader, # Or PyPDFLoader if preferred
        show_progress=True,
        use_multithreading=True
    )
    
    # Load documents using each loader
    docs = []
    print("Loading .txt files...")
    docs.extend(txt_loader.load())
    print("Loading .md files...")
    docs.extend(md_loader.load())
    print("Loading .pdf files...")
    docs.extend(pdf_loader.load())

    if not docs:
        print(f"No documents found in '{directory_path}' matching .txt, .md, or .pdf extensions.")
        return []

    print(f"--- Loaded {len(docs)} raw documents ---")

    # Split documents into smaller chunks for effective retrieval
    # This helps ensure that the relevant context fits into the LLM's context window.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # Max size of each chunk
        chunk_overlap=200,        # Overlap between chunks to maintain context
        length_function=len       # Use character length
    )
    
    # Create smaller, processed document splits
    document_splits = text_splitter.split_documents(docs)
    
    print(f"--- Split into {len(document_splits)} document chunks ---")
    
    return document_splits

# Chunk documents for effective retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Create the vector store (the "long-term memory" of our system)
vectorstore = FAISS.from_documents(documents=load_enterprise_documents(), embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("--- System Ready ---\n")


# --- 2. System Architecture & Agent Design ---

# This TypedDict defines the shared state that is passed between all agents.
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    validation_feedback: str

# Pydantic models provide structured output, making agent communication reliable.
class GradeDocuments(BaseModel):
    """Binary score for document relevance."""
    is_relevant: str = Field(description="Are the documents relevant to the question? 'yes' or 'no'")

class ValidationResult(BaseModel):
    """Detailed validation result with feedback."""
    is_grounded: str = Field(description="Is the generation fully supported by the provided documents? 'yes' or 'no'")
    is_relevant: str = Field(description="Does the generation directly answer the user's question? 'yes' or 'no'")
    feedback: str = Field(description="Constructive feedback for the generator if validation fails, otherwise 'None'.")

# --- Agent Implementations as Nodes in the Graph ---

def retriever_agent(state: GraphState) -> GraphState:
    """Agent 1: Retrieves documents from the knowledge base."""
    print("--- AGENT: Retriever ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {**state, "documents": documents}

def grader_agent(state: GraphState) -> GraphState:
    """Agent 2: Grades the relevance of retrieved documents."""
    print("--- AGENT: Grader ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        # If no documents are retrieved, they are inherently not relevant.
        print("--- GRADER: No documents retrieved. ---")
        return {**state, "documents": []}
        
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    prompt = ChatPromptTemplate.from_template(
        """You are a grader. Your task is to assess the relevance of retrieved documents to a user question.
        If the documents contain keywords or semantic meaning relevant to the question, grade it as 'yes'.

        Retrieved documents: {documents}
        User question: {question}"""
    )
    chain = prompt | structured_llm_grader
    result = chain.invoke({"documents": [d.page_content for d in documents], "question": question})
    
    if result.is_relevant == "yes":
        print("--- GRADER: Documents are RELEVANT ---")
        return {**state, "documents": documents}
    else:
        print("--- GRADER: Documents are NOT RELEVANT ---")
        return {**state, "documents": []}

def query_rewriter_agent(state: GraphState) -> GraphState:
    """Agent 3: Rewrites the query if documents are not relevant."""
    print("--- AGENT: Query Rewriter ---")
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        """You are a query rewriting expert. The initial search failed. Rephrase the user's question
        to be more specific and clear for a vector database search.

        Original question: {question}"""
    )
    chain = prompt | llm
    rewritten_response = chain.invoke({"question": question})
    rewritten_question = rewritten_response.content
    print(f"--- Rewritten Question: {rewritten_question} ---")
    return {**state, "question": rewritten_question}

def generator_agent(state: GraphState) -> GraphState:
    """Agent 4: Generates an answer based on the documents."""
    print("--- AGENT: Generator (Summarizer/QA) ---")
    question = state["question"]
    documents = state["documents"]
    feedback = state.get("validation_feedback", "")
    
    system_message = """You are an enterprise Q&A assistant. Use the following documents to provide a concise, accurate answer.
    If validation feedback is provided, use it to improve your response.
    If providing evidence, cite the source document ID (e.g., 'doc_id: HR-001')."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Documents:\n{documents}\n\nUser question: {question}\n\nValidation Feedback:\n{feedback}")
    ])
    chain = prompt | llm
    generation_response = chain.invoke({"documents": [f"Content: {d.page_content}, Source: {d.metadata}" for d in documents], "question": question, "feedback": feedback})
    generation = generation_response.content
    return {**state, "generation": generation, "validation_feedback": ""}

def validator_agent(state: GraphState) -> GraphState:
    """Agent 5: Validates the answer and acts as the orchestrator."""
    print("--- AGENT: Validator (Orchestrator) ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    structured_llm_validator = llm.with_structured_output(ValidationResult)
    prompt = ChatPromptTemplate.from_template(
        """You are a meticulous validator. Check if the generated answer is grounded in the provided documents and relevant to the user's question.
        Provide constructive feedback if it fails.

        Documents: {documents}
        User Question: {question}
        Generated Answer: {generation}"""
    )
    chain = prompt | structured_llm_validator
    result = chain.invoke({"documents": [d.page_content for d in documents], "question": question, "generation": generation})

    # Redaction Logic for compliance
    redacted_generation = re.sub(r"Project\s+\w+", "[REDACTED]", generation, flags=re.IGNORECASE)

    if result.is_grounded == "yes" and result.is_relevant == "yes":
        print("--- VALIDATOR: Verdict: Approve ---")
        return {**state, "generation": redacted_generation, "validation_feedback": "approve"}
    else:
        print("--- VALIDATOR: Verdict: Reject ---")
        return {**state, "generation": redacted_generation, "validation_feedback": result.feedback}

# --- Conditional Edges for Agentic Flow ---

def decide_to_generate(state: GraphState) -> str:
    """Conditional Edge 1: Determines if the retrieved documents are good enough to proceed."""
    print("--- CONDITIONAL EDGE: Assess Document Relevance ---")
    if not state["documents"]:
        print("--- DECISION: No relevant documents. Triggering query rewrite. ---")
        return "rewrite_query"
    else:
        print("--- DECISION: Relevant documents found. Proceeding to generation. ---")
        return "generate"

def decide_to_finish(state: GraphState) -> str:
    """Conditional Edge 2: Determines if the generated answer is good enough to be sent to the user."""
    print("--- CONDITIONAL EDGE: Final Validation ---")
    if state["validation_feedback"] == "approve":
        print("--- DECISION: Generation approved. Finishing workflow. ---")
        return END
    else:
        print("--- DECISION: Generation rejected. Retrying generation with feedback. ---")
        return "generate"

# --- Build the Graph ---

workflow = StateGraph(GraphState)

# Add nodes (the agents)
workflow.add_node("retrieve", retriever_agent)
workflow.add_node("grade_documents", grader_agent)
workflow.add_node("rewrite_query", query_rewriter_agent)
workflow.add_node("generate", generator_agent)
workflow.add_node("validate", validator_agent)

# Define the workflow edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edge("grade_documents", decide_to_generate)
workflow.add_edge("rewrite_query", "retrieve") # This creates the self-correction loop
workflow.add_edge("generate", "validate")
workflow.add_conditional_edge("validate", decide_to_finish) # This creates the revision loop

# Compile the graph into a runnable application
app = workflow.compile()
print("\n--- Agentic AI System Graph Compiled ---\n")


# --- 3. Main Execution Block ---

if __name__ == "__main__":
    sample_queries = [
        "Summarize the policy changes in our HR manual.",
        "Is the marketing budget for Q2 approved yet? Give evidence.",
        "What is the company's official position on work from home?", # This query is slightly ambiguous to test the rewriter
    ]

    for i, query in enumerate(sample_queries):
        print(f"\n--- Running Test Case {i+1} ---")
        print(f"Query: \"{query}\"")
        start_time = time.time()
        
        inputs = {"question": query}
        # We invoke the graph with our inputs. The recursion limit prevents infinite loops.
        final_state = app.invoke(inputs, {"recursion_limit": 5})
        
        end_time = time.time()
        latency = end_time - start_time
        
        print("\n" + "="*50)
        print("Final Answer:")
        print(final_state["generation"])
        print(f"\nLatency: {latency:.2f} seconds")
        print("="*50 + "\n")

    print("--- All test cases complete. Entering interactive mode. ---")
    print("Ask a question about the enterprise documents (or type 'quit' to exit).")
    while True:
        user_question = input("\nYou: ")
        if user_question.lower() in ["quit", "exit"]:
            break
        
        inputs = {"question": user_question}
        final_state = app.invoke(inputs, {"recursion_limit": 5})
        print("\nAssistant:")
        print(final_state["generation"])