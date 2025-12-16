Setup and Installation
1. Prerequisites
Python 3.8+

An Azure account with access to Azure OpenAI Service.

2. Environment Setup
Clone the Repository (or Save the Script):
Save the code as enterprise_agent.py.

Create a Virtual Environment (Recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies:
Create a requirements.txt file with the following content:

langchain
langgraph
langchain-openai
faiss-cpu
python-dotenv
tiktoken

Then, install the packages:

bash
pip install -r requirements.txt

Configure Environment Variables:
The system requires credentials to connect to your Azure OpenAI service. Create a file named .env in the same directory as the script and add the following information:

env
# Your Azure OpenAI API Key
OPENAI_API_KEY="<YOUR_AZURE_OPENAI_API_KEY>"

# Your Azure OpenAI Endpoint URL
OPENAI_API_ENDPOINT="<YOUR_AZURE_OPENAI_ENDPOINT>"