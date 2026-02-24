import os

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "Replace_With_Your_Azure_API_Key")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://openai-ragbot.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4.1-mini")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "Replace_With_Your_Serper_API_Key")