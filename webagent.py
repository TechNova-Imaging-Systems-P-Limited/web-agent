import os
import json
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from mem0 import Memory
import config


mem0_config = {
    "llm": {
        "provider": "azure_openai",
        "config": {
            "model": config.AZURE_DEPLOYMENT_NAME,
            "azure_kwargs": {
                "api_key": config.AZURE_OPENAI_API_KEY,
                "azure_deployment": config.AZURE_DEPLOYMENT_NAME,
                "azure_endpoint": config.AZURE_OPENAI_ENDPOINT,
                "api_version": config.AZURE_OPENAI_API_VERSION
            }
        }
    },
    "embedder": {
        "provider": "azure_openai",
        "config": {
            "model": config.AZURE_EMBEDDING_DEPLOYMENT_NAME,
            "azure_kwargs": {
                "api_key": config.AZURE_OPENAI_API_KEY,
                "azure_deployment": config.AZURE_EMBEDDING_DEPLOYMENT_NAME,
                "azure_endpoint": config.AZURE_OPENAI_ENDPOINT,
                "api_version": config.AZURE_OPENAI_API_VERSION
            }
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "web_agent_memory",
            "path": "d:/projects/web_agent/chroma_db"
        }
    },
    "history_db_path": "d:/projects/web_agent/history.db" 
}

m = Memory.from_config(mem0_config)

def get_recent_messages(session_id, limit=5):
    try:
        all_memories = m.get_all(user_id=session_id)


        if isinstance(all_memories, dict):
            msgs = all_memories.get("results", [])
        elif isinstance(all_memories, list):
            msgs = all_memories
        else:
            msgs = []

        chat_msgs = []
        for mem in msgs:
           if not isinstance(mem, dict):
               continue
               
           content = mem.get("memory", "")
           
           metadata = mem.get("metadata") or {}
           role = metadata.get("msg_role") or mem.get("role") or metadata.get("role") or "unknown"
           
           if role in ["user", "assistant"]:
               chat_msgs.append({"role": role, "content": content})
        
        return chat_msgs[-limit:]
    except Exception as e:
        print(f"Error fetching recent messages: {e}")
        return []

def get_session_history(session_id):
    return get_recent_messages(session_id, limit=1000)

search = GoogleSerperAPIWrapper(serper_api_key=config.SERPER_API_KEY)

web_search_tool = Tool(
    name="google_search",
    func=search.run,
    description="Useful for when you need to answer questions about current events, market data, or specific facts. Input should be a search query."
)

tools = [web_search_tool]

llm = AzureChatOpenAI(
    azure_deployment=config.AZURE_DEPLOYMENT_NAME,  
    openai_api_version=config.AZURE_OPENAI_API_VERSION, 
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_key=config.AZURE_OPENAI_API_KEY,
    temperature=0,             
    verbose=True
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized Web Search Agent. "
        "Your goal is to find accurate, up-to-date information on the internet. "
        "1. Always use the 'google_search' tool for factual queries. "
        "2. If the search results are not sufficient, you may search again with a better query. "
        "3. Summarize the findings concisely for the user. "
        "4. Do not make up information. If you can't find it, say so.\n"
        "5. **IMPORTANT LANGUAGE INSTRUCTION**: You MUST provide your final answer to the user in {user_language}, using daily/conversational language. DO NOT use overly formal language.\n\n"
        "Here is some relevant context from previous conversations (Mem0):\n"
        "{mem0_context}\n\n"
        "Here are the last few messages of this conversation:\n"
        "{recent_history}\n\n"
        "Current Date and Time: {current_date}"
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_web_agent(query: str, session_id: str, language: str = "Default English"):
    try:
        print(f" Web Agent Searching for: {query} (Session: {session_id}, Language: {language})")

        search_results = m.search(query, user_id=session_id)
        relevant_memories = search_results.get("results", []) if isinstance(search_results, dict) else search_results
        mem0_context = "\n".join([mem["memory"] for mem in relevant_memories]) if relevant_memories else "No relevant past context found."

        recent_msgs = get_recent_messages(session_id)
        recent_history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_msgs])

        response = agent_executor.invoke({
            "input": query,
            "mem0_context": mem0_context,
            "recent_history": recent_history_str,
            "current_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_language": language
        })
        output = response["output"]

        m.add(query, user_id=session_id, metadata={"role": "user", "msg_role": "user"}, infer=False)

        m.add(output, user_id=session_id, metadata={"role": "assistant", "msg_role": "assistant"}, infer=False)

        m.add(f"User asking about {query}", user_id=session_id, metadata={"type": "fact"}, infer=True)

        return output
    except Exception as e:
        return f"Web Agent Error: {str(e)}"

if __name__ == "__main__":
    test_session = "test_user_123"
    print(run_web_agent("My name is HB", test_session))
    print(run_web_agent("What is my name?", test_session))
