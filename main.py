import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# LangChain Agent Integrations
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool

load_dotenv()

app = FastAPI(
    title="McCarthy Ultra-Glass AI Agent",
    description="Knowledge-based AI incorporating John McCarthy's Advice Taker principles."
)

# 1. Initialize McCarthy Perception & Knowledge Tools
serp = SerpAPIWrapper()
web_search_tool = Tool(
    name="Web_Search",
    func=serp.run,
    description="Useful for current events, fresh facts, news, and real-time information."
)

wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1200)
wiki_tool = Tool(
    name="Wikipedia_Knowledge",
    func=wiki.run,
    description="Useful for general background facts, scientific concepts, and historical knowledge."
)

tools = [wiki_tool, web_search_tool]

# 2. McCarthy Reasoning Engine (Advice Taker Pattern)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = (
    "You are an AI system built on John McCarthy's principles of formal reasoning "
    "and knowledge representation. You evaluate user prompts logically, decide whether "
    "to query static encyclopedic knowledge (Wikipedia) or live environment data (SerpAPI), "
    "and synthesize clear, actionable advice."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. Request/Response Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str

# 4. API Endpoints
@app.post("/api/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    try:
        result = agent_executor.invoke({"input": request.query})
        return QueryResponse(query=request.query, answer=result["output"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Serve Ultra-Glass UI Frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")
