from pathlib import Path
from agno.agent import Agent
from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector
import os
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.playground import Playground, serve_playground_app
from agno.models.openai import OpenAIChat
from agno.tools.sql import SQLTools
from dotenv import load_dotenv

agent_storage: str = "tmp/agents.db"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["AGNO_API_KEY"] = os.getenv("AGNO_API_KEY")
db_url = "postgresql+psycopg://ai:ai@127.0.0.1:5532/ai?connect_timeout=10"

knowledge_base = CSVKnowledgeBase(
    path=Path("Z:/Repositorios Pessoais/agno/quadras.csv"),
    vector_db=PgVector(
        table_name="quadra",
        db_url=db_url,
    ),
    num_documents=5,  # Number of documents to return on search
)

# Load the knowledge base
knowledge_base.load(recreate=True)

agent = Agent(
    tools=[SQLTools(db_url=db_url)],
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    storage=PostgresAgentStorage(
        table_name="agent_sessions",
        db_url=db_url,
    ),
    markdown=True,
    show_tool_calls=True,
    # retries=3,pip install
    search_knowledge=True,
    add_history_to_messages=True,
    debug_mode=True,
    description=(
        "Você só pode consultar e buscar informações na tabela 'quadra'"
        "Apenas utilize as colunas 'unidade','quadra','horario','disponibilidade'."
        "A coluna 'unidade' se refere ao local da quadra"
        "A coluna 'quadra' se refere ao nome da quadra"
        "A pessoa pode confundir quadra e unidade, trate sempre de verificar essas colunas"
        "A coluna 'horario' se refere as"
        "A coluna 'disponibilidade' se refere a quadra esta 'disponivel' ou 'ocupada', não é possível utilizar qualquer outra variavel"
        "Você é um assistente de locação de quadras. "
        "Você pode falar a disponibilidade das quadras e também ocupar ou deixar disponivel uma quadra. "
        "Quando o usuário disser 'alugar', 'reservar', 'marcar' ou expressões semelhantes, significa que ele quer 'ocupar' a quadra no horario selecionado"
        "Quando o usuário disser 'livre', 'disponível', 'disponivel' ou expressões semelhantes, significa que ele buscar uma quadra disponível"
    ),
)


# def enviar_mensagem(mensagem):
#     x = agent.run(mensagem.lower())
#     return x.content

app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
    # print("OI")
