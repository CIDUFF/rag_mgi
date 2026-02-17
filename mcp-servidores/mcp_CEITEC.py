import os
import sys
import re
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dotenv import load_dotenv
from types import SimpleNamespace

# Adicionar o diretório raiz do projeto ao sys.path ANTES da importação local
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importações locais
import json
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from contextlib import asynccontextmanager

# Configuração de dispositivo (GPU/CPU)
CUDA_DEVICE = int(os.getenv("CUDA_DEVICE_CEITEC", "0"))
DEVICE = f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu'

# Importações para reranking
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# Importar FastMCP v2
from fastmcp import FastMCP, Context

# Definir as constantes de configuração
CHROMA_DB_DIR_CEITEC = "./chroma_db_semantic_CEITEC"
PROCESSED_FILES_RECORD = "./processed_files_CEITEC.json"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INITIAL_RETRIEVAL_K = 30  # Recuperação inicial mais ampla
LLM_CALL = os.getenv("LLM_CALL_SERVER", "Ollama")  # "API", "Ollama" ou "Anthropic" — configurável via .env
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:30b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def strip_think_tags(text: str) -> str:
    """Remove blocos <think>...</think> de modelos reasoning (ex: Qwen3, DeepSeek-R1)."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

# Verificar se a chave de API Cohere está definida
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
USE_CROSS_ENCODER = True  # Determina se deve usar CrossEncoder local se a API Cohere não estiver disponível

def load_processed_files() -> dict:
    """Carrega o registro de arquivos processados."""
    if os.path.exists(PROCESSED_FILES_RECORD):
        try:
            with open(PROCESSED_FILES_RECORD, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

async def initialize_vectorstore():
    try:
        if os.path.exists(CHROMA_DB_DIR_CEITEC) and os.path.isdir(CHROMA_DB_DIR_CEITEC):
            print(f"Carregando base de dados vetorial existente de {CHROMA_DB_DIR_CEITEC}...")
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': DEVICE},
                encode_kwargs={'device': DEVICE, 'normalize_embeddings': True}
            )
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR_CEITEC, embedding_function=embeddings)
            registro = load_processed_files()
            total_chunks = registro.get('total_chunks', 'N/A')
            print(f"Base de dados vetorial CEITEC carregada. Chunks: {total_chunks}")
        else:
            raise FileNotFoundError(f"Base vetorial não trovata em {CHROMA_DB_DIR_CEITEC}. Execute rag_rebuild_from_postgres.py primeiro.")
        return vectorstore
    except Exception as e:
        print(f"Erro ao inicializar base de dados vetorial para CEITEC: {e}")
        raise

def initialize_rag_chain(vectorstore):
    # Configurar o retrievador base
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": INITIAL_RETRIEVAL_K, "fetch_k": 50, "lambda_mult": 0.7}
    )
    
    # Inicializar o componente de reranking
    if USE_CROSS_ENCODER:
        try:
            from sentence_transformers import CrossEncoder
            
            print("Utilizando CrossEncoder local para reranking...")
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
            
            def cross_encoder_rerank(query: str, docs: List[Document], top_n: int = 15) -> List[Document]:
                if not docs:
                    return []
                doc_pairs = [(query, doc.page_content) for doc in docs]
                scores = cross_encoder.predict(doc_pairs)
                doc_score_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in doc_score_pairs[:top_n]]
            
            # Definir a classe wrapper de retriever aqui
            class CustomRerankingRetriever(BaseRetriever):
                underlying_retriever: BaseRetriever
                custom_rerank_function: Callable[[str, List[Document]], List[Document]]

                async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
                    docs = await self.underlying_retriever.aget_relevant_documents(query, **kwargs)
                    return self.custom_rerank_function(query, docs)
                
                def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
                    docs = self.underlying_retriever.get_relevant_documents(query, **kwargs)
                    return self.custom_rerank_function(query, docs)

            retriever = CustomRerankingRetriever(
                underlying_retriever=base_retriever,
                custom_rerank_function=lambda q, d: cross_encoder_rerank(q, d, top_n=15)
            )
            print("Retriever com reranking customizado (CrossEncoder) inicializado.")
        except ImportError:
            print("Aviso: Não foi possível inicializar CrossEncoder, utilizando retrievador base...")
            retriever = base_retriever
    else:
        print("Aviso: CrossEncoder desativado. Utilizando retrievador base...")
        retriever = base_retriever
    
    # Template de prompt
    template = """
    Você é um assistente IA especializado em análise de documentos da empresa CEITEC. Sua função é fornecer informações precisas e contextualizadas com base nos documentos fornecidos.

    INSTRUÇÕES GERAIS:
    1. Analise cuidadosamente o contexto e a pergunta para identificar o tipo de informação solicitada.
    2. Use EXCLUSIVAMENTE as informações fornecidas no contexto para responder à pergunta.
    3. Seja específico e cite fontes quando possível, incluindo referências a documentos específicos.
    4. IMPORTANTE: Suas respostas devem se referir APENAS à empresa CEITEC. NÃO inclua informações sobre outras empresas (como IMBEL e Telebras) mesmo que a pergunta mencione essas empresas.
    5. Se a pergunta solicitar explicitamente uma comparação com outras empresas, informe que você só possui informações sobre CEITEC.
    6. Para perguntas sobre finanças, utilize terminologia financeira precisa e apresente dados quantitativos quando disponíveis.
    7. Para perguntas sobre projetos, estruture a resposta cronologicamente e inclua informações sobre status, orçamento e cronograma quando disponíveis.

    Contexto: {context}
    Pergunta: {question}
    Resposta:
    """
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # Selecionar LLM com base na configuração LLM_CALL
    if LLM_CALL == "API":
        print("Usando LLM via API: ChatDeepSeek para CEITEC")
        llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, max_tokens=5000, timeout=None, max_retries=3)
    elif LLM_CALL == "Ollama":
        print("Usando LLM via Ollama: deepseek-llm para CEITEC")
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=1.0, top_k=40, top_p=0.9, num_ctx=4096)
    elif LLM_CALL == "Anthropic":
        print("Usando LLM via Anthropic: Claude para CEITEC")
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=1.0, max_tokens=5000, api_key=ANTHROPIC_API_KEY)
    else:
        print(f"AVISO (CEITEC): Valor de LLM_CALL ('{LLM_CALL}') não reconhecido. Usando ChatDeepSeek por padrão.")
        llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, max_tokens=5000, timeout=None, max_retries=3)
        
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True, chain_type_kwargs={"prompt": PROMPT}
    )
    return rag_chain

@asynccontextmanager
async def app_lifespan(app: FastMCP):
    estado_app = SimpleNamespace()

    print(f"Lifespan: Inicializando recursos para {app.name}...")
    try:
        estado_app.vectorstore = await initialize_vectorstore()
        estado_app.rag_chain = initialize_rag_chain(estado_app.vectorstore)
        print(f"Lifespan: Recursos para {app.name} inicializados com sucesso.")
        
        # CORREÇÃO IMPORTANTE: Definir explicitamente app.state
        app.state = estado_app
        
        yield estado_app
    except Exception as e:
        error_type = type(e).__name__
        print(f"Lifespan ERRO CRÍTICO ({error_type}) durante a inicialização de recursos: {e}")
        raise
    finally:
        print(f"Lifespan: Encerrando servidor {app.name}, limpando recursos...")
        if hasattr(estado_app, 'vectorstore'):
            print("Lifespan: Limpando estado_app.vectorstore")
            del estado_app.vectorstore
        if hasattr(estado_app, 'rag_chain'):
            print("Lifespan: Limpando estado_app.rag_chain")
            del estado_app.rag_chain
        print(f"Lifespan: Recursos para {app.name} limpos.")

mcp = FastMCP(
    name="Base de Conhecimento CEITEC",
    lifespan=app_lifespan,
    instructions="Servidor de conhecimento sobre a empresa CEITEC e seus produtos de semicondutores.",
    cors_origins=["*"]
)

@mcp.tool()
async def query_ceitec(query: str, max_results: int = 15, ctx: Context = None) -> dict:
    """
    Busca informações específicas sobre a empresa CEITEC na base de conhecimento.
    Args:
        query: A consulta ou pergunta sobre a CEITEC.
        max_results: Número máximo de documentos a retornar (opcional).
    Returns:
        dict: Contém a resposta e as fontes de informação.
    """
    if ctx: await ctx.info(f"Iniciando consulta CEITEC: {query}")

    if not hasattr(ctx.fastmcp.state, 'rag_chain') or ctx.fastmcp.state.rag_chain is None:
        error_msg = "Cadeia RAG (CEITEC) não inicializada. O lifespan pode ter falhado."
        if ctx: await ctx.error(error_msg)
        return {"error": error_msg, "answer": "Erro: Base de conhecimento CEITEC não disponível."}

    rag_chain = ctx.fastmcp.state.rag_chain
    start_time = time.time()
    response = await asyncio.to_thread(rag_chain, {"query": query})
    processing_time = time.time() - start_time

    result = {
        "answer": strip_think_tags(response["result"]),
        "sources": [doc.metadata['source'] for doc in response["source_documents"][:max_results]],
        "processing_time": processing_time
    }
    if ctx: await ctx.info(f"Consulta CEITEC finalizada em {processing_time:.2f}s")
    return result

@mcp.tool()
async def list_document_categories(ctx: Context = None) -> dict:
    """Lista as categorias de documentos disponíveis sobre a CEITEC."""
    if ctx: await ctx.info("Listando categorias de documentos da CEITEC")
    categories = [
        "Relatórios Financeiros", "Especificações Técnicas", "Documentos Legais",
        "Notícias", "Artigos Científicos"
    ]
    return {"categories": categories}

# Definir recursos MCP
@mcp.resource("ceitec://overview")
def ceitec_overview() -> dict:
    """Visão geral sobre a empresa CEITEC e suas áreas de atuação."""
    return {
        "name": "CEITEC S.A.",
        "full_name": "Centro Nacional de Tecnologia Eletrônica Avançada S.A.",
        "founded": "2008",
        "headquarters": "Porto Alegre, RS, Brasil",
        "industry": "Semicondutores",
        "main_products": ["Chips RFID", "Circuitos integrados para aplicações específicas"],
        "description": "A CEITEC S.A. é uma empresa pública federal brasileira, vinculada ao Ministério da Ciência, Tecnologia e Inovações, focada no desenvolvimento e produção de componentes semicondutores."
    }

@mcp.resource("ceitec://document-count")
def document_count() -> dict:
    """Informações sobre a quantidade de documentos na base de conhecimento da CEITEC."""
    return {
        "total_documents": len(load_processed_files()),
        "total_chunks": "Variável conforme processamento",
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

# Definir prompts MCP
@mcp.prompt()
def financial_response(company_name: str, financial_data: str, analysis: str, context: str, sources: str) -> str:
    """Formato para respostas sobre informações financeiras."""
    return f"""
    # Análise Financeira: {company_name}
    
    ## Dados Financeiros
    {financial_data}
    
    ## Análise
    {analysis}
    
    ## Contexto Econômico e Setorial
    {context}
    
    ## Fontes
    {sources}
    """

@mcp.prompt()
def project_detailed_response(company_name: str, project_name: str, description: str, timeline: str, budget: str, status: str, stakeholders: str, sources: str) -> str:
    """Formato para respostas detalhadas sobre projetos específicos."""
    return f"""
    # Projeto: {project_name} ({company_name})
    
    ## Descrição e Objetivos
    {description}
    
    ## Cronograma
    {timeline}
    
    ## Orçamento
    {budget}
    
    ## Status Atual
    {status}
    
    ## Partes Interessadas
    {stakeholders}
    
    ## Fontes
    {sources}
    """

@mcp.prompt()
def technical_response(summary: str, details: str, specs: str, sources: str) -> str:
    """Formato para respostas técnicas sobre semicondutores e chips da CEITEC."""
    return f"""
    # Resposta Técnica sobre CEITEC
    ## Resumo Técnico
    {summary}
    ## Detalhamento
    {details}
    ## Especificações Técnicas
    {specs}
    ## Fontes
    {sources}
    """

@mcp.prompt()
def general_response(main_content: str, additional_info: str, sources: str) -> str:
    """Formato para respostas gerais sobre a CEITEC."""
    return f"""
    # Informações sobre CEITEC
    {main_content}
    ## Informações Adicionais
    {additional_info}
    ## Fontes
    {sources}
    """

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    healthy_status = {"status": "íntegro"}
    
    # Usar app.state diretamente agora que sabemos que está definido corretamente
    if not hasattr(mcp, 'state') or mcp.state is None or not hasattr(mcp.state, 'rag_chain'):
        healthy_status["status"] = "degradado"
        healthy_status["reason"] = "Cadeia RAG não inicializada"
    return JSONResponse(healthy_status)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8009))
    load_dotenv()
    print(f"Iniciando servidor MCP para CEITEC na porta {port}...")
    print(f"Usando dispositivo: {DEVICE}")
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
        path="/mcp/"
    )