import os
import re
import time
import json
import asyncio
import traceback
import logging
import hashlib
from datetime import datetime
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder  # Importando CrossEncoder para reranking local
import httpx

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chat_client.log", mode='w')
    ]
)
logger = logging.getLogger("rag_chat")

# FastMCP v2 importações
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain_community.chat_models import ChatOllama # Adicionar importação do ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage # Para formatar mensagens para ChatOllama

# Variáveis de ambiente
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Configuração LLM via .env (cliente usa síntese, pode ser diferente dos servidores)
# Opções: "API" (DeepSeek), "Ollama", "Anthropic"
LLM_CALL = os.getenv("LLM_CALL_CLIENT", "API")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:30b")

def strip_think_tags(text: str) -> str:
    """Remove blocos <think>...</think> de modelos reasoning (ex: Qwen3, DeepSeek-R1)."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

if LLM_CALL == "API" and not DEEPSEEK_API_KEY:
    logger.error("DEEPSEEK_API_KEY não encontrada e LLM_CALL='API'. Verifique o arquivo .env")
elif LLM_CALL == "Anthropic" and not ANTHROPIC_API_KEY:
    logger.error("ANTHROPIC_API_KEY não encontrada e LLM_CALL='Anthropic'. Verifique o arquivo .env")
elif LLM_CALL == "Ollama":
    logger.info("LLM_CALL configurado para 'Ollama'.")
elif LLM_CALL == "Anthropic":
    logger.info("LLM_CALL configurado para 'Anthropic' (Claude).")

# Cliente OpenAI para DeepSeek (será usado se LLM_CALL == "API")
openai_client = None
if LLM_CALL == "API" and DEEPSEEK_API_KEY:
    openai_client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=DEEPSEEK_API_KEY
    )
elif LLM_CALL == "API" and not DEEPSEEK_API_KEY:
    logger.warning("LLM_CALL é 'API', mas DEEPSEEK_API_KEY não está definida. A síntese via API falhará.")

# Cliente Anthropic (será usado se LLM_CALL == "Anthropic")
anthropic_client = None
if LLM_CALL == "Anthropic" and ANTHROPIC_API_KEY:
    try:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Cliente Anthropic inicializado com sucesso.")
    except ImportError:
        logger.error("Biblioteca 'anthropic' não instalada. Execute: pip install anthropic")
elif LLM_CALL == "Anthropic" and not ANTHROPIC_API_KEY:
    logger.warning("LLM_CALL é 'Anthropic', mas ANTHROPIC_API_KEY não está definida.")


# Configurações dos servidores MCP
MCP_SERVERS = {
    "TELEBRAS": {"url": "http://localhost:8011/mcp/", "description": "Conhecimento TELEBRAS."},
    "CEITEC": {"url": "http://localhost:8009/mcp/", "description": "Conhecimento CEITEC."},
    "IMBEL": {"url": "http://localhost:8010/mcp/", "description": "Conhecimento IMBEL."}
}

# ===== Sistema de Autenticação =====
USERS_FILE = Path(__file__).parent / "users.json"
CHAT_HISTORY_DIR = Path(__file__).parent / "chat_history"
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

def load_users() -> dict:
    """Carrega usuários do arquivo users.json."""
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        users = {}
        for u in data.get("users", []):
            users[u["username"]] = {
                "password": u["password"],
                "nome": u.get("nome", u["username"])
            }
        logger.info(f"Carregados {len(users)} usuários do arquivo {USERS_FILE}")
        return users
    except FileNotFoundError:
        logger.warning(f"Arquivo {USERS_FILE} não encontrado. Usando credenciais padrão.")
        return {"admin": {"password": "mgi2024", "nome": "Administrador"}}
    except Exception as e:
        logger.error(f"Erro ao carregar usuários: {e}")
        return {"admin": {"password": "mgi2024", "nome": "Administrador"}}

AUTH_USERS = load_users()

def authenticate(username: str, password: str) -> bool:
    """Valida credenciais do usuário."""
    if username in AUTH_USERS and AUTH_USERS[username]["password"] == password:
        logger.info(f"Login bem-sucedido: {username}")
        return True
    logger.warning(f"Tentativa de login falhou para: {username}")
    return False

def get_user_display_name(username: str) -> str:
    """Retorna o nome de exibição do usuário."""
    if username in AUTH_USERS:
        return AUTH_USERS[username].get("nome", username)
    return username

def save_chat_history(username: str, history: list, session_file: str | None = None) -> str:
    """
    Salva o histórico de chat do usuário em arquivo JSON.
    
    Args:
        username: Nome do usuário
        history: Lista de mensagens do chat
        session_file: Nome do arquivo de sessão existente (opcional).
                      Se None, cria um novo arquivo.
    
    Returns:
        Nome do arquivo usado para salvar (para rastreamento da sessão)
    """
    try:
        user_dir = CHAT_HISTORY_DIR / username
        user_dir.mkdir(exist_ok=True)
        
        # Se temos um arquivo de sessão existente, usa ele; senão, cria um novo
        if session_file:
            filepath = user_dir / session_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_file = f"chat_{timestamp}.json"
            filepath = user_dir / session_file
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "username": username,
                "timestamp": datetime.now().isoformat(),
                "messages": history
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Histórico salvo: {filepath}")
        return session_file
    except Exception as e:
        logger.error(f"Erro ao salvar histórico de {username}: {e}")
        return session_file or ""

def load_chat_sessions(username: str) -> list:
    """Carrega lista de sessões de chat do usuário."""
    user_dir = CHAT_HISTORY_DIR / username
    if not user_dir.exists():
        return []
    sessions = []
    for f in sorted(user_dir.glob("chat_*.json"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            first_msg = ""
            for msg in data.get("messages", []):
                if msg.get("role") == "user":
                    first_msg = msg["content"][:80] + ("..." if len(msg["content"]) > 80 else "")
                    break
            sessions.append({
                "file": f.name,
                "timestamp": data.get("timestamp", ""),
                "preview": first_msg or "Chat vazio",
                "message_count": len(data.get("messages", []))
            })
        except Exception:
            continue
    return sessions

def load_chat_session(username: str, filename: str) -> list:
    """Carrega uma sessão de chat específica."""
    filepath = CHAT_HISTORY_DIR / username / filename
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", [])
    except Exception as e:
        logger.error(f"Erro ao carregar sessão {filename}: {e}")
        return []

def rename_chat_session(username: str, old_filename: str, new_name: str) -> tuple[bool, str]:
    """
    Renomeia uma sessão de chat.
    
    Args:
        username: Nome do usuário
        old_filename: Nome atual do arquivo
        new_name: Novo nome desejado (sem extensão)
    
    Returns:
        tuple: (sucesso, novo_filename ou mensagem de erro)
    """
    if not new_name or not new_name.strip():
        return False, "Nome não pode ser vazio"
    
    # Sanitizar o novo nome (remover caracteres inválidos)
    safe_name = re.sub(r'[^\w\s\-_]', '', new_name.strip())[:50]
    if not safe_name:
        return False, "Nome inválido após sanitização"
    
    user_dir = CHAT_HISTORY_DIR / username
    old_path = user_dir / old_filename
    
    if not old_path.exists():
        return False, "Arquivo não encontrado"
    
    # Criar novo nome de arquivo (preservar timestamp se existir)
    timestamp_match = re.search(r'(\d{8}_\d{6})', old_filename)
    if timestamp_match:
        new_filename = f"{safe_name}_{timestamp_match.group(1)}.json"
    else:
        new_filename = f"{safe_name}.json"
    
    new_path = user_dir / new_filename
    
    # Verificar se já existe arquivo com esse nome
    if new_path.exists() and new_path != old_path:
        return False, "Já existe um chat com esse nome"
    
    try:
        # Renomear o arquivo
        old_path.rename(new_path)
        logger.info(f"Chat renomeado: {old_filename} -> {new_filename}")
        return True, new_filename
    except Exception as e:
        logger.error(f"Erro ao renomear chat: {e}")
        return False, f"Erro ao renomear: {str(e)}"

# ===== Sistema de Gerenciamento de Tokens e Contexto =====
# Limites de tokens para DeepSeek (modelo deepseek-chat tem 64k de contexto)
MAX_CONTEXT_TOKENS = 50000  # Limite seguro para input (deixando espaço para output)
COMPACTION_THRESHOLD = 0.80  # Compactar quando atingir 80% do limite
TOKENS_PER_CHAR = 0.25  # Estimativa: ~4 caracteres = 1 token para português

def estimate_tokens(text: str) -> int:
    """Estima o número de tokens em um texto (aproximação para português)."""
    if not text:
        return 0
    return int(len(text) * TOKENS_PER_CHAR)

def estimate_history_tokens(history: list) -> int:
    """Estima o total de tokens no histórico de conversa."""
    total = 0
    for msg in history:
        content = msg.get("content", "")
        total += estimate_tokens(content)
        # Adicionar overhead por mensagem (role, formatação)
        total += 4
    return total

def get_token_usage_percentage(history: list) -> float:
    """Retorna a porcentagem de tokens usados em relação ao limite."""
    tokens_used = estimate_history_tokens(history)
    return min((tokens_used / MAX_CONTEXT_TOKENS) * 100, 100.0)

def should_compact_history(history: list) -> bool:
    """Verifica se o histórico precisa ser compactado."""
    usage = get_token_usage_percentage(history)
    return usage >= (COMPACTION_THRESHOLD * 100)

def compact_history(history: list) -> tuple[list, str]:
    """
    Compacta o histórico de conversa, resumindo mensagens antigas.
    Mantém as últimas 4 mensagens intactas e resume o resto.
    
    Returns:
        tuple: (histórico compactado, resumo gerado)
    """
    if len(history) <= 6:
        return history, ""
    
    # Separar mensagens antigas das recentes
    messages_to_summarize = history[:-4]
    recent_messages = history[-4:]
    
    # Criar resumo das mensagens antigas
    summary_parts = []
    for msg in messages_to_summarize:
        role = "Usuário" if msg.get("role") == "user" else "Assistente"
        content = msg.get("content", "")[:500]  # Limitar tamanho
        summary_parts.append(f"[{role}]: {content}")
    
    summary_text = "\n".join(summary_parts)
    
    # Criar prompt para resumo
    summary_prompt = f"""Resuma de forma MUITO concisa (máximo 300 palavras) a seguinte conversa anterior, 
mantendo apenas os pontos essenciais, decisões tomadas e informações importantes mencionadas:

{summary_text}

Formato do resumo:
- Tópicos principais discutidos
- Informações importantes mencionadas
- Contexto relevante para continuidade"""
    
    # Gerar resumo usando a LLM
    compacted_summary = ""
    try:
        if LLM_CALL == "Anthropic" and anthropic_client:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system="Você é um assistente que faz resumos concisos de conversas.",
                messages=[{"role": "user", "content": summary_prompt}]
            )
            compacted_summary = response.content[0].text
        elif LLM_CALL == "API" and openai_client:
            response = openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Você é um assistente que faz resumos concisos de conversas."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            compacted_summary = strip_think_tags(response.choices[0].message.content)
        elif LLM_CALL == "Ollama":
            ollama_llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3, num_gpu=1)
            response = ollama_llm.invoke([
                SystemMessage(content="Você é um assistente que faz resumos concisos de conversas."),
                HumanMessage(content=summary_prompt)
            ])
            compacted_summary = strip_think_tags(response.content)
    except Exception as e:
        logger.error(f"Erro ao gerar resumo para compactação: {e}")
        # Fallback: criar resumo simples
        compacted_summary = f"[Resumo de {len(messages_to_summarize)} mensagens anteriores - contexto preservado]"
    
    # Criar histórico compactado
    compacted_history = [
        {"role": "system", "content": f"📋 **Resumo da conversa anterior:**\n{compacted_summary}"}
    ] + recent_messages
    
    logger.info(f"Histórico compactado: {len(history)} mensagens -> {len(compacted_history)} mensagens")
    return compacted_history, compacted_summary

def format_history_for_api(history: list) -> list:
    """Formata o histórico de chat para envio à API."""
    formatted = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ["user", "assistant", "system"]:
            formatted.append({"role": role, "content": content})
    return formatted

# Configurar device do cliente (CrossEncoder)
CUDA_DEVICE_CLIENT = int(os.getenv("CUDA_DEVICE_CLIENT", "1"))
CLIENT_DEVICE = f'cuda:{CUDA_DEVICE_CLIENT}'

# Inicializar o CrossEncoder para reranking local
try:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=CLIENT_DEVICE)
    RERANKING_ENABLED = True
    logger.info(f"CrossEncoder para reranking local inicializado com sucesso em {CLIENT_DEVICE}")
except Exception as e:
    logger.warning(f"Não foi possível inicializar o CrossEncoder: {e}")
    RERANKING_ENABLED = False

def rerank_results(query: str, results_dict: dict, top_n: int = 15) -> dict:
    """
    Reordena os resultados da consulta usando CrossEncoder para reranking.
    
    Args:
        query: A consulta original
        results_dict: Dicionário de resultados para reordenar
        top_n: Número de fontes a manter após reranking
        
    Returns:
        Dicionário com resultados reordenados
    """
    if not RERANKING_ENABLED:
        return results_dict
    
    try:
        reranked_results = {}
        
        # Para cada servidor, aplicar reranking às fontes se possível
        for server_name, result_data in results_dict.items():
            if not result_data or "error" in result_data or "content" not in result_data:
                reranked_results[server_name] = result_data
                continue
                
            content = result_data["content"]
            answer, sources = content.get("answer", ""), content.get("sources", [])
            
            if sources and len(sources) > top_n:
                # Preparar pares consulta-documento para o CrossEncoder
                doc_pairs = [(query, source) for source in sources]
                
                # Calcular scores com o CrossEncoder
                try:
                    scores = cross_encoder.predict(doc_pairs)
                    
                    # Ordenar fontes por score
                    sorted_pairs = sorted(zip(sources, scores), key=lambda x: x[1], reverse=True)
                    top_sources = [source for source, _ in sorted_pairs[:top_n]]
                    
                    # Atualizar as fontes no resultado
                    content["sources"] = top_sources
                    content["reranked"] = True
                    
                    logger.info(f"Reranking aplicado com sucesso para {server_name}: {len(sources)} -> {len(top_sources)} fontes")
                except Exception as e:
                    logger.error(f"Erro ao aplicar reranking para {server_name}: {e}")
                    # Manter as fontes originais em caso de erro
            
            reranked_results[server_name] = result_data
            
        return reranked_results
        
    except Exception as e:
        logger.error(f"Erro no processo de reranking: {e}")
        return results_dict  # Retornar resultados originais em caso de erro

async def parallel_mcp_query(query: str, max_results: int = 5, target_server: str = None) -> tuple[dict, list]:
    results = {}
    errors = []
    servers_to_query = [] # Inicializa a lista

    # Se um servidor específico for solicitado, consultar apenas ele
    if target_server and target_server in MCP_SERVERS:
        servers_to_query = [target_server]
    else:
        servers_to_query = MCP_SERVERS.keys()
        
    async def query_server(server_name: str) -> tuple[str, dict]:
        server_config = MCP_SERVERS.get(server_name)
        if not server_config:
            msg = f"Configuração para {server_name} não encontrada."
            logger.error(msg)
            return server_name, {"error": msg}
        
        # Adicionar tentativas de reconexão
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Consultando {server_name} ({server_config['url']}) com query: '{query[:50]}...' (tentativa {attempt}/{max_retries})")
                start_time_query = time.time()
                
                # Remover o parâmetro timeout que estava causando o erro
                transport = StreamableHttpTransport(url=server_config['url'])
                
                async with Client(transport=transport) as mcp_client_instance:
                    # Manter o timeout apenas na chamada wait_for
                    tools_list = await asyncio.wait_for(mcp_client_instance.list_tools(), timeout=30.0)
                    tool_name_to_call = f"query_{server_name.lower()}"
                    if not any(tool_obj.name == tool_name_to_call for tool_obj in tools_list):
                        available_tools_names = [tool_obj.name for tool_obj in tools_list]
                        msg = f"Ferramenta '{tool_name_to_call}' não encontrada em {server_name}. Disponíveis: {available_tools_names}"
                        logger.error(msg)
                        return server_name, {"error": msg}
                    
                    logger.info(f"Chamando '{tool_name_to_call}' em {server_name}...")
                    response_content_list = await asyncio.wait_for(
                        mcp_client_instance.call_tool(
                            name=tool_name_to_call,
                            arguments={"query": query, "max_results": max_results}
                        ),
                        timeout=300.0  # 5 minutos para primeira consulta (Ollama carrega modelo)
                    )
                    
                    # Processar resposta como antes...
                    response_data = None
                    if response_content_list:
                        content_item = response_content_list[0]
                        if hasattr(content_item, 'text'):
                            try:
                                response_data = json.loads(content_item.text)
                            except json.JSONDecodeError as je:
                                raw_text = content_item.text
                                msg = f"Erro ao decodificar JSON de {server_name}: {je}. Recebido: '{raw_text[:200]}...'"
                                logger.error(msg)
                                return server_name, {"error": msg, "raw_response": raw_text}
                        elif isinstance(content_item, dict):
                            response_data = content_item
                    
                    if isinstance(response_data, dict):
                        logger.info(f"Resposta recebida com sucesso de {server_name}.")
                        return server_name, {
                            "content": {
                                "answer": response_data.get("answer", "N/A"),
                                "sources": response_data.get("sources", []),
                                "processing_time": response_data.get("processing_time", time.time() - start_time_query),
                                "source_server": server_name
                            }
                        }
                    else:
                        raw_resp_str = str(response_content_list[0]) if response_content_list else "Lista de conteúdo vazia"
                        msg = f"Formato de dados de resposta inesperado de {server_name}: {type(response_data)}. Conteúdo bruto: '{raw_resp_str[:200]}...'"
                        logger.warning(msg)
                        return server_name, {"error": msg, "raw_response": raw_resp_str}
                    
            except asyncio.TimeoutError:
                msg = f"Timeout ao consultar {server_name} (tentativa {attempt}/{max_retries})."
                logger.error(msg)
                if attempt < max_retries:
                    logger.info(f"Tentando novamente em {retry_delay} segundos...")
                    await asyncio.sleep(retry_delay)
                    continue  # Tenta novamente
                return server_name, {"error": msg}
            
            except (httpx.RemoteProtocolError, httpx.ReadTimeout) as conn_err:
                msg = f"Erro de conexão com {server_name}: {type(conn_err).__name__} - {str(conn_err)} (tentativa {attempt}/{max_retries})"
                logger.error(msg)
                if attempt < max_retries:
                    logger.info(f"Tentando novamente em {retry_delay} segundos...")
                    await asyncio.sleep(retry_delay)
                    continue  # Tenta novamente
                return server_name, {"error": msg}
            
            except Exception as e:
                msg = f"Erro ao consultar {server_name}: {type(e).__name__} - {str(e)}"
                logger.error(msg)
                logger.error(traceback.format_exc())
                return server_name, {"error": msg}

        # Se chegou aqui, todas as tentativas falharam
        return server_name, {"error": f"Falha em todas as {max_retries} tentativas de conexão com {server_name}."}

    tasks = [query_server(name) for name in servers_to_query]
    if not tasks:
        logger.warning("Nenhum servidor MCP configurado para consulta.")
        return {}, ["Nenhum servidor configurado"]
    logger.info(f"Iniciando {len(tasks)} consultas MCP paralelas...")
    task_results_tuples = await asyncio.gather(*tasks, return_exceptions=True)
    for i, server_name_key in enumerate(servers_to_query):
        result_or_exc = task_results_tuples[i]
        if isinstance(result_or_exc, Exception):
            msg = f"Exceção na tarefa de consulta para {server_name_key}: {result_or_exc}"
            logger.error(msg)
            results[server_name_key] = {"error": msg}
            errors.append(msg)
        elif isinstance(result_or_exc, tuple) and len(result_or_exc) == 2:
            actual_server_name, result_dict = result_or_exc
            results[actual_server_name] = result_dict
            if result_dict and "error" in result_dict:
                errors.append(f"{actual_server_name}: {result_dict['error']}")
            elif result_dict and "content" in result_dict:
                 logger.info(f"Consulta a {actual_server_name} bem-sucedida.")
            else:
                msg = f"Resposta inesperada ou malformada de {actual_server_name}: {result_dict}"
                logger.error(msg)
                results[actual_server_name] = {"error": msg}
                errors.append(msg)
        else:
            msg = f"Formato de resultado de tarefa inesperado para {server_name_key}: {result_or_exc}"
            logger.error(msg)
            results[server_name_key] = {"error": msg}
            errors.append(msg)
    
    # Aplicar reranking às fontes se o CrossEncoder estiver disponível
    if RERANKING_ENABLED:
        logger.info("Aplicando reranking às fontes...")
        results = rerank_results(query, results)
            
    return results, errors

def create_consolidated_summary(query: str, results_dict: dict, chat_history: list = None) -> str:
    logger.info(f"Criando resumo consolidado para query: '{query[:50]}...'")
    valid_responses = []
    all_sources_dict = {} # Initialize as a dictionary
    
    # Detectar se a pergunta é específica para uma empresa
    query_lower = query.lower()
    target_company = None
    for company in MCP_SERVERS.keys():
        if company.lower() in query_lower:
            target_company = company
            break
        
    for server_name, result_data in results_dict.items():
        if not result_data or "error" in result_data or "content" not in result_data:
            error_info = result_data.get('error', 'Dados ausentes') if result_data else 'Nulo'
            logger.warning(f"Ignorando {server_name}: {error_info}")
            continue
        answer = result_data["content"].get("answer")
        if not answer:
            logger.warning(f"Resposta vazia de {server_name}, ignorando.")
            continue
        valid_responses.append({"server": server_name, "answer": answer})
        
        # Organizar fontes por empresa
        sources = result_data["content"].get("sources", [])
        if sources:
            all_sources_dict[server_name] = sources # Now using dictionary assignment
    if not valid_responses:
        logger.error("Nenhuma resposta válida para sintetizar.")
        return "Não foi possível obter informações relevantes."
    
    logger.info(f"Sintetizando {len(valid_responses)} respostas usando {LLM_CALL}...")
    
    # Prompt aprimorado para síntese de alta qualidade
    system_prompt_content = """
Você é um analista especializado em empresas estatais brasileiras (TELEBRAS, CEITEC, IMBEL), com expertise em síntese de informações complexas de múltiplas fontes.

**SEU PAPEL:**
- Sintetizar respostas de 3 bases de conhecimento especializadas
- Produzir análises coesas, precisas e bem estruturadas
- Manter rigor técnico e clareza na comunicação

**REGRA CRÍTICA DE SEPARAÇÃO DE EMPRESAS (OBRIGATÓRIO):**

⚠️ QUANDO O USUÁRIO PERGUNTAR ESPECIFICAMENTE SOBRE **UMA** EMPRESA:
- Se perguntar sobre **IMBEL**: NÃO mencione NADA sobre CEITEC ou TELEBRAS. NÃO fale do mercado da CEITEC. NÃO fale do mercado da TELEBRAS. NÃO inclua notícias da CEITEC. NÃO inclua notícias da TELEBRAS. IGNORE completamente qualquer informação das outras duas empresas.
- Se perguntar sobre **CEITEC**: NÃO mencione NADA sobre IMBEL ou TELEBRAS. NÃO fale do mercado da IMBEL. NÃO fale do mercado da TELEBRAS. NÃO inclua notícias da IMBEL. NÃO inclua notícias da TELEBRAS. IGNORE completamente qualquer informação das outras duas empresas.
- Se perguntar sobre **TELEBRAS**: NÃO mencione NADA sobre IMBEL ou CEITEC. NÃO fale do mercado da IMBEL. NÃO fale do mercado da CEITEC. NÃO inclua notícias da IMBEL. NÃO inclua notícias da CEITEC. IGNORE completamente qualquer informação das outras duas empresas.

🚫 O QUE NÃO FAZER (LISTA EXPLÍCITA):
- NÃO adicione "contexto" de outras empresas quando a pergunta for sobre uma específica
- NÃO faça comparações não solicitadas entre empresas
- NÃO mencione "enquanto isso, na empresa X..." ou "por outro lado, a empresa Y..."
- NÃO inclua dados de mercado, financeiros ou notícias de empresas não perguntadas
- NÃO "complemente" a resposta com informações de outras empresas
- NÃO sugira que o usuário "também pode se interessar" por outra empresa na resposta principal

✅ A ÚNICA exceção é quando o usuário EXPLICITAMENTE pedir comparação entre empresas ou fizer uma pergunta geral sobre "as estatais" ou "todas as empresas".

**INSTRUÇÕES DE ESTRUTURAÇÃO:**

1. **Para perguntas sobre UMA empresa:**
   - Foque EXCLUSIVAMENTE e UNICAMENTE na empresa mencionada
   - IGNORE COMPLETAMENTE informações de outras empresas (mesmo que estejam disponíveis nos dados)
   - Estruture: Introdução breve → Análise detalhada → Conclusão
   - Se os dados das FONTES incluírem informações de outras empresas, DESCARTE essas informações

2. **Para perguntas comparativas ou gerais (SOMENTE quando explicitamente solicitado):**
   - Organize por empresa com subtítulos claros (## EMPRESA)
   - Após cobrir todas, adicione seção "### Análise Comparativa" (se relevante)
   - Destaque diferenças, similaridades e contextos únicos

3. **Para perguntas técnicas/financeiras:**
   - Use terminologia precisa (EBITDA, CAPEX, ROI, etc.)
   - Apresente dados quantitativos quando disponíveis
   - Inclua contexto temporal ("em 2023", "no último triênio")

4. **Para perguntas sobre projetos/cronogramas:**
   - Estruture cronologicamente
   - Destaque marcos importantes, status atual e previsões
   - Mencione riscos ou desafios identificados

**REGRAS DE CITAÇÃO:**
- Atribua cada informação à empresa fonte ("Segundo dados da TELEBRAS...")
- Para dados específicos, cite diretamente: "A CEITEC reportou..."
- Não invente dados nem misture informações de fontes diferentes

**FORMATAÇÃO:**
- Use Markdown: títulos (##), listas, **negrito** para ênfase
- Parágrafos concisos (3-5 linhas)
- Listas para múltiplos itens

**LIMITAÇÕES:**
- Se a pergunta não relacionar-se às empresas, responda: "Esta pergunta está fora do escopo. Posso ajudar com informações sobre TELEBRAS, CEITEC ou IMBEL."
- Se faltar informação: "Os dados disponíveis não cobrem [aspecto X]. Posso detalhar [aspecto Y]."

**TOM:**
Profissional, objetivo, analítico. Evite prolixidade, mas garanta completude.

**PROIBIÇÕES ABSOLUTAS:**
- NUNCA mencione erros de servidores, falhas de conexão, timeouts ou problemas técnicos internos na resposta.
- NUNCA exiba mensagens como "servidor com erro", "falha na comunicação", "timeout" ou qualquer informação técnica de infraestrutura.
- Se uma fonte não retornou dados, simplesmente ignore-a e responda com as fontes disponíveis, sem mencionar a ausência.
- NUNCA misture informações de empresas diferentes quando a pergunta for sobre UMA empresa específica.
- NUNCA adicione "informações complementares" de CEITEC/TELEBRAS quando perguntarem sobre IMBEL.
- NUNCA adicione "informações complementares" de IMBEL/TELEBRAS quando perguntarem sobre CEITEC.
- NUNCA adicione "informações complementares" de IMBEL/CEITEC quando perguntarem sobre TELEBRAS.
- NUNCA faça comparações entre empresas a menos que o usuário PEÇA EXPLICITAMENTE.
- NUNCA inclua notícias, mercado ou dados de empresas não mencionadas na pergunta do usuário.

**ENCERRAMENTO OBRIGATÓRIO:**
Ao final de TODA resposta, você DEVE incluir uma seção de acompanhamento. Use o formato:

---

**Consigo ajudar em algo mais, como por exemplo:**
- [Sugestão 1 relacionada ao tema da pergunta — ex: aprofundar algum ponto mencionado]
- [Sugestão 2 — ex: explicar algum termo técnico que apareceu na resposta]
- [Sugestão 3 — ex: comparar com outra empresa ou explorar um aspecto diferente]

As sugestões devem ser ESPECÍFICAS e CONTEXTUAIS ao que foi perguntado e respondido, nunca genéricas. Ofereça explicações de termos técnicos, comparações entre empresas, detalhamentos de dados mencionados, ou explorações de temas adjacentes.
    """
    context_str = "\n\n".join([f"FONTE {r['server']}:\n{r['answer']}" for r in valid_responses])
    user_prompt_content = f"PERGUNTA: {query}\n\nDADOS DAS FONTES:\n{context_str}\n\nRESPOSTA SINTETIZADA:"
    
    synthesized_answer = ""
    
    # Construir mensagens incluindo histórico de conversa (se existir)
    api_messages = [{"role": "system", "content": system_prompt_content}]
    
    # Adicionar histórico de conversa para manter contexto
    if chat_history:
        history_for_api = format_history_for_api(chat_history)
        # Filtrar mensagens de sistema duplicadas e adicionar histórico
        for msg in history_for_api:
            if msg["role"] != "system":  # Evitar múltiplos system prompts
                api_messages.append(msg)
        logger.info(f"Incluindo {len(history_for_api)} mensagens de histórico no contexto")
    
    # Adicionar a pergunta atual
    api_messages.append({"role": "user", "content": user_prompt_content})

    try:
        if LLM_CALL == "Anthropic":
            if not anthropic_client:
                raise ValueError("Cliente Anthropic não inicializado. Verifique ANTHROPIC_API_KEY.")
            logger.info(f"Enviando para síntese LLM via Anthropic (Claude) com {len(api_messages)} mensagens...")
            # Separar system message das demais para API Anthropic
            system_content = ""
            anthropic_messages = []
            for msg in api_messages:
                if msg["role"] == "system":
                    system_content += msg["content"] + "\n"
                else:
                    anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
            
            api_response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=6000,
                system=system_content.strip(),
                messages=anthropic_messages
            )
            synthesized_answer = api_response.content[0].text
        
        elif LLM_CALL == "API":
            if not openai_client:
                raise ValueError("Cliente OpenAI (DeepSeek API) não inicializado. Verifique DEEPSEEK_API_KEY.")
            logger.info(f"Enviando para síntese LLM via API (DeepSeek) com {len(api_messages)} mensagens...")
            api_response = openai_client.chat.completions.create(
                model="deepseek-chat", 
                messages=api_messages,
                temperature=1.0, 
                max_tokens=6000
            )
            synthesized_answer = strip_think_tags(api_response.choices[0].message.content)
        
        elif LLM_CALL == "Ollama":
            logger.info(f"Enviando para síntese LLM via Ollama ({OLLAMA_MODEL}) com {len(api_messages)} mensagens...")
            ollama_llm = ChatOllama(model=OLLAMA_MODEL, temperature=1.0, num_gpu=1)
            # Converter para formato LangChain
            messages_for_ollama = []
            for msg in api_messages:
                if msg["role"] == "system":
                    messages_for_ollama.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    messages_for_ollama.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    from langchain_core.messages import AIMessage
                    messages_for_ollama.append(AIMessage(content=msg["content"]))
            response_ollama = ollama_llm.invoke(messages_for_ollama)
            synthesized_answer = strip_think_tags(response_ollama.content)
        
        else:
            logger.error(f"Valor de LLM_CALL ('{LLM_CALL}') não reconhecido. Não foi possível sintetizar.")
            raise ValueError(f"Configuração LLM_CALL inválida: {LLM_CALL}")

        logger.info(f"Síntese LLM gerada ({len(synthesized_answer)} chars).")
        
        # Processar o dicionário de fontes em uma lista plana para exibição
        # flat_all_sources = []
        # for server_key in all_sources_dict:
        #     for source_item in all_sources_dict[server_key]:
        #         flat_all_sources.append(f"{server_key}: {source_item}") # Add server prefix to source

        # if flat_all_sources:
        #     unique_sources = sorted(list(set(flat_all_sources)))[:15]
        #     return f"{synthesized_answer}\n\n**Fontes:**\n" + "\n".join(f"- {s}" for s in unique_sources)
        return synthesized_answer

    except Exception as e:
        logger.error(f"Erro na síntese LLM ({LLM_CALL}): {e}", exc_info=True)
        fallback_answer = "**Respostas Individuais:**\n" + "\n".join(f"**{r['server']}:**\n{r['answer']}" for r in valid_responses)
        
        # Processar o dicionário de fontes em uma lista plana para exibição
        # Fallback para fontes
        # flat_all_sources_fallback = []
        # for server_key_fb in all_sources_dict:
        #     for source_item_fb in all_sources_dict[server_key_fb]:
        #         flat_all_sources_fallback.append(f"{server_key_fb}: {source_item_fb}")

        # if flat_all_sources_fallback:
        #      fallback_answer += "\n\n**Fontes:**\n" + "\n".join(f"- {s}" for s in sorted(list(set(flat_all_sources_fallback)))[:10])
        return fallback_answer

def detect_company_in_query(query: str) -> str:
    """
    Detecta se a consulta é específica para uma empresa.
    
    Args:
        query: A consulta do usuário.
        
    Returns:
        str: Nome da empresa detectada ou None se nenhuma for detectada.
    """
    query_lower = query.lower()
    
    # Palavras-chave específicas para cada empresa
    company_keywords = {
        "CEITEC": ["ceitec", "semicondutores", "chips", "circuitos integrados", "rfid"],
        "IMBEL": ["imbel", "material bélico", "defesa", "armamentos", "munições", "explosivos"],
        "TELEBRAS": ["telebras", "telecomunicações", "internet", "satélite", "sgdc", "banda larga"]
    }
    
    # Verificar menções explícitas às empresas
    for company, keywords in company_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                return company
    
    return None


async def async_rag_mcp_response(message: str, history: list, mode: str = "aggregated") -> str:
    logger.info(f"Processando consulta (modo: {mode}): '{message[:50]}...'")
    start_time_main = time.time()
    results_data, errors_list = await parallel_mcp_query(message)
    final_response_str = ""

    if not results_data and errors_list:
        return f"Erro: Falha na comunicação com servidores. Detalhes: {'; '.join(errors_list)}"
    elif not results_data:
         return "Erro: Nenhum servidor respondeu."

    all_failed_or_empty = all(
        not rd or "content" not in rd or ("error" in rd) for rd in results_data.values()
    )
    if all_failed_or_empty:
        error_msgs = "; ".join(errors_list) if errors_list else "Respostas vazias/malformadas."
        return f"Erro ao consultar bases: {error_msgs}"

    if mode == "aggregated":
        final_response_str = create_consolidated_summary(message, results_data, chat_history=history)
    else:
        if mode in results_data and results_data[mode] and "content" in results_data[mode]:
            content = results_data[mode]["content"]
            answer, sources, proc_time = content.get("answer", "Sem resposta."), content.get("sources", []), content.get("processing_time", 0)
            final_response_str = f"{answer}"
            # if sources: final_response_str += "\n\n**Fontes:**\n" + "\n".join(f"- {s}" for s in sorted(list(set(sources)))[:5])
            final_response_str += f"\n\n[{mode}, {proc_time:.2f}s]"
        elif mode in results_data and results_data[mode] and "error" in results_data[mode]:
            final_response_str = f"Erro ({mode}): {results_data[mode]['error']}"
        else:
            final_response_str = f"{mode} não disponível ou resposta inválida."
    
    processing_time_total = time.time() - start_time_main
    logger.info(f"Processamento total da consulta: {processing_time_total:.2f}s.")
    
    # Logar erros internamente, mas NUNCA expor ao usuário
    if errors_list:
        logger.warning(f"Erros internos (não exibidos ao usuário): {', '.join(errors_list)}")
    
    return final_response_str

async def rag_aggregated_response_async(message, history):
    return await async_rag_mcp_response(message, history, "aggregated")

# Deixando as funções específicas comentadas para simplificar a interface inicial
# async def rag_telebras_response_async(message, history):
#     return await async_rag_mcp_response(message, history, "TELEBRAS")
# async def rag_ceitec_response_async(message, history):
#     return await async_rag_mcp_response(message, history, "CEITEC")
# async def rag_imbel_response_async(message, history):
#     return await async_rag_mcp_response(message, history, "IMBEL")

async def check_server_availability(name: str, url: str) -> tuple[bool, list[str] | str]:
    logger.info(f"Verificando servidor {name} em {url}")
    try:
        # Remover o parâmetro timeout daqui também
        transport = StreamableHttpTransport(url=url)
        async with Client(transport=transport) as client_check:
            try:
                tools = await asyncio.wait_for(client_check.list_tools(), timeout=10.0)
                tool_names = [tool.name for tool in tools]
                logger.info(f"Servidor {name} OK. Ferramentas: {', '.join(tool_names)}")
                return True, tool_names
            except (httpx.HTTPStatusError, RuntimeError) as server_err:
                msg = f"Erro do servidor MCP ({url}): {type(server_err).__name__} - {server_err}"
                logger.error(msg)
                return False, msg
    except asyncio.TimeoutError:
        msg = f"Timeout MCP ({url})"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"Erro MCP ({url}): {type(e).__name__} - {str(e)}"
        logger.error(msg, exc_info=True)
        return False, msg

def setup_and_launch_gradio():
    with gr.Blocks(title="Chat RAG MGI", theme=gr.themes.Soft(), css="""
        * {
            font-family: Arial, Helvetica, sans-serif !important;
        }
        .user-header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding: 8px 16px; 
            background: linear-gradient(135deg, #1a5276, #2e86c1);
            border-radius: 8px; 
            margin-bottom: 12px;
            color: white;
        }
        .user-header span { font-size: 14px; }
        .user-header .username { font-weight: bold; font-size: 15px; }
        .history-item {
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            cursor: pointer;
            font-size: 13px;
        }
        .history-item:hover { background: #f0f4f8; }
        .logout-btn {
            background: rgba(255,255,255,0.2) !important;
            border: 1px solid rgba(255,255,255,0.4) !important;
            color: white !important;
            padding: 4px 12px !important;
            border-radius: 4px !important;
            font-size: 13px !important;
            min-width: auto !important;
        }
        .logout-btn:hover {
            background: rgba(255,255,255,0.3) !important;
        }
    """) as demo:
        
        # Header com info do usuário e logout
        with gr.Row(elem_classes="user-header"):
            user_display = gr.Markdown("")
            logout_btn = gr.Button("Sair", elem_classes="logout-btn", size="sm", scale=0)
        
        # Função de logout (recarrega a página para forçar novo login)
        logout_btn.click(
            fn=None,
            js="() => { window.location.href = window.location.pathname; }"
        )
        
        gr.Markdown("# Chat RAG Unificado - MGI")
        gr.Markdown("Faça uma pergunta para consultar as bases de conhecimento TELEBRAS, CEITEC e IMBEL.")
        
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    height=600, 
                    label="Chat Consolidado", 
                    type='messages',
                    show_copy_button=True  # Botão de copiar em cada mensagem
                )
                query_input = gr.Textbox(placeholder="Digite sua pergunta...", container=False)
            
            with gr.Column(scale=3):
                company_radio = gr.Radio(
                    choices=["Todas", "TELEBRAS", "CEITEC", "IMBEL"],
                    label="Empresa específica (opcional)",
                    value="Todas"
                )
                
                gr.Markdown("---")
                
                # Indicador de uso de tokens
                gr.Markdown("### Uso do Contexto")
                token_progress = gr.Slider(
                    minimum=0, maximum=100, value=0, 
                    label="Capacidade da conversa",
                    interactive=False,
                    info="Quando chegar a 80%, a conversa será resumida automaticamente para liberar espaço."
                )
                token_status = gr.Markdown("🟢 0% - Conversa iniciada")
                
                gr.Markdown("---")
                gr.Markdown("### Histórico de Chats")
                
                new_chat_btn = gr.Button("🆕 Novo Chat", variant="primary", size="sm")
                save_chat_btn = gr.Button("💾 Salvar Chat", variant="secondary", size="sm")
                
                history_list = gr.Dropdown(
                    label="Conversas anteriores",
                    choices=[],
                    interactive=True
                )
                load_chat_btn = gr.Button("📂 Carregar Conversa", size="sm")
                
                with gr.Row():
                    rename_input = gr.Textbox(
                        placeholder="Novo nome...",
                        container=False,
                        scale=3,
                        max_lines=1
                    )
                    rename_btn = gr.Button("✏️", size="sm", scale=1)
        
        # State para rastrear o arquivo da sessão atual
        current_session_file = gr.State(value=None)
        
        # Exibir nome do usuário logado no header
        def show_user_info(request: gr.Request):
            if request and request.username:
                display_name = get_user_display_name(request.username)
                return f"👤 **{display_name}** ({request.username})"
            return "👤 Não identificado"
        
        demo.load(show_user_info, inputs=None, outputs=user_display)
        
        # Carregar lista de sessões ao abrir
        def load_user_sessions(request: gr.Request):
            if request and request.username:
                sessions = load_chat_sessions(request.username)
                choices = [(f"{s['preview']} ({s['timestamp'][:10]})", s['file']) for s in sessions]
                return gr.Dropdown(choices=choices)
            return gr.Dropdown(choices=[])
        
        demo.load(load_user_sessions, inputs=None, outputs=history_list)
        
        # Novo chat
        def new_chat():
            # Retorna None para session_file, indicando que é uma nova conversa
            return [], "", 0, "🟢 0% - Conversa iniciada", None
        
        new_chat_btn.click(new_chat, outputs=[chatbot, query_input, token_progress, token_status, current_session_file])
        
        # Salvar chat
        def save_current_chat(history: list, session_file: str, request: gr.Request):
            if request and request.username and history:
                new_session_file = save_chat_history(request.username, history, session_file)
                sessions = load_chat_sessions(request.username)
                choices = [(f"{s['preview']} ({s['timestamp'][:10]})", s['file']) for s in sessions]
                return gr.Dropdown(choices=choices), new_session_file
            return gr.Dropdown(choices=[]), session_file
        
        save_chat_btn.click(save_current_chat, inputs=[chatbot, current_session_file], outputs=[history_list, current_session_file])
        
        def get_token_status_display(percentage: float, was_compacted: bool = False) -> str:
            """Gera o texto de status baseado na porcentagem de tokens."""
            if was_compacted:
                return f"🔄 {percentage:.0f}% - Conversa foi resumida automaticamente"
            elif percentage < 50:
                return f"🟢 {percentage:.0f}% - Amplo espaço disponível"
            elif percentage < 80:
                return f"🟡 {percentage:.0f}% - Moderado"
            else:
                return f"🟠 {percentage:.0f}% - Próximo do limite (será resumido em breve)"
        
        # Carregar conversa anterior
        def load_previous_chat(selected_file: str, request: gr.Request):
            if request and request.username and selected_file:
                messages = load_chat_session(request.username, selected_file)
                token_percentage = get_token_usage_percentage(messages)
                # Retorna também o arquivo de sessão para continuar editando o mesmo chat
                return messages, token_percentage, get_token_status_display(token_percentage), selected_file
            return [], 0, "🟢 0% - Conversa iniciada", None
        
        load_chat_btn.click(load_previous_chat, inputs=[history_list], outputs=[chatbot, token_progress, token_status, current_session_file])
        
        # Renomear chat
        def rename_selected_chat(selected_file: str, new_name: str, request: gr.Request):
            if not request or not request.username:
                gr.Warning("Usuário não identificado")
                return gr.Dropdown(choices=[]), ""
            
            if not selected_file:
                gr.Warning("Selecione uma conversa para renomear")
                return gr.Dropdown(choices=[]), new_name
            
            if not new_name or not new_name.strip():
                gr.Warning("Digite um nome para a conversa")
                return gr.Dropdown(choices=[]), new_name
            
            success, result = rename_chat_session(request.username, selected_file, new_name)
            
            if success:
                gr.Info(f"Conversa renomeada com sucesso!")
                # Atualizar lista de sessões
                sessions = load_chat_sessions(request.username)
                choices = [(f"{s['preview']} ({s['timestamp'][:10]})", s['file']) for s in sessions]
                return gr.Dropdown(choices=choices, value=result), ""
            else:
                gr.Warning(f"Erro: {result}")
                sessions = load_chat_sessions(request.username)
                choices = [(f"{s['preview']} ({s['timestamp'][:10]})", s['file']) for s in sessions]
                return gr.Dropdown(choices=choices), new_name
        
        rename_btn.click(rename_selected_chat, inputs=[history_list, rename_input], outputs=[history_list, rename_input])
        
        async def process_query(message: str, history: list, company: str, session_file: str, request: gr.Request):
            username = request.username if request else "anonymous"
            logger.info(f"[{username}] Nova consulta: '{message[:50]}...' (sessão: {session_file})")
            
            was_compacted = False
            
            # Verificar se precisa compactar o histórico ANTES de adicionar a nova mensagem
            if should_compact_history(history):
                logger.info(f"[{username}] Histórico próximo do limite, compactando...")
                history, summary = compact_history(history)
                was_compacted = True
                if summary:
                    logger.info(f"[{username}] Histórico compactado. Resumo: {summary[:100]}...")
            
            # Modificar a consulta com base na empresa selecionada
            enhanced_query = message
            if company != "Todas":
                enhanced_query = f"[{company}] {enhanced_query}"
            
            # Adiciona a mensagem do usuário ao histórico no formato correto
            history.append({"role": "user", "content": message})
            
            # Passar o histórico para manter contexto da conversa
            bot_response_string = await async_rag_mcp_response(enhanced_query, history, "aggregated")
            
            # Adiciona a resposta do bot ao histórico no formato correto
            history.append({"role": "assistant", "content": bot_response_string})
            
            # Calcular uso de tokens após a resposta
            token_percentage = get_token_usage_percentage(history)
            token_status_text = get_token_status_display(token_percentage, was_compacted)
            
            # Auto-salvar chat (mantendo o mesmo arquivo de sessão)
            new_session_file = session_file
            if username != "anonymous":
                new_session_file = save_chat_history(username, history, session_file)
            
            # Retorna o histórico atualizado, indicadores de tokens e o arquivo de sessão
            return history, token_percentage, token_status_text, new_session_file
        
        submit_btn = gr.Button("Enviar")
        submit_btn.click(
            process_query,
            inputs=[query_input, chatbot, company_radio, current_session_file],
            outputs=[chatbot, token_progress, token_status, current_session_file]
        )

    # Parâmetros de autenticação
    auth_params = {
        "auth": authenticate,
        "auth_message": "🔐 Chat RAG MGI - Sistema de Consulta\n\nInsira suas credenciais para acessar o sistema."
    }
    
    env_port = os.getenv("GRADIO_SERVER_PORT")
    port_to_use = 0
    if env_port:
        try:
            port_to_use = int(env_port)
            logger.info(f"Usando porta da variável de ambiente GRADIO_SERVER_PORT: {port_to_use}")
        except ValueError:
            logger.error(f"Valor inválido para GRADIO_SERVER_PORT: '{env_port}'. Usando portas padrão.")
            port_to_use = 0 # Reseta para que tente as portas padrão
    
    if port_to_use > 0:
        try:
            demo.launch(share=False, server_name="0.0.0.0", server_port=port_to_use, show_error=True, debug=True, prevent_thread_lock=True, **auth_params)
            return # Sucesso
        except Exception as e_launch:
            logger.error(f"Erro ao usar porta {port_to_use} da variável de ambiente: {e_launch}", exc_info=True)
            raise # Relança se a porta especificada falhar
            
    ports_to_try = [8521, 8522, 8523, 7860, 7861] # Lista de portas para DEVELOP (8520 reservada para produção)
    logger.info(f"Tentando portas para o servidor Gradio: {ports_to_try}")
    for port_val in ports_to_try:
        try:
            logger.info(f"Tentando iniciar Gradio na porta {port_val}...")
            demo.launch(share=False, server_name="0.0.0.0", server_port=port_val, show_error=True, debug=True, prevent_thread_lock=True, **auth_params)
            break 
        except OSError as e_os:
            if "address already in use" in str(e_os).lower() or "cannot assign requested address" in str(e_os).lower():
                logger.warning(f"Porta {port_val} já está em uso ou endereço não pode ser atribuído.")
            else:
                logger.error(f"OSError ao tentar iniciar Gradio na porta {port_val}: {e_os}", exc_info=True)
            if port_val == ports_to_try[-1]:
                logger.critical(f"Falha ao encontrar porta disponível para Gradio.")
                raise
            logger.info("Tentando próxima porta...")
        except Exception as e_other_launch:
            logger.critical(f"Erro inesperado ao tentar iniciar Gradio na porta {port_val}: {e_other_launch}", exc_info=True)
            if port_val == ports_to_try[-1]: raise
            logger.info("Tentando próxima porta...")

async def main():
    logger.info("Iniciando RAG Chat Client...")
    print("Verificando conexão com servidores MCP...")
    all_servers_ok = True
    for name, config in MCP_SERVERS.items():
        print(f"Verificando {name}...")
        available, details = await check_server_availability(name, config["url"])
        if available:
            print(f"✓ {name} OK. Ferramentas: {', '.join(details) if isinstance(details, list) else details}")
        else:
            print(f"✗ {name} FALHA: {details}")
            all_servers_ok = False
    if not all_servers_ok:
        logger.warning("Um ou mais servidores MCP não estão disponíveis.")
    print("Configurando e iniciando interface Gradio...")
    setup_and_launch_gradio()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Aplicação encerrada.")
    except Exception as e_fatal:
        logger.critical(f"Erro fatal: {e_fatal}", exc_info=True)