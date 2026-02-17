import os
import re
import time
import json
import asyncio
import traceback
import logging
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

# Configuração LLM via .env (cliente usa síntese, pode ser diferente dos servidores)
LLM_CALL = os.getenv("LLM_CALL_CLIENT", "API")  # "API" ou "Ollama"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:30b")

def strip_think_tags(text: str) -> str:
    """Remove blocos <think>...</think> de modelos reasoning (ex: Qwen3, DeepSeek-R1)."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

if LLM_CALL == "API" and not DEEPSEEK_API_KEY:
    logger.error("DEEPSEEK_API_KEY não encontrada e LLM_CALL='API'. Verifique o arquivo .env")
    # Considerar sair ou definir um fallback se a API for essencial e a chave estiver faltando
    # sys.exit(1) 
elif LLM_CALL == "Ollama":
    logger.info("LLM_CALL configurado para 'Ollama'. A chave DEEPSEEK_API_KEY não será usada para a síntese principal.")

# Cliente OpenAI para DeepSeek (será usado se LLM_CALL == "API")
openai_client = None
if LLM_CALL == "API" and DEEPSEEK_API_KEY:
    openai_client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=DEEPSEEK_API_KEY
    )
elif LLM_CALL == "API" and not DEEPSEEK_API_KEY:
    logger.warning("LLM_CALL é 'API', mas DEEPSEEK_API_KEY não está definida. A síntese via API falhará.")


# Configurações dos servidores MCP
MCP_SERVERS = {
    "TELEBRAS": {"url": "http://localhost:8011/mcp/", "description": "Conhecimento TELEBRAS."},
    "CEITEC": {"url": "http://localhost:8009/mcp/", "description": "Conhecimento CEITEC."},
    "IMBEL": {"url": "http://localhost:8010/mcp/", "description": "Conhecimento IMBEL."}
}

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
                        timeout=120.0
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

def create_consolidated_summary(query: str, results_dict: dict) -> str:
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

**INSTRUÇÕES DE ESTRUTURAÇÃO:**

1. **Para perguntas sobre UMA empresa:**
   - Foque EXCLUSIVAMENTE na empresa mencionada
   - Ignore informações de outras empresas
   - Estruture: Introdução breve → Análise detalhada → Conclusão

2. **Para perguntas comparativas ou gerais:**
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
    """
    context_str = "\n\n".join([f"FONTE {r['server']}:\n{r['answer']}" for r in valid_responses])
    user_prompt_content = f"PERGUNTA: {query}\n\nDADOS DAS FONTES:\n{context_str}\n\nRESPOSTA SINTETIZADA:"
    
    synthesized_answer = ""

    try:
        if LLM_CALL == "API":
            if not openai_client:
                raise ValueError("Cliente OpenAI (DeepSeek API) não inicializado. Verifique DEEPSEEK_API_KEY.")
            logger.info("Enviando para síntese LLM via API (DeepSeek)...")
            api_response = openai_client.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": user_prompt_content}
                ],
                temperature=1.0, 
                max_tokens=6000
            )
            synthesized_answer = strip_think_tags(api_response.choices[0].message.content)
        
        elif LLM_CALL == "Ollama":
            logger.info(f"Enviando para síntese LLM via Ollama ({OLLAMA_MODEL})...")
            ollama_llm = ChatOllama(model=OLLAMA_MODEL, temperature=1.0, num_gpu=1)
            messages_for_ollama = [
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=user_prompt_content)
            ]
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
        final_response_str = create_consolidated_summary(message, results_data)
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
    
    if errors_list and (mode != "aggregated" or not any(err_msg.startswith(mode) for err_msg in errors_list)):
        filtered_errors = [e for e in errors_list if not e.startswith(f"{mode}:")] if mode != "aggregated" else errors_list
        if filtered_errors:
            final_response_str += f"\n\nObs: Outros servidores com erros: {', '.join(filtered_errors)}"
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
    with gr.Blocks(title="Chat RAG MGI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Chat RAG Unificado - MGI")
        gr.Markdown("Faça uma pergunta para consultar as bases de conhecimento TELEBRAS, CEITEC e IMBEL.")
        
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(height=600, label="Chat Consolidado", type='messages')
                query_input = gr.Textbox(placeholder="Digite sua pergunta...", container=False)
            
            with gr.Column(scale=3):
                company_radio = gr.Radio(
                    choices=["Todas", "TELEBRAS", "CEITEC", "IMBEL"],
                    label="Empresa específica (opcional)",
                    value="Todas"
                )
                
                topic_radio = gr.Radio(
                    choices=["Geral", "Finanças", "Projetos", "Produtos", "Institucional"],
                    label="Tópico específico (opcional)",
                    value="Geral"
                )
        
        async def process_query(message: str, history: list, company: str, topic: str):
            # Modificar a consulta com base nas seleções
            enhanced_query = message
            if company != "Todas":
                enhanced_query = f"[{company}] {enhanced_query}"
            if topic != "Geral":
                enhanced_query = f"[{topic}] {enhanced_query}"
            
            # Adiciona a mensagem do usuário ao histórico no formato correto
            history.append({"role": "user", "content": message})
            
            # Await a chamada de função assíncrona
            # Passando um histórico vazio para async_rag_mcp_response, pois ele não utiliza o histórico de chat para contexto.
            bot_response_string = await async_rag_mcp_response(enhanced_query, [], "aggregated")
            
            # Adiciona a resposta do bot ao histórico no formato correto
            history.append({"role": "assistant", "content": bot_response_string})
            
            # Retorna o histórico atualizado
            return history
        
        submit_btn = gr.Button("Enviar")
        submit_btn.click(
            process_query,
            inputs=[query_input, chatbot, company_radio, topic_radio],
            outputs=chatbot
        )

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
            demo.launch(share=False, server_name="0.0.0.0", server_port=port_to_use, show_error=True, debug=True, prevent_thread_lock=True)
            return # Sucesso
        except Exception as e_launch:
            logger.error(f"Erro ao usar porta {port_to_use} da variável de ambiente: {e_launch}", exc_info=True)
            raise # Relança se a porta especificada falhar
            
    ports_to_try = [8520, 8525, 8530, 7860, 7861] # Lista de portas comuns para Gradio
    logger.info(f"Tentando portas para o servidor Gradio: {ports_to_try}")
    for port_val in ports_to_try:
        try:
            logger.info(f"Tentando iniciar Gradio na porta {port_val}...")
            demo.launch(share=False, server_name="0.0.0.0", server_port=port_val, show_error=True, debug=True, prevent_thread_lock=True)
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