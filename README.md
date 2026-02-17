# RAG MGI - Sistema de Consulta a Documentos de Empresas Estatais

Sistema de consulta baseado em **RAG** (Retrieval-Augmented Generation) para análise de empresas estatais brasileiras (**CEITEC**, **IMBEL**, **Telebras**), utilizando LLM DeepSeek, embeddings locais com aceleração GPU e arquitetura **MCP** (Model Context Protocol).

## Arquitetura

O sistema segue uma arquitetura **MCP (Model Context Protocol)** com separação clara entre servidores e cliente:

- **3 MCP Servers** — cada empresa possui seu próprio servidor MCP com base vetorial ChromaDB dedicada, cadeia RAG independente (retrieval + reranking + LLM) e ferramentas/recursos específicos
- **1 MCP Client** — ponto de entrada unificado que consulta os 3 servidores em paralelo, aplica reranking global, sintetiza as respostas via LLM e expõe uma interface Gradio para o usuário

```
┌──────────────────────────────────────────────────────────────────┐
│                       MCP Client (Gradio)                        │
│                      mcp-client/mcp_client.py                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Consulta     │  │ Reranking    │  │ Síntese LLM            │  │
│  │ paralela     │  │ CrossEncoder │  │ (DeepSeek API / Ollama)│  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
└────────────┬──────────────┬──────────────┬───────────────────────┘
             │ HTTP/MCP     │ HTTP/MCP     │ HTTP/MCP
             ▼              ▼              ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ MCP Server     │ │ MCP Server     │ │ MCP Server     │
│ CEITEC (:8009) │ │ IMBEL (:8010)  │ │ Telebras(:8011)│
│                │ │                │ │                │
│ • query_ceitec │ │ • query_imbel  │ │• query_telebras│
│ • RAG Chain    │ │ • RAG Chain    │ │• RAG Chain     │
│ • CrossEncoder │ │ • CrossEncoder │ │• CrossEncoder  │
│ • Recursos     │ │ • Recursos     │ │• Recursos      │
│ • Prompts      │ │ • Prompts      │ │• Prompts       │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                   │
        ▼                  ▼                   ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ ChromaDB       │ │ ChromaDB       │ │ ChromaDB       │
│ CEITEC         │ │ IMBEL          │ │ Telebras       │
│ 233k chunks    │ │ 232k chunks    │ │ 224k chunks    │
└────────────────┘ └────────────────┘ └────────────────┘
        ▲                  ▲                   ▲
        └──────────────────┼───────────────────┘
                           │
                  ┌────────────────┐
                  │  PostgreSQL    │
                  │  (lei_bem)     │
                  │  178k notícias │
                  │  129 artigos   │
                  │  5.6k páginas  │
                  └────────────────┘
```

### MCP Servers (mcp-servidores/)

Cada servidor MCP é uma aplicação **FastMCP** independente que:

1. **Inicializa** a base vetorial ChromaDB no startup (lifespan)
2. **Configura** uma cadeia RAG completa: retriever MMR → reranking CrossEncoder → geração LLM
3. **Expõe** ferramentas (tools), recursos (resources) e prompts via protocolo MCP sobre HTTP Streamable
4. Suporta **DeepSeek API** ou **Ollama local** como LLM (configurável via `LLM_CALL`)

| Servidor | Porta | Tool principal | Recurso |
|----------|-------|----------------|---------|
| CEITEC   | 8009  | `query_ceitec` | `ceitec://overview` |
| IMBEL    | 8010  | `query_imbel`  | `imbel://overview` |
| Telebras | 8011  | `query_telebras` | `telebras://overview` |

### MCP Client (mcp-client/)

O cliente MCP (`mcp_client.py`) é o **orquestrador** do sistema:

1. **Consulta paralela** — envia a query a todos os servidores MCP simultaneamente via `asyncio.gather`
2. **Reranking global** — aplica CrossEncoder (`ms-marco-MiniLM-L6-v2`) para reordenar as fontes retornadas
3. **Detecção de empresa** — identifica automaticamente se a pergunta é sobre uma empresa específica
4. **Síntese LLM** — consolida as respostas dos diferentes servidores em uma resposta coesa via DeepSeek/Ollama
5. **Interface Gradio** — expõe chat web com seleção de empresa e tópico

## Tecnologias

| Componente | Tecnologia |
|-----------|------------|
| Linguagem | Python 3.12+ |
| Gerenciamento de pacotes | [uv](https://docs.astral.sh/uv/) |
| Framework LLM | LangChain |
| Modelo de linguagem | DeepSeek Chat (API) / Ollama (local) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (GPU) |
| Reranking | CrossEncoder `ms-marco-MiniLM-L6-v2` |
| Banco vetorial | ChromaDB |
| Banco de dados fonte | PostgreSQL 15 |
| Protocolo de comunicação | MCP (Model Context Protocol) via FastMCP v2 |
| Interface | Gradio |
| Aceleração | CUDA (PyTorch) — testado com 2x NVIDIA RTX 3060 12GB |

## Estrutura do Projeto

```
rag_mgi/
├── mcp-servidores/                     # Servidores MCP (1 por empresa)
│   ├── mcp_CEITEC.py                   #   Servidor CEITEC (porta 8009)
│   ├── mcp_IMBEL.py                    #   Servidor IMBEL (porta 8010)
│   └── mcp_Telebras.py                 #   Servidor Telebras (porta 8011)
├── mcp-client/                         # Cliente MCP unificado
│   └── mcp_client.py                   #   Orquestrador + interface Gradio
├── rag_2/                              # Scripts de construção do RAG
│   ├── rag_rebuild_from_postgres.py    #   Rebuild completo (GPU, paralelo)
│   └── update_vectordb_from_postgres.py#   Atualização incremental
├── processing/                         # Processamento de documentos
│   ├── agentic_chunker.py              #   Chunking baseado em agente
│   ├── semantic_chunker.py             #   Chunker semântico
│   └── conversao.py                    #   Conversor PDF/DOCX → Markdown
├── chroma_db_semantic_CEITEC/          # Base vetorial CEITEC (gerada)
├── chroma_db_semantic_IMBEL/           # Base vetorial IMBEL (gerada)
├── chroma_db_semantic_Telebras/        # Base vetorial Telebras (gerada)
├── MCP-materiais/                      # Documentação de referência MCP
├── data/                               # Dados brutos e processados
├── logs/                               # Logs dos serviços
├── .env                                # Variáveis de ambiente
├── pyproject.toml                      # Configuração e dependências (uv)
└── README.md
```

## Configuração

### 1. Instalar dependências

```bash
# Recomendado: usar uv
uv sync

# Alternativa: pip
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# LLM (DeepSeek API)
DEEPSEEK_API_KEY=sua_chave_aqui

# Banco de Dados PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=lei_bem
DB_USER=seu_usuario
DB_PASSWORD=sua_senha

# Portas dos servidores MCP (opcionais, valores padrão)
PORT_CEITEC=8009
PORT_IMBEL=8010
PORT_TELEBRAS=8011
```

## Banco de Dados PostgreSQL

O sistema utiliza o banco **`lei_bem`** como fonte de dados, com as seguintes tabelas:

| Tabela | Descrição | Registros |
|--------|-----------|-----------|
| `tbl_noticias` | Notícias coletadas | ~178.888 |
| `tbl_resultados_busca` | Relação termo → notícia | ~183.467 |
| `tbl_termos_busca` | Termos de busca | 27 |
| `tbl_artigos_ceitec` | Artigos científicos CEITEC | 68 |
| `tbl_artigos_imbel` | Artigos científicos IMBEL | 29 |
| `tbl_artigos_telebras` | Artigos científicos Telebras | 32 |
| `tbl_paginas_ceitec` | Páginas web CEITEC | 251 |
| `tbl_paginas_imbel` | Páginas web IMBEL | 1.120 |
| `tbl_paginas_telebras` | Páginas web Telebras | 4.256 |

## Executando o Sistema

### 1. Construir/Reconstruir Bancos Vetoriais

Os scripts extraem dados do PostgreSQL, aplicam chunking com paralelização em CPU e geram embeddings com aceleração GPU.

```bash
# Reconstruir todos os bancos (apaga e recria)
uv run python rag_2/rag_rebuild_from_postgres.py

# Reconstruir apenas uma empresa
uv run python rag_2/rag_rebuild_from_postgres.py --empresa CEITEC

# Atualização incremental (adiciona apenas dados novos)
uv run python rag_2/update_vectordb_from_postgres.py

# Dry-run (apenas mostra o que seria feito)
uv run python rag_2/rag_rebuild_from_postgres.py --dry-run

# Controlar paralelização
uv run python rag_2/rag_rebuild_from_postgres.py --workers 4
```

**Benchmark** (2x RTX 3060, 8 workers):

| Empresa | Documentos | Chunks gerados | Tempo |
|---------|-----------|----------------|-------|
| CEITEC | 88.860 | 233.087 | ~17 min |
| IMBEL | 86.829 | 231.930 | ~23 min |
| Telebras | 95.915 | 223.529 | ~23 min |
| **Total** | **271.604** | **688.546** | **~63 min** |

### 2. Iniciar os Servidores MCP

Cada servidor precisa rodar em um processo separado:

```bash
# Terminal 1
uv run python mcp-servidores/mcp_CEITEC.py

# Terminal 2
uv run python mcp-servidores/mcp_IMBEL.py

# Terminal 3
uv run python mcp-servidores/mcp_Telebras.py
```

Cada servidor inicializa sua base vetorial, configura a cadeia RAG e fica escutando na respectiva porta via HTTP Streamable.

### 3. Iniciar o Cliente MCP (interface de chat)

```bash
uv run python mcp-client/mcp_client.py
```

O cliente verifica automaticamente a disponibilidade dos servidores, inicializa o CrossEncoder para reranking e abre a interface Gradio.

## Fluxo de Dados

```
PostgreSQL (lei_bem)
       │
       │  rag_rebuild_from_postgres.py
       │  (chunking paralelo + embeddings GPU)
       ▼
ChromaDB × 3 (CEITEC, IMBEL, Telebras)
       │
       │  MCP Servers (FastMCP, Streamable HTTP)
       │  (retrieval MMR → reranking CrossEncoder → LLM)
       ▼
MCP Client
       │
       │  (consulta paralela → reranking global → síntese LLM)
       ▼
Interface Gradio (chat web)
```

## Licença

MIT
