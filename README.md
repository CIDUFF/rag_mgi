# RAG MGI - Sistema de Consulta a Documentos

Sistema de consulta baseado em RAG (Retrieval-Augmented Generation) para análise de empresas estatais (CEITEC, IMBEL, Telebras), utilizando LLM DeepSeek, embeddings locais e arquitetura MCP (Model Context Protocol).

## Arquitetura

O sistema utiliza uma arquitetura baseada em MCP (Model Context Protocol) onde:
- Cada empresa possui seu próprio servidor MCP com base vetorial dedicada
- Um cliente MCP unifica as consultas aos diferentes servidores
- Os dados são extraídos do banco PostgreSQL (`mgi_raspagem`)

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client                              │
│                   (mcp-client/)                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│MCP CEITEC │  │MCP IMBEL  │  │MCP Telebras│
│(mcp-serv.)│  │(mcp-serv.)│  │(mcp-serv.) │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │
      ▼              ▼              ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ChromaDB   │  │ChromaDB   │  │ChromaDB   │
│CEITEC     │  │IMBEL      │  │Telebras   │
└───────────┘  └───────────┘  └───────────┘
```

## Tecnologias Utilizadas

### Core
- Python 3.12+
- LangChain - Framework para LLM
- DeepSeek - Modelo de linguagem
- Sentence Transformers (E5 Multilingual) - Embeddings locais
- ChromaDB - Banco de dados vetorial
- PostgreSQL - Banco de dados fonte (mgi_raspagem)
- MCP (Model Context Protocol) - Protocolo de comunicação

### Dependências Principais
- langchain-deepseek: Integração com modelo DeepSeek
- langchain-community: Componentes comunitários do LangChain
- sentence-transformers/multilingual-e5-large: Modelo de embeddings
- chromadb: Armazenamento vetorial
- psycopg2-binary: Driver PostgreSQL
- mcp: Model Context Protocol

## Estrutura do Projeto

```
rag_mgi/
├── mcp-servidores/              # Servidores MCP por empresa
│   ├── mcp_CEITEC.py            # Servidor MCP para CEITEC
│   ├── mcp_IMBEL.py             # Servidor MCP para IMBEL
│   └── mcp_Telebras.py          # Servidor MCP para Telebras
├── mcp-client/                  # Cliente MCP unificado
│   ├── mcp_client.py            # Cliente que conecta aos servidores
│   └── test_telebras_client.py  # Testes do cliente
├── rag_2/                       # Lógica RAG por empresa
│   ├── rag_cria_bd_CEITEC.py    # Criação do banco vetorial CEITEC
│   ├── rag_cria_bd_IMBEL.py     # Criação do banco vetorial IMBEL
│   ├── rag_cria_bd_Telebras.py  # Criação do banco vetorial Telebras
│   ├── rag_rebuild_from_postgres.py  # Reconstrução a partir do PostgreSQL
│   ├── update_vectordb_CEITEC.py
│   ├── update_vectordb_IMBEL.py
│   └── update_vectordb_Telebras.py
├── processing/                  # Processamento de documentos
│   ├── agentic_chunker.py       # Chunking baseado em agente
│   ├── semantic_chunker.py      # Chunker semântico E5
│   └── conversao.py             # Conversor PDF/DOCX -> MD
├── chroma_db_semantic_CEITEC/   # Base vetorial CEITEC (gerada)
├── chroma_db_semantic_IMBEL/    # Base vetorial IMBEL (gerada)
├── chroma_db_semantic_Telebras/ # Base vetorial Telebras (gerada)
├── logs/                        # Logs dos serviços
├── .env                         # Variáveis de ambiente
├── pyproject.toml               # Configuração do projeto
├── requirements.txt             # Dependências Python
└── README.md                    # Este arquivo
```

## Configuração

1. Crie um ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instale as dependências:
```bash
uv pip install -r requirements.txt
# ou
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite .env e adicione sua DEEPSEEK_API_KEY
```

## Banco de Dados PostgreSQL

O sistema utiliza o banco `mgi_raspagem` com as seguintes tabelas:

### Por Empresa
- `tbl_artigos_{empresa}`: Artigos científicos
- `tbl_paginas_{empresa}`: Páginas web raspadas

### Notícias
- `tbl_noticias`: Todas as notícias coletadas
- `tbl_termos_busca`: Termos de busca utilizados
- `tbl_resultados_busca`: Relacionamento termo → notícia

## Executando o Sistema

### 1. Reconstruir Bancos Vetoriais (a partir do PostgreSQL)

```bash
# Reconstruir todos
python rag_2/rag_rebuild_from_postgres.py

# Reconstruir apenas uma empresa
python rag_2/rag_rebuild_from_postgres.py --empresa CEITEC
python rag_2/rag_rebuild_from_postgres.py --empresa IMBEL
python rag_2/rag_rebuild_from_postgres.py --empresa Telebras

# Modo dry-run (apenas mostra o que seria feito)
python rag_2/rag_rebuild_from_postgres.py --dry-run
```

### 2. Iniciar Servidores MCP

```bash
# Em terminais separados:
python mcp-servidores/mcp_CEITEC.py
python mcp-servidores/mcp_IMBEL.py
python mcp-servidores/mcp_Telebras.py
```

### 3. Usar o Cliente MCP

```bash
python mcp-client/mcp_client.py
```

## Fluxo de Dados

1. **Coleta**: Dados são coletados e armazenados no PostgreSQL (`mgi_raspagem`)
2. **Extração**: Script `rag_rebuild_from_postgres.py` extrai dados do PostgreSQL
3. **Processamento**: Chunking semântico com modelo E5 multilingual
4. **Indexação**: Criação de embeddings e armazenamento no ChromaDB
5. **Consulta**: Servidores MCP respondem consultas usando RAG
6. **Unificação**: Cliente MCP agrega resultados dos diferentes servidores

## Variáveis de Ambiente

```env
DEEPSEEK_API_KEY=sua_chave_aqui
```

## Licença

MIT
