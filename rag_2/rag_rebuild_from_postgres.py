#!/usr/bin/env python3
"""
Script para reconstruir os bancos vetoriais ChromaDB a partir do PostgreSQL.
VERSÃO OTIMIZADA COM PROCESSAMENTO EM PARALELO.

Otimizações:
- Chunking em paralelo usando multiprocessing
- Batch processing para embeddings
- Processamento de empresas em paralelo (opcional)

Uso:
    python rag_rebuild_from_postgres_parallel.py [--empresa EMPRESA] [--workers N]
    
    EMPRESA pode ser: CEITEC, IMBEL, Telebras ou ALL (padrão)
    --workers: Número de workers para processamento paralelo (padrão: número de CPUs)
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Adicionar diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Carregar variáveis de ambiente
load_dotenv()

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Configurações do banco de dados PostgreSQL (via variáveis de ambiente)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'mgi_raspagem'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# Validar configurações do banco
if not DB_CONFIG['user'] or not DB_CONFIG['password']:
    print("ERRO: Variáveis de ambiente DB_USER e DB_PASSWORD são obrigatórias.")
    print("Configure no arquivo .env:")
    print("  DB_USER=seu_usuario")
    print("  DB_PASSWORD=sua_senha")
    sys.exit(1)

# Configurações dos bancos vetoriais
CHROMA_DB_DIRS = {
    'CEITEC': './chroma_db_semantic_CEITEC',
    'IMBEL': './chroma_db_semantic_IMBEL',
    'Telebras': './chroma_db_semantic_Telebras'
}

PROCESSED_FILES_RECORDS = {
    'CEITEC': './processed_files_CEITEC.json',
    'IMBEL': './processed_files_IMBEL.json',
    'Telebras': './processed_files_Telebras.json'
}

# Modelo de embeddings
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Configurações de paralelização
DEFAULT_WORKERS = min(mp.cpu_count(), 8)  # Limitar a 8 workers
BATCH_SIZE = 100  # Tamanho do batch para embeddings

# Mapeamento de termos de busca para empresas
TERMOS_EMPRESA = {
    'CEITEC': {
        'exclusivos': ['ceitec'],
        'compartilhados': ['semicondutores', 'semiconductors']
    },
    'IMBEL': {
        'exclusivos': ['imbel'],
        'compartilhados': ['defense-industry', 'industria-de-defesa', 'industria-de-defesa-BR']
    },
    'Telebras': {
        'exclusivos': ['telebrás', 'telebrás-inss'],
        'compartilhados': ['telecomunicações', 'satélite']
    }
}

TERMOS_GERAIS = [
    'empresa', 'privatização', 'ministério-mgi', 
    'ministério-das-comunicações', 'ministério-da-defesa',
    'ministério-da-educação'
]

# ============================================================================
# FUNÇÕES DE CONEXÃO E CONSULTA AO BANCO
# ============================================================================

def get_db_connection():
    """Estabelece conexão com o PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Erro ao conectar ao PostgreSQL: {e}")
        raise

def get_termos_busca(conn) -> Dict[int, str]:
    """Retorna dicionário com id -> termo_busca."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, termo_busca FROM tbl_termos_busca")
        rows = cur.fetchall()
        return {row['id']: row['termo_busca'] for row in rows}

def get_termo_ids_por_nome(termos_dict: Dict[int, str], nomes: List[str]) -> Set[int]:
    """Retorna os IDs dos termos que correspondem aos nomes fornecidos."""
    ids = set()
    for termo_id, termo_nome in termos_dict.items():
        if termo_nome and termo_nome.lower() in [n.lower() for n in nomes]:
            ids.add(termo_id)
    return ids

def get_noticias_por_termos(conn, termo_ids: Set[int]) -> List[Dict]:
    """Busca notícias associadas aos termos de busca especificados."""
    if not termo_ids:
        return []
    
    placeholders = ','.join(['%s'] * len(termo_ids))
    query = f"""
        SELECT DISTINCT n.id, n.title, n.description, n.content, 
               n.link, n.published, n.publisher_name
        FROM tbl_noticias n
        INNER JOIN tbl_resultados_busca rb ON n.id = rb.id_noticia
        WHERE rb.id_termo_busca IN ({placeholders})
        AND n.content IS NOT NULL
        AND LENGTH(n.content) > 100
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, tuple(termo_ids))
        return cur.fetchall()

def get_artigos(conn, tabela: str) -> List[Dict]:
    """Busca artigos de uma tabela específica."""
    query = f"""
        SELECT id, titulo, ano, abstract, conteudo, autores, doi
        FROM {tabela}
        WHERE conteudo IS NOT NULL AND LENGTH(conteudo) > 50
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query)
        return cur.fetchall()

def get_paginas(conn, tabela: str) -> List[Dict]:
    """Busca páginas de uma tabela específica."""
    query = f"""
        SELECT id, content, link, resumo, dt_download
        FROM {tabela}
        WHERE content IS NOT NULL AND LENGTH(content) > 50
        AND (status IS NULL OR status != 'error')
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query)
        return cur.fetchall()

# ============================================================================
# FUNÇÕES DE CRIAÇÃO DE DOCUMENTOS
# ============================================================================

def criar_documento_noticia(noticia: Dict, empresa: str) -> Document:
    """Cria um Document LangChain a partir de uma notícia."""
    partes = []
    if noticia.get('title'):
        partes.append(f"# {noticia['title']}")
    if noticia.get('description'):
        partes.append(noticia['description'])
    if noticia.get('content'):
        partes.append(noticia['content'])
    
    texto = "\n\n".join(partes)
    
    return Document(
        page_content=texto,
        metadata={
            'source': f"noticia_{noticia['id']}",
            'type': 'noticia',
            'empresa': empresa,
            'title': noticia.get('title', ''),
            'link': noticia.get('link', ''),
            'published': str(noticia.get('published', '')),
            'publisher': noticia.get('publisher_name', ''),
            'db_id': noticia['id']
        }
    )

def criar_documento_artigo(artigo: Dict, empresa: str) -> Document:
    """Cria um Document LangChain a partir de um artigo."""
    partes = []
    if artigo.get('titulo'):
        partes.append(f"# {artigo['titulo']}")
    if artigo.get('autores'):
        partes.append(f"**Autores:** {artigo['autores']}")
    if artigo.get('ano'):
        partes.append(f"**Ano:** {artigo['ano']}")
    if artigo.get('abstract'):
        partes.append(f"**Resumo:** {artigo['abstract']}")
    if artigo.get('conteudo'):
        partes.append(artigo['conteudo'])
    
    texto = "\n\n".join(partes)
    
    return Document(
        page_content=texto,
        metadata={
            'source': f"artigo_{artigo['id']}",
            'type': 'artigo',
            'empresa': empresa,
            'title': artigo.get('titulo', ''),
            'ano': artigo.get('ano', ''),
            'autores': artigo.get('autores', ''),
            'doi': artigo.get('doi', ''),
            'db_id': artigo['id']
        }
    )

def criar_documento_pagina(pagina: Dict, empresa: str) -> Document:
    """Cria um Document LangChain a partir de uma página."""
    partes = []
    if pagina.get('resumo'):
        partes.append(f"**Resumo:** {pagina['resumo']}")
    if pagina.get('content'):
        partes.append(pagina['content'])
    
    texto = "\n\n".join(partes) if partes else pagina.get('content', '')
    
    return Document(
        page_content=texto,
        metadata={
            'source': f"pagina_{pagina['id']}",
            'type': 'pagina',
            'empresa': empresa,
            'link': pagina.get('link', ''),
            'dt_download': str(pagina.get('dt_download', '')),
            'db_id': pagina['id']
        }
    )

# ============================================================================
# FUNÇÕES DE CHUNKING PARALELO
# ============================================================================

def chunk_single_document(args: Tuple[int, str, dict, str]) -> List[Tuple[str, dict]]:
    """
    Processa um único documento com chunking simples (por tamanho).
    Retorna lista de (texto, metadata) para cada chunk.
    
    Usando chunking por tamanho fixo para performance.
    """
    idx, text, metadata, empresa = args
    
    # Chunking simples por tamanho (mais rápido que semântico)
    MAX_CHUNK_SIZE = 2000  # caracteres
    OVERLAP = 200  # sobreposição entre chunks
    
    chunks = []
    
    if len(text) <= MAX_CHUNK_SIZE:
        # Documento pequeno, não precisa dividir
        new_metadata = {
            **metadata,
            'chunk_id': f"{metadata.get('source', 'doc')}_{0}",
            'chunk_idx': 0,
            'total_chunks': 1,
            'processed_date': datetime.now().isoformat()
        }
        chunks.append((text, new_metadata))
    else:
        # Dividir em chunks
        start = 0
        chunk_idx = 0
        text_chunks = []
        
        while start < len(text):
            end = start + MAX_CHUNK_SIZE
            
            # Tentar quebrar em um espaço para não cortar palavras
            if end < len(text):
                # Procurar último espaço antes do limite
                last_space = text.rfind(' ', start, end)
                if last_space > start + OVERLAP:
                    end = last_space
            
            text_chunks.append(text[start:end].strip())
            start = end - OVERLAP if end < len(text) else end
            chunk_idx += 1
        
        total_chunks = len(text_chunks)
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text:
                new_metadata = {
                    **metadata,
                    'chunk_id': f"{metadata.get('source', 'doc')}_{i}",
                    'chunk_idx': i,
                    'total_chunks': total_chunks,
                    'processed_date': datetime.now().isoformat()
                }
                chunks.append((chunk_text, new_metadata))
    
    return chunks

def processar_documentos_paralelo(documentos: List[Document], empresa: str, num_workers: int) -> List[Document]:
    """
    Processa os documentos com chunking em paralelo.
    """
    print(f"\nProcessando {len(documentos)} documentos com {num_workers} workers...")
    
    # Preparar argumentos para processamento paralelo
    args_list = [
        (i, doc.page_content, doc.metadata, empresa) 
        for i, doc in enumerate(documentos)
    ]
    
    processed_chunks = []
    total = len(args_list)
    
    # Usar ProcessPoolExecutor para paralelização
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submeter todos os trabalhos
        futures = {executor.submit(chunk_single_document, args): i for i, args in enumerate(args_list)}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 500 == 0 or completed == total:
                print(f"  Progresso: {completed}/{total} documentos ({100*completed/total:.1f}%)")
            
            try:
                chunks_result = future.result()
                for text, metadata in chunks_result:
                    processed_chunks.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                idx = futures[future]
                print(f"  Erro no documento {idx}: {e}")
    
    print(f"  Total de chunks gerados: {len(processed_chunks)}")
    return processed_chunks

# ============================================================================
# FUNÇÕES PRINCIPAIS
# ============================================================================

def coletar_documentos_empresa(conn, empresa: str, termos_dict: Dict[int, str]) -> List[Document]:
    """
    Coleta todos os documentos para uma empresa específica.
    """
    print(f"\n{'='*60}")
    print(f"Coletando documentos para: {empresa}")
    print(f"{'='*60}")
    
    documentos = []
    
    # 1. Buscar artigos
    tabela_artigos = f"tbl_artigos_{empresa.lower()}"
    print(f"\nBuscando artigos em {tabela_artigos}...")
    artigos = get_artigos(conn, tabela_artigos)
    print(f"  Encontrados {len(artigos)} artigos")
    
    for artigo in artigos:
        doc = criar_documento_artigo(artigo, empresa)
        if len(doc.page_content) > 50:
            documentos.append(doc)
    
    # 2. Buscar páginas
    tabela_paginas = f"tbl_paginas_{empresa.lower()}"
    print(f"\nBuscando páginas em {tabela_paginas}...")
    paginas = get_paginas(conn, tabela_paginas)
    print(f"  Encontradas {len(paginas)} páginas")
    
    for pagina in paginas:
        doc = criar_documento_pagina(pagina, empresa)
        if len(doc.page_content) > 50:
            documentos.append(doc)
    
    # 3. Buscar notícias relacionadas
    print(f"\nBuscando notícias para {empresa}...")
    
    config_termos = TERMOS_EMPRESA.get(empresa, {'exclusivos': [], 'compartilhados': []})
    termos_exclusivos = config_termos['exclusivos']
    termos_compartilhados = config_termos['compartilhados']
    todos_termos = termos_exclusivos + termos_compartilhados + TERMOS_GERAIS
    
    termo_ids = get_termo_ids_por_nome(termos_dict, todos_termos)
    print(f"  Termos de busca utilizados: {todos_termos}")
    print(f"  IDs dos termos: {termo_ids}")
    
    noticias = get_noticias_por_termos(conn, termo_ids)
    print(f"  Encontradas {len(noticias)} notícias")
    
    for noticia in noticias:
        doc = criar_documento_noticia(noticia, empresa)
        if len(doc.page_content) > 100:
            documentos.append(doc)
    
    print(f"\nTotal de documentos para {empresa}: {len(documentos)}")
    return documentos

def reconstruir_banco_vetorial(empresa: str, chunks: List[Document]):
    """
    Reconstrói o banco vetorial ChromaDB para uma empresa.
    """
    chroma_dir = CHROMA_DB_DIRS[empresa]
    processed_file = PROCESSED_FILES_RECORDS[empresa]
    
    print(f"\n{'='*60}")
    print(f"Reconstruindo banco vetorial: {chroma_dir}")
    print(f"{'='*60}")
    
    # 1. Apagar banco antigo se existir
    if os.path.exists(chroma_dir):
        print(f"Removendo banco antigo: {chroma_dir}")
        shutil.rmtree(chroma_dir)
    
    # 2. Apagar registro de arquivos processados
    if os.path.exists(processed_file):
        print(f"Removendo registro antigo: {processed_file}")
        os.remove(processed_file)
    
    # 3. Criar embeddings e banco vetorial
    print(f"\nCriando embeddings com modelo {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={'batch_size': BATCH_SIZE}  # Batch processing para embeddings
    )
    
    print(f"Criando banco vetorial com {len(chunks)} chunks...")
    print(f"  Processando em batches de {BATCH_SIZE}...")
    
    # Criar em batches para melhor uso de memória
    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE * 10):
        batch = chunks[i:i + BATCH_SIZE * 10]
        print(f"  Processando batch {i//BATCH_SIZE//10 + 1} ({i}/{len(chunks)} chunks)...")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=chroma_dir
            )
        else:
            vectorstore.add_documents(batch)
    
    # 4. Salvar registro detalhado
    artigos_ids = set()
    paginas_ids = set()
    noticias_ids = set()
    
    for chunk in chunks:
        doc_type = chunk.metadata.get('type', '')
        db_id = chunk.metadata.get('db_id')
        if db_id:
            if doc_type == 'artigo':
                artigos_ids.add(db_id)
            elif doc_type == 'pagina':
                paginas_ids.add(db_id)
            elif doc_type == 'noticia':
                noticias_ids.add(db_id)
    
    registro = {
        'empresa': empresa,
        'data_criacao': datetime.now().isoformat(),
        'data_atualizacao': datetime.now().isoformat(),
        'total_chunks': len(chunks),
        'fonte': 'PostgreSQL (mgi_raspagem)',
        'versao': 'parallel_v1',
        'estatisticas': {
            'artigos': len(artigos_ids),
            'paginas': len(paginas_ids),
            'noticias': len(noticias_ids)
        },
        'documentos_processados': {
            'artigos': sorted(list(artigos_ids)),
            'paginas': sorted(list(paginas_ids)),
            'noticias': sorted(list(noticias_ids))
        }
    }
    
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(registro, f, ensure_ascii=False, indent=2)
    
    print(f"\nBanco vetorial criado com sucesso!")
    print(f"  - Diretório: {chroma_dir}")
    print(f"  - Registro: {processed_file}")
    print(f"  - Total de chunks: {len(chunks)}")
    print(f"  - Artigos: {len(artigos_ids)}, Páginas: {len(paginas_ids)}, Notícias: {len(noticias_ids)}")
    
    return vectorstore

def processar_empresa(empresa: str, num_workers: int):
    """
    Processa uma empresa completa: coleta dados, aplica chunking e cria banco vetorial.
    """
    inicio = datetime.now()
    print(f"\n{'#'*70}")
    print(f"# PROCESSANDO: {empresa}")
    print(f"# Início: {inicio.isoformat()}")
    print(f"{'#'*70}")
    
    # Conectar ao banco
    conn = get_db_connection()
    
    try:
        # Obter dicionário de termos
        termos_dict = get_termos_busca(conn)
        print(f"Termos de busca disponíveis: {len(termos_dict)}")
        
        # Coletar documentos
        documentos = coletar_documentos_empresa(conn, empresa, termos_dict)
        
        if not documentos:
            print(f"AVISO: Nenhum documento encontrado para {empresa}")
            return
        
        # Aplicar chunking paralelo
        chunks = processar_documentos_paralelo(documentos, empresa, num_workers)
        
        # Reconstruir banco vetorial
        reconstruir_banco_vetorial(empresa, chunks)
        
        fim = datetime.now()
        duracao = fim - inicio
        print(f"\n{'#'*70}")
        print(f"# {empresa} CONCLUÍDO em {duracao}")
        print(f"{'#'*70}")
        
    finally:
        conn.close()

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Reconstrói bancos vetoriais ChromaDB a partir do PostgreSQL (versão paralela)'
    )
    parser.add_argument(
        '--empresa',
        type=str,
        default='ALL',
        choices=['CEITEC', 'IMBEL', 'Telebras', 'ALL'],
        help='Empresa a processar (padrão: ALL)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help=f'Número de workers para processamento paralelo (padrão: {DEFAULT_WORKERS})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Apenas mostra o que seria feito, sem executar'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("RECONSTRUÇÃO DOS BANCOS VETORIAIS RAG (VERSÃO PARALELA)")
    print(f"Data/Hora: {datetime.now().isoformat()}")
    print(f"Workers: {args.workers}")
    print("="*70)
    
    # Testar conexão
    print("\nTestando conexão com PostgreSQL...")
    try:
        conn = get_db_connection()
        print("  Conexão estabelecida com sucesso!")
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) as total FROM tbl_noticias")
            total_noticias = cur.fetchone()['total']
            print(f"  Total de notícias: {total_noticias}")
            
            for empresa in ['ceitec', 'imbel', 'telebras']:
                cur.execute(f"SELECT COUNT(*) as total FROM tbl_artigos_{empresa}")
                total_artigos = cur.fetchone()['total']
                cur.execute(f"SELECT COUNT(*) as total FROM tbl_paginas_{empresa}")
                total_paginas = cur.fetchone()['total']
                print(f"  {empresa.upper()}: {total_artigos} artigos, {total_paginas} páginas")
        
        conn.close()
    except Exception as e:
        print(f"  ERRO na conexão: {e}")
        sys.exit(1)
    
    if args.dry_run:
        print("\n[DRY-RUN] Simulação - nenhuma alteração será feita")
        return
    
    # Processar empresas
    if args.empresa == 'ALL':
        empresas = ['CEITEC', 'IMBEL', 'Telebras']
    else:
        empresas = [args.empresa]
    
    inicio_total = datetime.now()
    
    for empresa in empresas:
        processar_empresa(empresa, args.workers)
    
    fim_total = datetime.now()
    duracao_total = fim_total - inicio_total
    
    print("\n" + "="*70)
    print("RECONSTRUÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"Tempo total: {duracao_total}")
    print("="*70)

if __name__ == "__main__":
    main()
