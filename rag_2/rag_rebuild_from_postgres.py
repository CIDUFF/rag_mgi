#!/usr/bin/env python3
"""
Script para reconstruir os bancos vetoriais ChromaDB a partir do PostgreSQL.

Este script:
1. Conecta ao banco de dados PostgreSQL (mgi_raspagem)
2. Extrai dados das tabelas específicas de cada empresa:
   - CEITEC: tbl_artigos_ceitec, tbl_paginas_ceitec + notícias relacionadas
   - IMBEL: tbl_artigos_imbel, tbl_paginas_imbel + notícias relacionadas  
   - Telebras: tbl_artigos_telebras, tbl_paginas_telebras + notícias relacionadas
3. Apaga os bancos vetoriais antigos
4. Recria os bancos vetoriais com os novos dados

Uso:
    python rag_rebuild_from_postgres.py [--empresa EMPRESA]
    
    EMPRESA pode ser: CEITEC, IMBEL, Telebras ou ALL (padrão)
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Set
import json

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

from processing.semantic_chunker import E5SemanticChunker

# Carregar variáveis de ambiente
load_dotenv()

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Configurações do banco de dados PostgreSQL
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'mgi_raspagem',
    'user': 'leocamilo',
    'password': 'leocamilo'
}

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

# Mapeamento de termos de busca para empresas
# Termos específicos de cada empresa
TERMOS_EMPRESA = {
    'CEITEC': {
        'exclusivos': ['ceitec'],  # IDs dos termos exclusivos
        'compartilhados': ['semicondutores', 'semiconductors']  # Termos compartilhados
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

# Termos que vão para TODAS as empresas (contexto geral)
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
# FUNÇÕES DE PROCESSAMENTO DE DOCUMENTOS
# ============================================================================

def criar_documento_noticia(noticia: Dict, empresa: str) -> Document:
    """Cria um Document LangChain a partir de uma notícia."""
    # Combinar título, descrição e conteúdo
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
    
    # Obter IDs dos termos exclusivos e compartilhados
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

def processar_documentos_com_chunking(documentos: List[Document], empresa: str) -> List[Document]:
    """
    Processa os documentos com chunking semântico.
    """
    print(f"\nProcessando {len(documentos)} documentos com chunking semântico...")
    
    chunker = E5SemanticChunker(
        model_name=EMBEDDING_MODEL,
        similarity_threshold=0.7,
        max_tokens_per_chunk=1000,
        min_tokens_per_chunk=100,
        print_logging=False  # Desabilitar logs individuais para menos poluição
    )
    
    processed_chunks = []
    
    for i, doc in enumerate(documentos):
        if (i + 1) % 100 == 0:
            print(f"  Processando documento {i+1}/{len(documentos)}...")
        
        try:
            # Aplicar chunking semântico
            chunks_text = chunker.chunk_text(doc.page_content)
            
            # Criar documentos para cada chunk
            for j, chunk_text in enumerate(chunks_text):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        'chunk_id': f"{doc.metadata.get('source', 'doc')}_{j}",
                        'chunk_idx': j,
                        'total_chunks': len(chunks_text),
                        'processed_date': datetime.now().isoformat()
                    }
                )
                processed_chunks.append(chunk_doc)
        except Exception as e:
            print(f"  Aviso: Erro ao processar documento {i}: {e}")
            # Em caso de erro, adicionar documento sem chunking
            doc.metadata['processed_date'] = datetime.now().isoformat()
            doc.metadata['chunk_id'] = f"{doc.metadata.get('source', 'doc')}_0"
            doc.metadata['chunk_idx'] = 0
            doc.metadata['total_chunks'] = 1
            processed_chunks.append(doc)
    
    print(f"  Total de chunks gerados: {len(processed_chunks)}")
    return processed_chunks

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
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"Criando banco vetorial com {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir
    )
    
    # 4. Salvar registro detalhado dos documentos processados
    # Extrair IDs únicos de cada tipo de documento
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
    
    return vectorstore

def processar_empresa(empresa: str):
    """
    Processa uma empresa completa: coleta dados, aplica chunking e cria banco vetorial.
    """
    print(f"\n{'#'*70}")
    print(f"# PROCESSANDO: {empresa}")
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
        
        # Aplicar chunking semântico
        chunks = processar_documentos_com_chunking(documentos, empresa)
        
        # Reconstruir banco vetorial
        reconstruir_banco_vetorial(empresa, chunks)
        
    finally:
        conn.close()

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Reconstrói bancos vetoriais ChromaDB a partir do PostgreSQL'
    )
    parser.add_argument(
        '--empresa',
        type=str,
        default='ALL',
        choices=['CEITEC', 'IMBEL', 'Telebras', 'ALL'],
        help='Empresa a processar (padrão: ALL)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Apenas mostra o que seria feito, sem executar'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("RECONSTRUÇÃO DOS BANCOS VETORIAIS RAG")
    print(f"Data/Hora: {datetime.now().isoformat()}")
    print("="*70)
    
    # Testar conexão
    print("\nTestando conexão com PostgreSQL...")
    try:
        conn = get_db_connection()
        print("  Conexão estabelecida com sucesso!")
        
        # Mostrar estatísticas
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
    
    for empresa in empresas:
        processar_empresa(empresa)
    
    print("\n" + "="*70)
    print("RECONSTRUÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*70)

if __name__ == "__main__":
    main()
