#!/usr/bin/env python3
"""
Script para atualização incremental dos bancos vetoriais a partir do PostgreSQL.

Este script:
1. Carrega o registro de documentos já processados (processed_files_*.json)
2. Consulta o PostgreSQL para identificar novos documentos
3. Processa apenas os documentos novos usando chunking semântico
4. Adiciona os novos chunks ao banco vetorial existente
5. Atualiza o registro de documentos processados

Uso:
    python update_vectordb_from_postgres.py [--empresa EMPRESA]
    
    EMPRESA pode ser: CEITEC, IMBEL, Telebras ou ALL (padrão)
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Set
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

# Termos gerais (vão para todas as empresas)
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

# ============================================================================
# FUNÇÕES PARA BUSCAR DOCUMENTOS NOVOS
# ============================================================================

def get_novos_artigos(conn, tabela: str, ids_processados: Set[int]) -> List[Dict]:
    """Busca artigos que ainda não foram processados."""
    query = f"""
        SELECT id, titulo, ano, abstract, conteudo, autores, doi
        FROM {tabela}
        WHERE conteudo IS NOT NULL AND LENGTH(conteudo) > 50
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query)
        todos = cur.fetchall()
        # Filtrar apenas os novos
        novos = [a for a in todos if a['id'] not in ids_processados]
        return novos

def get_novas_paginas(conn, tabela: str, ids_processados: Set[int]) -> List[Dict]:
    """Busca páginas que ainda não foram processadas."""
    query = f"""
        SELECT id, content, link, resumo, dt_download
        FROM {tabela}
        WHERE content IS NOT NULL AND LENGTH(content) > 50
        AND (status IS NULL OR status != 'error')
    """
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query)
        todas = cur.fetchall()
        # Filtrar apenas as novas
        novas = [p for p in todas if p['id'] not in ids_processados]
        return novas

def get_novas_noticias(conn, termo_ids: Set[int], ids_processados: Set[int]) -> List[Dict]:
    """Busca notícias que ainda não foram processadas."""
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
        todas = cur.fetchall()
        # Filtrar apenas as novas
        novas = [n for n in todas if n['id'] not in ids_processados]
        return novas

# ============================================================================
# FUNÇÕES DE PROCESSAMENTO DE DOCUMENTOS
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
# FUNÇÕES DE REGISTRO
# ============================================================================

def carregar_registro(empresa: str) -> Dict:
    """Carrega o registro de documentos processados."""
    processed_file = PROCESSED_FILES_RECORDS[empresa]
    
    if os.path.exists(processed_file):
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                registro = json.load(f)
                # Verificar se é o novo formato
                if 'documentos_processados' in registro:
                    return registro
                else:
                    # Formato antigo, retornar estrutura vazia
                    print(f"  Formato antigo detectado, será reconstruído")
                    return None
        except json.JSONDecodeError:
            print(f"  Erro ao ler registro, será reconstruído")
            return None
    return None

def extrair_ids_processados(registro: Dict) -> Dict[str, Set[int]]:
    """Extrai os IDs já processados do registro."""
    if not registro or 'documentos_processados' not in registro:
        return {'artigos': set(), 'paginas': set(), 'noticias': set()}
    
    docs = registro['documentos_processados']
    return {
        'artigos': set(docs.get('artigos', [])),
        'paginas': set(docs.get('paginas', [])),
        'noticias': set(docs.get('noticias', []))
    }

def atualizar_registro(empresa: str, registro_atual: Dict, novos_chunks: List[Document]) -> Dict:
    """Atualiza o registro com os novos documentos processados."""
    
    # Se não existe registro, criar um novo
    if not registro_atual:
        registro_atual = {
            'empresa': empresa,
            'data_criacao': datetime.now().isoformat(),
            'data_atualizacao': datetime.now().isoformat(),
            'total_chunks': 0,
            'fonte': 'PostgreSQL (mgi_raspagem)',
            'estatisticas': {'artigos': 0, 'paginas': 0, 'noticias': 0},
            'documentos_processados': {'artigos': [], 'paginas': [], 'noticias': []}
        }
    
    # Extrair IDs dos novos chunks
    novos_artigos = set()
    novas_paginas = set()
    novas_noticias = set()
    
    for chunk in novos_chunks:
        doc_type = chunk.metadata.get('type', '')
        db_id = chunk.metadata.get('db_id')
        if db_id:
            if doc_type == 'artigo':
                novos_artigos.add(db_id)
            elif doc_type == 'pagina':
                novas_paginas.add(db_id)
            elif doc_type == 'noticia':
                novas_noticias.add(db_id)
    
    # Atualizar registro
    docs = registro_atual['documentos_processados']
    docs['artigos'] = sorted(list(set(docs.get('artigos', [])) | novos_artigos))
    docs['paginas'] = sorted(list(set(docs.get('paginas', [])) | novas_paginas))
    docs['noticias'] = sorted(list(set(docs.get('noticias', [])) | novas_noticias))
    
    registro_atual['documentos_processados'] = docs
    registro_atual['data_atualizacao'] = datetime.now().isoformat()
    registro_atual['total_chunks'] = registro_atual.get('total_chunks', 0) + len(novos_chunks)
    registro_atual['estatisticas'] = {
        'artigos': len(docs['artigos']),
        'paginas': len(docs['paginas']),
        'noticias': len(docs['noticias'])
    }
    
    return registro_atual

def salvar_registro(empresa: str, registro: Dict):
    """Salva o registro de documentos processados."""
    processed_file = PROCESSED_FILES_RECORDS[empresa]
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(registro, f, ensure_ascii=False, indent=2)
    print(f"  Registro salvo em {processed_file}")

# ============================================================================
# FUNÇÕES PRINCIPAIS
# ============================================================================

def processar_documentos_com_chunking(documentos: List[Document]) -> List[Document]:
    """Processa os documentos com chunking semântico."""
    if not documentos:
        return []
    
    print(f"  Processando {len(documentos)} documentos com chunking semântico...")
    
    chunker = E5SemanticChunker(
        model_name=EMBEDDING_MODEL,
        similarity_threshold=0.7,
        max_tokens_per_chunk=1000,
        min_tokens_per_chunk=100,
        print_logging=False
    )
    
    processed_chunks = []
    
    for i, doc in enumerate(documentos):
        if (i + 1) % 50 == 0:
            print(f"    Processando documento {i+1}/{len(documentos)}...")
        
        try:
            chunks_text = chunker.chunk_text(doc.page_content)
            
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
            print(f"    Aviso: Erro ao processar documento {i}: {e}")
            doc.metadata['processed_date'] = datetime.now().isoformat()
            doc.metadata['chunk_id'] = f"{doc.metadata.get('source', 'doc')}_0"
            doc.metadata['chunk_idx'] = 0
            doc.metadata['total_chunks'] = 1
            processed_chunks.append(doc)
    
    print(f"  Total de chunks gerados: {len(processed_chunks)}")
    return processed_chunks

def atualizar_empresa(empresa: str, dry_run: bool = False):
    """
    Atualiza o banco vetorial de uma empresa com novos documentos.
    """
    print(f"\n{'='*60}")
    print(f"ATUALIZANDO: {empresa}")
    print(f"{'='*60}")
    
    chroma_dir = CHROMA_DB_DIRS[empresa]
    
    # Verificar se o banco vetorial existe
    if not os.path.exists(chroma_dir):
        print(f"ERRO: Banco vetorial não existe em {chroma_dir}")
        print("Execute primeiro: python rag_rebuild_from_postgres.py --empresa " + empresa)
        return
    
    # Carregar registro existente
    print("\n1. Carregando registro de documentos processados...")
    registro = carregar_registro(empresa)
    ids_processados = extrair_ids_processados(registro)
    
    print(f"  Artigos já processados: {len(ids_processados['artigos'])}")
    print(f"  Páginas já processadas: {len(ids_processados['paginas'])}")
    print(f"  Notícias já processadas: {len(ids_processados['noticias'])}")
    
    # Conectar ao PostgreSQL
    print("\n2. Buscando novos documentos no PostgreSQL...")
    conn = get_db_connection()
    
    try:
        termos_dict = get_termos_busca(conn)
        
        # Buscar documentos novos
        tabela_artigos = f"tbl_artigos_{empresa.lower()}"
        tabela_paginas = f"tbl_paginas_{empresa.lower()}"
        
        novos_artigos = get_novos_artigos(conn, tabela_artigos, ids_processados['artigos'])
        novas_paginas = get_novas_paginas(conn, tabela_paginas, ids_processados['paginas'])
        
        # Buscar novas notícias
        config_termos = TERMOS_EMPRESA.get(empresa, {'exclusivos': [], 'compartilhados': []})
        todos_termos = config_termos['exclusivos'] + config_termos['compartilhados'] + TERMOS_GERAIS
        termo_ids = get_termo_ids_por_nome(termos_dict, todos_termos)
        novas_noticias = get_novas_noticias(conn, termo_ids, ids_processados['noticias'])
        
        print(f"  Novos artigos encontrados: {len(novos_artigos)}")
        print(f"  Novas páginas encontradas: {len(novas_paginas)}")
        print(f"  Novas notícias encontradas: {len(novas_noticias)}")
        
        total_novos = len(novos_artigos) + len(novas_paginas) + len(novas_noticias)
        
        if total_novos == 0:
            print("\n✓ Nenhum documento novo encontrado. Banco vetorial está atualizado!")
            return
        
        if dry_run:
            print(f"\n[DRY-RUN] Seriam processados {total_novos} novos documentos")
            return
        
        # Criar documentos LangChain
        print("\n3. Criando documentos...")
        documentos = []
        
        for artigo in novos_artigos:
            doc = criar_documento_artigo(artigo, empresa)
            if len(doc.page_content) > 50:
                documentos.append(doc)
        
        for pagina in novas_paginas:
            doc = criar_documento_pagina(pagina, empresa)
            if len(doc.page_content) > 50:
                documentos.append(doc)
        
        for noticia in novas_noticias:
            doc = criar_documento_noticia(noticia, empresa)
            if len(doc.page_content) > 100:
                documentos.append(doc)
        
        print(f"  Documentos válidos: {len(documentos)}")
        
        # Aplicar chunking
        print("\n4. Aplicando chunking semântico...")
        chunks = processar_documentos_com_chunking(documentos)
        
        if not chunks:
            print("  Nenhum chunk gerado")
            return
        
        # Adicionar ao banco vetorial existente
        print(f"\n5. Adicionando {len(chunks)} chunks ao banco vetorial...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Carregar banco existente e adicionar documentos
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings
        )
        
        vectorstore.add_documents(chunks)
        
        # Atualizar e salvar registro
        print("\n6. Atualizando registro...")
        registro_atualizado = atualizar_registro(empresa, registro, chunks)
        salvar_registro(empresa, registro_atualizado)
        
        print(f"\n✓ Atualização concluída!")
        print(f"  - Novos chunks adicionados: {len(chunks)}")
        print(f"  - Total de artigos: {registro_atualizado['estatisticas']['artigos']}")
        print(f"  - Total de páginas: {registro_atualizado['estatisticas']['paginas']}")
        print(f"  - Total de notícias: {registro_atualizado['estatisticas']['noticias']}")
        
    finally:
        conn.close()

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Atualiza bancos vetoriais com novos documentos do PostgreSQL'
    )
    parser.add_argument(
        '--empresa',
        type=str,
        default='ALL',
        choices=['CEITEC', 'IMBEL', 'Telebras', 'ALL'],
        help='Empresa a atualizar (padrão: ALL)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Apenas mostra o que seria feito, sem executar'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ATUALIZAÇÃO INCREMENTAL DOS BANCOS VETORIAIS")
    print(f"Data/Hora: {datetime.now().isoformat()}")
    print("="*70)
    
    # Testar conexão
    print("\nTestando conexão com PostgreSQL...")
    try:
        conn = get_db_connection()
        print("  Conexão estabelecida com sucesso!")
        conn.close()
    except Exception as e:
        print(f"  ERRO na conexão: {e}")
        sys.exit(1)
    
    # Processar empresas
    if args.empresa == 'ALL':
        empresas = ['CEITEC', 'IMBEL', 'Telebras']
    else:
        empresas = [args.empresa]
    
    for empresa in empresas:
        atualizar_empresa(empresa, args.dry_run)
    
    print("\n" + "="*70)
    print("ATUALIZAÇÃO CONCLUÍDA!")
    print("="*70)

if __name__ == "__main__":
    main()
