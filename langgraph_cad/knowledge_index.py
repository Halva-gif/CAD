"""Persistent, incremental hybrid search over the user's Markdown sources.

SQLite stores verbatim heading sections and vectors, with FTS5 for exact API
names. Vector search is exact cosine search (appropriate for a local wiki), not
an external vector server. Nothing in this module calls a chat model.
"""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
from langchain_core.documents import Document

from knowledge_retrieval import safe_wiki_path, split_markdown

SPLITTER_VERSION = 'markdown-h1-h2-h3-v2'


class PersistentKnowledgeIndex:
    def __init__(self, directory, database, counter, embeddings=None,
                 embedding_namespace='lexical', embedding_max_tokens=8000, batch_size=32):
        self.directory = Path(directory).resolve()
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.counter = counter
        self.embeddings = embeddings
        self.embedding_max_tokens = embedding_max_tokens
        self.batch_size = batch_size
        self.namespace = hashlib.sha256((str(self.directory) + '\0' +
            SPLITTER_VERSION + '\0' + embedding_namespace + '\0' +
            str(embedding_max_tokens) + '\0' + counter.mode).encode()).hexdigest()
        self._rows = None
        self._vectors = None
        self._vector_ids = []
        self.last_sync = {}
        with closing(self._connect()) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS files (
                    namespace TEXT, source TEXT, digest TEXT,
                    PRIMARY KEY(namespace, source));
                CREATE TABLE IF NOT EXISTS sections (
                    namespace TEXT, id TEXT, source TEXT, heading TEXT,
                    source_hash TEXT, body TEXT, tokens INTEGER,
                    start_line INTEGER, end_line INTEGER, vector BLOB,
                    PRIMARY KEY(namespace, id));
                CREATE INDEX IF NOT EXISTS sections_source ON sections(namespace, source);
                CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
                    namespace UNINDEXED, id UNINDEXED, source, heading, body);
            ''')

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def sync(self):
        """Scan all .md files; embed only new/changed sources. Commit per file."""
        if not self.directory.is_dir():
            raise FileNotFoundError(f'Knowledge directory does not exist: {self.directory}')
        stats = {'files': 0, 'changed_files': 0, 'deleted_files': 0,
                 'embedded_sections': 0, 'oversized_embedding_sections': 0}
        with closing(self._connect()) as db:
            previous = {row['source']: row['digest'] for row in db.execute(
                'SELECT source,digest FROM files WHERE namespace=?', (self.namespace,))}
            seen = set()
            for path in sorted(self.directory.rglob('*.md')):
                source = path.relative_to(self.directory).as_posix()
                # Also reject symlinks escaping the supplied knowledge root.
                safe_wiki_path(self.directory, source)
                content = path.read_text(encoding='utf-8')
                digest = hashlib.sha256(content.encode()).hexdigest()
                seen.add(source)
                stats['files'] += 1
                if previous.get(source) == digest:
                    continue
                docs = split_markdown(content, source)
                vectors = {}
                eligible = []
                if self.embeddings is not None:
                    for doc in docs:
                        if self.counter.count(doc.page_content) <= self.embedding_max_tokens:
                            eligible.append(doc)
                        else:
                            stats['oversized_embedding_sections'] += 1
                    for offset in range(0, len(eligible), self.batch_size):
                        batch = eligible[offset:offset + self.batch_size]
                        batch_vectors = self.embeddings.embed_documents([d.page_content for d in batch])
                        if len(batch_vectors) != len(batch):
                            raise ValueError('embedding provider returned wrong vector count')
                        for doc, vector in zip(batch, batch_vectors):
                            array = np.asarray(vector, dtype=np.float32)
                            if array.ndim != 1 or not array.size or not np.isfinite(array).all():
                                raise ValueError('embedding provider returned invalid vector')
                            vectors[doc.metadata['section_id']] = array.tobytes()
                    stats['embedded_sections'] += len(eligible)
                # Failed embedding leaves the previous complete file untouched.
                with db:
                    self._delete_source(db, source)
                    for doc in docs:
                        m = doc.metadata
                        db.execute('INSERT INTO sections VALUES (?,?,?,?,?,?,?,?,?,?)',
                            (self.namespace, m['section_id'], source, m['heading'], digest,
                             doc.page_content, self.counter.count(doc.page_content),
                             m['start_line'], m['end_line'], vectors.get(m['section_id'])))
                        db.execute('INSERT INTO sections_fts VALUES (?,?,?,?,?)',
                            (self.namespace, m['section_id'], source, m['heading'], doc.page_content))
                    db.execute('INSERT INTO files VALUES (?,?,?)', (self.namespace, source, digest))
                stats['changed_files'] += 1
            for source in previous.keys() - seen:
                with db:
                    self._delete_source(db, source)
                stats['deleted_files'] += 1
        self._rows = None
        self._vectors = None
        self.last_sync = stats
        return stats

    def _delete_source(self, db, source):
        for table in ('sections', 'sections_fts', 'files'):
            db.execute(f'DELETE FROM {table} WHERE namespace=? AND source=?',
                       (self.namespace, source))

    def _load(self):
        if self._rows is not None:
            return
        with closing(self._connect()) as db:
            rows = db.execute('SELECT * FROM sections WHERE namespace=? ORDER BY source,start_line',
                              (self.namespace,)).fetchall()
        self._rows = {row['id']: dict(row) for row in rows}
        self._vector_ids = [row['id'] for row in rows if row['vector'] is not None]
        if self._vector_ids:
            arrays = [np.frombuffer(self._rows[key]['vector'], dtype=np.float32)
                      for key in self._vector_ids]
            if len({array.size for array in arrays}) != 1:
                raise ValueError('embedding dimensions changed; choose a new embedding model namespace')
            self._vectors = np.stack(arrays)
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            self._vectors = self._vectors / np.maximum(norms, 1e-12)

    def sources(self):
        with closing(self._connect()) as db:
            return [row['source'] for row in db.execute(
                'SELECT source FROM files WHERE namespace=? ORDER BY source', (self.namespace,))]

    @staticmethod
    def _document(row):
        return Document(page_content=row['body'], metadata={
            'source': row['source'], 'heading': row['heading'],
            'section_id': row['id'], 'source_hash': row['source_hash'],
            'tokens': row['tokens'], 'start_line': row['start_line'],
            'end_line': row['end_line']})

    def section(self, section_id, source=None):
        self._load()
        row = self._rows.get(section_id)
        if row is None or (source is not None and row['source'] != source):
            return None
        return self._document(row)

    def search(self, query, k=8, sources=None):
        """Reciprocal-rank fusion of exact API/word matching and vector similarity."""
        self._load()
        allowed = set(sources) if sources is not None else None
        pool = {key for key, row in self._rows.items()
                if allowed is None or row['source'] in allowed}
        if not pool:
            return []
        candidate_limit = max(32, k * 4)
        # API names and snake_case also contribute individual words.
        terms = list(dict.fromkeys(re.findall(r'[^\W_]+', query.lower(), re.UNICODE)))[:32]
        lexical = []
        if terms:
            expression = ' OR '.join('"' + term + '"' for term in terms)
            with closing(self._connect()) as db:
                sql = ('SELECT id FROM sections_fts WHERE sections_fts MATCH ? '
                       'AND namespace=?')
                args = [expression, self.namespace]
                if allowed is not None:
                    sql += ' AND source IN (' + ','.join('?' for _ in allowed) + ')'
                    args.extend(sorted(allowed))
                sql += ' ORDER BY bm25(sections_fts,0,0,2,3,1) LIMIT ?'
                args.append(candidate_limit)
                lexical = [row['id'] for row in db.execute(sql, args)]
        semantic = []
        if self.embeddings is not None and self._vectors is not None:
            vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
            if vector.shape != (self._vectors.shape[1],) or not np.isfinite(vector).all():
                raise ValueError('query embedding dimension/values do not match stored index')
            similarities = self._vectors @ (vector / max(float(np.linalg.norm(vector)), 1e-12))
            candidates = [i for i, key in enumerate(self._vector_ids) if key in pool]
            candidates.sort(key=lambda i: (-float(similarities[i]), self._vector_ids[i]))
            semantic = [self._vector_ids[i] for i in candidates[:candidate_limit]]
        scores = {}
        for ranking in (lexical, semantic):
            for rank, key in enumerate(ranking, 1):
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(scores, key=lambda key: (-scores[key], key))[:k]
        return [self._document(self._rows[key]) for key in ordered]

    def stats(self):
        self._load()
        return {**self.last_sync, 'sections': len(self._rows),
                'vector_sections': len(self._vector_ids), 'database': str(self.database),
                'token_counter': self.counter.mode}
