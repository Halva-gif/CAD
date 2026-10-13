"""Retrieval helpers for the original vector-RAG pipeline; no CAD code templates."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def split_markdown(text, source=''):
    """All headings, with fenced examples and surrounding prose preserved."""
    sections=[]; lines=[]; fence=None; heading=''
    for line in text.splitlines(keepends=True):
        stripped=line.lstrip()
        if fence is None and '{{Code|' in stripped and 'code=' in stripped:
            fence='wiki'
        if fence is None and '<syntaxhighlight' in stripped:
            fence='html'
        if fence=='wiki' and (stripped.startswith('}}') or ('{{Code|' in stripped and stripped.rstrip().endswith('}}'))):
            fence=None
        if fence=='html' and '</syntaxhighlight>' in stripped:
            fence=None
        match=re.match(r'(`{3,}|~{3,})',stripped)
        if match and fence not in ('wiki','html'):
            mark=match.group(1)[0]
            fence=mark if fence is None else None if fence==mark else fence
        if fence is None and re.match(r'^#{1,6} ',stripped):
            if lines: sections.append((heading,''.join(lines)))
            heading=stripped.strip(); lines=[line]
        else: lines.append(line)
    if lines: sections.append((heading,''.join(lines)))
    digest=hashlib.sha256(text.encode()).hexdigest()
    return [Document(page_content=body,metadata={'source':source,'heading':title,'source_hash':digest}) for title,body in sections]


def pack_sections(documents, limit):
    """Whole sections only. Never cut an example at a character boundary."""
    result=[]; seen=set(); used=0
    for doc in documents:
        key=hashlib.sha256(doc.page_content.encode()).hexdigest()
        if key in seen: continue
        seen.add(key)
        if used+len(doc.page_content)>limit: continue
        result.append(doc); used+=len(doc.page_content)
    return result


def fenced_code(text):
    blocks=[m.group(2).rstrip() for m in re.finditer(
        r'^[ \t]*(`{3,}|~{3,})(?:python3?|py)?[ \t]*\n(.*?)^[ \t]*\1[ \t]*$',text,re.M|re.S|re.I)]
    blocks.extend(m.group(1).rstrip() for m in re.finditer(r'\{\{Code\|code=(.*?)\}\}',text,re.S))
    blocks.extend(m.group(1).rstrip() for m in re.finditer(r'<syntaxhighlight\s+lang=["\']?python["\']?[^>]*>(.*?)</syntaxhighlight>',text,re.S|re.I))
    return '\n\n'.join(blocks)


class LexicalRetriever:
    """Explicit offline alternative; vector remains the application's default."""
    def __init__(self,docs,k=5): self.docs=docs; self.k=k
    def invoke(self,query):
        terms=set(re.findall(r'[\w]+',query.lower()))
        ranked=[]
        for index,doc in enumerate(self.docs):
            words=Counter(re.findall(r'[\w]+',doc.page_content.lower()))
            score=sum(min(words[t],3) for t in terms)
            ranked.append((score,index,doc))
        return [doc for score,index,doc in sorted(ranked,key=lambda x:(-x[0],x[1]))[:self.k]]


class CachedEmbeddings(Embeddings):
    """Persist provider embeddings by model/endpoint/content hash, without keys."""
    def __init__(self,wrapped,database,namespace):
        self.wrapped=wrapped; self.database=Path(database); self.namespace=namespace
        self.database.parent.mkdir(parents=True,exist_ok=True)
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    def _connect(self): return sqlite3.connect(self.database,timeout=30)
    def _key(self,text): return hashlib.sha256((self.namespace+'\0'+text).encode()).hexdigest()
    def embed_documents(self,texts):
        vectors=[None]*len(texts); missing={}
        with self._connect() as db:
            for i,text in enumerate(texts):
                row=db.execute('SELECT value FROM embeddings WHERE key=?',(self._key(text),)).fetchone()
                if row: vectors[i]=json.loads(row[0])
                else: missing.setdefault(text,[]).append(i)
            if missing:
                fresh=self.wrapped.embed_documents(list(missing))
                if len(fresh)!=len(missing): raise ValueError('embedding provider returned wrong vector count')
                for text,vector in zip(missing,fresh):
                    db.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?)',(self._key(text),json.dumps(vector)))
                    for i in missing[text]: vectors[i]=vector
        return vectors
    def embed_query(self,text):
        key=self._key('QUERY\0'+text)
        with self._connect() as db:
            row=db.execute('SELECT value FROM embeddings WHERE key=?',(key,)).fetchone()
            if row: return json.loads(row[0])
            vector=self.wrapped.embed_query(text)
            db.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?)',(key,json.dumps(vector)))
            return vector


def safe_wiki_path(directory, name):
    root=Path(directory).resolve()
    if Path(name).name != name or not name.endswith('.md'):
        raise ValueError('documentation filename must be a catalog basename ending in .md')
    path=(root/name).resolve()
    if not path.is_relative_to(root): raise ValueError('documentation path escapes knowledge directory')
    return path
