import json
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from knowledge_index import PersistentKnowledgeIndex
from knowledge_retrieval import split_markdown
from token_budget import KnowledgeLedger, TokenCounter
from test_pipeline import FakeEmbeddings


class HeadingChunks(unittest.TestCase):
    def test_only_first_three_heading_levels_are_boundaries(self):
        text='Preamble\n# A\nbody\n## B\nbody\n### C\nbody\n#### D\nbody\n##### E\nbody\n###### F\nbody\n'
        docs=split_markdown(text,'source.md')
        self.assertEqual(len(docs),4)
        self.assertEqual(''.join(d.page_content for d in docs),text)
        self.assertIn('###### F',docs[-1].page_content)
        self.assertEqual(docs[-1].metadata['start_line'],6)
        self.assertEqual(docs[-1].metadata['end_line'],13)

    def test_never_splits_fences_lists_blank_lines_or_long_text(self):
        text='# Example\n```python\n# Python comment\n## another comment\n### still code\n```\n'+('paragraph\n\n- item\n'*500)
        docs=split_markdown(text)
        self.assertEqual(len(docs),1)
        self.assertEqual(docs[0].page_content,text)

    def test_stable_identifiers_and_changed_source_hash(self):
        a=split_markdown('# Example\nx\n','a.md')[0]
        b=split_markdown('# Example\nx\n','a.md')[0]
        c=split_markdown('# Example\ny\n','a.md')[0]
        self.assertEqual(a.metadata,b.metadata)
        self.assertNotEqual(a.metadata['section_id'],c.metadata['section_id'])


class PersistentIndex(unittest.TestCase):
    def make(self,root,embedding=None,namespace='test',maximum=8000):
        wiki=root/'wiki';wiki.mkdir(exist_ok=True)
        return PersistentKnowledgeIndex(wiki,root/'index.sqlite',TokenCounter(),embedding,namespace,maximum)

    def test_reopen_does_not_reembed_and_file_changes_are_incremental(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);embedding=FakeEmbeddings();a=self.make(root,embedding)
            (a.directory/'a.md').write_text('# Sketch\naddGeometry\n## Pad\npad.Length\n')
            (a.directory/'b.md').write_text('# Revolution\nrevolve profile\n')
            stats=a.sync();self.assertEqual(stats['embedded_sections'],3)
            calls=embedding.document_calls
            b=self.make(root,embedding);self.assertEqual(b.sync()['changed_files'],0)
            self.assertEqual(embedding.document_calls,calls)
            self.assertEqual(b.stats()['sections'],3)
            (b.directory/'a.md').write_text('# Sketch\nnewGeometry\n')
            stats=b.sync();self.assertEqual(stats['changed_files'],1)
            self.assertEqual(stats['embedded_sections'],1)
            self.assertEqual(b.stats()['sections'],2)
            (b.directory/'b.md').unlink();self.assertEqual(b.sync()['deleted_files'],1)
            self.assertEqual(b.sources(),['a.md'])

    def test_model_namespace_never_reuses_other_model_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);emb=FakeEmbeddings();a=self.make(root,emb,'model-a')
            (a.directory/'a.md').write_text('# API\naddGeometry\n');a.sync()
            b=self.make(root,emb,'model-b');self.assertEqual(b.sync()['embedded_sections'],1)
            self.assertNotEqual(a.namespace,b.namespace)

    def test_exact_api_search_and_source_filter_work_without_embeddings(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.make(Path(d))
            (a.directory/'a.md').write_text('# Placement\nApp.Placement moves a solid.\n')
            (a.directory/'b.md').write_text('# Sketch\nSketcher.Constraint closes a profile.\n')
            a.sync()
            result=a.search('Sketcher.Constraint')
            self.assertEqual(result[0].metadata['source'],'b.md')
            self.assertEqual(a.search('Sketcher.Constraint',sources=['a.md']),[])
            self.assertEqual(a.section(result[0].metadata['section_id']).page_content,result[0].page_content)

    def test_long_heading_section_is_not_split_to_fit_embedding_model(self):
        with tempfile.TemporaryDirectory() as d:
            emb=FakeEmbeddings();a=self.make(Path(d),emb,maximum=30)
            text='# Long\n'+('Sketcher.Constraint\n'*100)
            (a.directory/'long.md').write_text(text)
            stats=a.sync()
            self.assertEqual(stats['oversized_embedding_sections'],1)
            self.assertEqual(emb.document_calls,0)
            docs=a.search('Sketcher.Constraint')
            self.assertEqual(len(docs),1);self.assertEqual(docs[0].page_content,text)

    def test_failed_update_keeps_last_complete_file_and_fts(self):
        class Broken(FakeEmbeddings):
            def embed_documents(self,texts):raise RuntimeError('provider down')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=self.make(root,FakeEmbeddings())
            (a.directory/'a.md').write_text('# Old\noldAPI\n');a.sync()
            (a.directory/'a.md').write_text('# New\nnewAPI\n')
            b=self.make(root,Broken())
            with self.assertRaises(RuntimeError):b.sync()
            self.assertTrue(b.search('oldAPI'));self.assertFalse(b.search('newAPI',sources=[]))
            with sqlite3.connect(b.database) as db:
                self.assertEqual(db.execute('SELECT body FROM sections').fetchone()[0],'# Old\noldAPI\n')

    def test_indexes_markdown_absent_from_catalog_and_nested_files(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.make(Path(d));(a.directory/'extra').mkdir()
            (a.directory/'extra'/'API.md').write_text('# Extra\nundocumentedInCatalogAPI\n')
            a.sync()
            self.assertEqual(a.search('undocumentedInCatalogAPI')[0].metadata['source'],'extra/API.md')


class Ledger(unittest.TestCase):
    def test_code_sent_once_and_repeated_read_costs_zero(self):
        counter=TokenCounter();doc=split_markdown('# Code\n```python\nuniqueAPI()\n```\n','a.md')[0]
        ledger=KnowledgeLedger(counter,500)
        self.assertEqual(ledger.add(doc)[0],'loaded');cost=ledger.used
        self.assertEqual(ledger.add(doc),('already_loaded',0));self.assertEqual(ledger.used,cost)
        payload=json.dumps(ledger.payload())
        self.assertEqual(payload.count('uniqueAPI()'),1)
        self.assertNotIn('"code":',payload)

    def test_oversized_section_is_rejected_whole(self):
        doc=split_markdown('# Code\n'+('x=1\n'*100),'a.md')[0]
        ledger=KnowledgeLedger(TokenCounter(),100)
        self.assertEqual(ledger.add(doc)[0],'too_large');self.assertEqual(ledger.payload(),[])

    def test_missing_tokenizer_never_initiates_a_download(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'TIKTOKEN_CACHE_DIR':d}):
            counter=TokenCounter()
            self.assertEqual(counter.mode,'utf8_bytes_upper_bound')
            self.assertEqual(counter.count('سلام'),len('سلام'.encode()))

    def test_exact_encoding_can_be_supplied(self):
        class Encoder:
            name='test'
            def encode(self,text,**kwargs):return text.split()
        counter=TokenCounter(Encoder())
        self.assertEqual(counter.count('two words'),2)
        self.assertEqual(counter.mode,'tiktoken:test')

    def test_corrupt_tokenizer_cache_does_not_trigger_redownload(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'TIKTOKEN_CACHE_DIR':d,'CAD_TOKEN_ENCODING':'cl100k_base'}):
            url='https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken'
            (Path(d)/hashlib.sha1(url.encode()).hexdigest()).write_text('corrupt')
            with patch('tiktoken.get_encoding') as loader:
                self.assertEqual(TokenCounter().mode,'utf8_bytes_upper_bound')
                loader.assert_not_called()
