"""Offline token budgets. Use cached tiktoken; otherwise a UTF-8 byte upper bound.

No tokenizer download is performed implicitly. For cl100k/o200k byte BPE, the
number of UTF-8 bytes is a conservative upper bound on text tokens, not an
estimate of provider billing. Chat protocol overhead is reserved separately.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


class TokenCounter:
    def __init__(self, encoding=None):
        self.encoding = encoding
        self.mode = 'utf8_bytes_upper_bound'
        name = os.environ.get('CAD_TOKEN_ENCODING', 'cl100k_base')
        if encoding is None and name in ('cl100k_base', 'o200k_base'):
            url = f'https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken'
            cache = Path(os.environ.get('TIKTOKEN_CACHE_DIR', os.environ.get(
                'DATA_GYM_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'data-gym-cache'))))
            # tiktoken downloads assets when they are absent. Keep imports/runs offline.
            asset = cache / hashlib.sha1(url.encode()).hexdigest()
            expected = {
                'cl100k_base': '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7',
                'o200k_base': '446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d',
            }
            if asset.is_file() and hashlib.sha256(asset.read_bytes()).hexdigest() == expected[name]:
                import tiktoken
                self.encoding = tiktoken.get_encoding(name)
        if self.encoding is not None:
            self.mode = 'tiktoken:' + getattr(self.encoding, 'name', 'custom')

    def count(self, text):
        if not isinstance(text, str):
            text = compact_json(text)
        if self.encoding is not None:
            return len(self.encoding.encode(text, disallowed_special=()))
        return len(text.encode('utf-8'))


class BudgetExceeded(RuntimeError):
    pass


class KnowledgeLedger:
    """One copy of each whole section; a bounded set of source material per run."""
    def __init__(self, counter, limit):
        self.counter = counter
        self.limit = limit
        self.sections = {}
        self.content_hashes = set()
        self.used = 0

    def add(self, doc, allowance=None):
        section = {
            'section_id': doc.metadata['section_id'],
            'file_name': doc.metadata['source'],
            'heading': doc.metadata['heading'],
            'source_hash': doc.metadata['source_hash'],
            'text': doc.page_content,
        }
        key = section['section_id']
        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        if key in self.sections or content_hash in self.content_hashes:
            return 'already_loaded', 0
        cost = self.counter.count(section) + 8
        if self.used + cost > self.limit or (allowance is not None and cost > allowance):
            return 'too_large', cost
        self.sections[key] = section
        self.content_hashes.add(content_hash)
        self.used += cost
        return 'loaded', cost

    def payload(self):
        return list(self.sections.values())

    def sources(self):
        return sorted({section['file_name'] for section in self.sections.values()})
