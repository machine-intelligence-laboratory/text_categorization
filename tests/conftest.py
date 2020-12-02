import logging

import pytest


class BpeStub:
    def encode(self, tokens, output_type):
        return tokens.split()


@pytest.fixture(scope="module")
def bpe_models():
    default_bpe = BpeStub()
    langs = ["en", "es", "nl", "fr", "gf", "rq"]
    return {lang: default_bpe for lang in langs}


logging.basicConfig(level=logging.DEBUG)
