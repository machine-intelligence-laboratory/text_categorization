import logging
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture(scope="module")
def bpe_models():
    default_bpe = MagicMock()
    default_bpe.encode = Mock(side_effect=lambda tokens, outputs: tokens.split())
    return default_bpe


logging.basicConfig(level=logging.DEBUG)
