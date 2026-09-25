import pytest

from etl.paths import DEFAULT
from etl.tokens import TokenCounter, pinned_counter


@pytest.mark.skipif(not DEFAULT.tokenizer_json.exists(), reason="pinned tokenizer.json not downloaded")
def test_pinned_tokenizer_loads_and_counts():
    counter = pinned_counter(DEFAULT.tokenizer_json)
    counts = counter.count(["int main(void) { return 0; }", ""])
    assert counts[0] > 5 and counts[1] == 0


def test_sha_mismatch_rejected(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="sha256"):
        TokenCounter(path, "0" * 64)
