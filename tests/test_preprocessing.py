"""
Basic tests for the preprocessing and recommendation pipeline.
"""
import re
import unicodedata
import pytest


# ---- Replicate the normalize_text function from the notebook ----
def normalize_text(s):
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strip_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_ampersand_replacement(self):
        assert normalize_text("Tom & Jerry") == "tom and jerry"

    def test_special_characters_removed(self):
        assert normalize_text("hello! @world#") == "hello world"

    def test_accents_removed(self):
        assert normalize_text("Beyoncé") == "beyonce"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_input(self):
        assert normalize_text(None) == ""

    def test_multiple_spaces_collapsed(self):
        assert normalize_text("a   b   c") == "a b c"


class TestRecommendationLogic:
    """Test basic recommendation helper logic."""

    def test_popularity_excludes_observed(self):
        """Popular recommendations should not include songs already in the playlist."""
        popular_ranking = ["song_a", "song_b", "song_c", "song_d", "song_e"]
        observed = {"song_a", "song_c"}

        recs = []
        for s in popular_ranking:
            if s not in observed:
                recs.append(s)
            if len(recs) >= 3:
                break

        assert "song_a" not in recs
        assert "song_c" not in recs
        assert len(recs) == 3

    def test_cooccurrence_scoring(self):
        """Co-occurrence scores should reflect how often songs appear together."""
        from collections import defaultdict

        # Simulate: song_a appears in playlists with song_b (2 times) and song_c (1 time)
        playlist_song_sets = {
            "p1": {"song_a", "song_b"},
            "p2": {"song_a", "song_b", "song_c"},
            "p3": {"song_d"},
        }
        song_to_playlists = defaultdict(set)
        for pk, songs in playlist_song_sets.items():
            for s in songs:
                song_to_playlists[s].add(pk)

        observed = {"song_a"}
        scores = defaultdict(float)
        for s in observed:
            for pk in song_to_playlists.get(s, set()):
                for candidate in playlist_song_sets[pk]:
                    if candidate not in observed:
                        scores[candidate] += 1.0

        # song_b co-occurs with song_a in p1 and p2 -> score 2
        # song_c co-occurs with song_a in p2 -> score 1
        assert scores["song_b"] == 2.0
        assert scores["song_c"] == 1.0
        assert "song_d" not in scores

    def test_masked_test_case_creation(self):
        """Masking should produce valid observed/hidden splits."""
        import random
        random.seed(42)

        songs = ["s1", "s2", "s3", "s4", "s5"]
        min_input_size = 2
        n_hide = max(1, min(3, int(round(0.2 * len(songs)))))

        hidden = set(random.sample(songs, n_hide))
        observed = [s for s in songs if s not in hidden]

        assert len(observed) >= min_input_size
        assert len(hidden) >= 1
        assert set(observed).isdisjoint(hidden)
        assert set(observed) | hidden == set(songs)
