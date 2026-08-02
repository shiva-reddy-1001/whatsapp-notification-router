import unittest

from code.history_media import enrich_historical_media
from code.models import HistoryItem, Message


class FakeDataset:
    def __init__(self):
        message = Message("h1", "u1", "group", "g1", None, "s1", "2026-01-01",
                          "caption", "image", "i1", 0)
        self.history = [HistoryItem(message)]

    def media_path(self, _message):
        return None


class FakeProcessor:
    def extract(self, media_type, _path):
        return ("type=event poster; facts=deadline tomorrow", 0.9)


class HistoryMediaTests(unittest.TestCase):
    def test_extraction_is_attached_to_retrievable_history_text(self):
        dataset = FakeDataset()
        self.assertEqual(1, enrich_historical_media(dataset, FakeProcessor()))
        text = dataset.history[0].message.message_text
        self.assertIn("caption", text)
        self.assertIn("deadline tomorrow", text)


if __name__ == "__main__":
    unittest.main()
