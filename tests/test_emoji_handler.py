import unittest
from src.features.emoji_handler import EmojiHandler

class TestEmojiHandler(unittest.TestCase):

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            EmojiHandler(strategy="invalid_strat")

    def test_preserve_strategy_with_emoticon(self):
        handler = EmojiHandler(strategy='preserve')
        text = "Hello world :-) ❤️"
        # The :-) should become 🙂, ❤️ remains
        expected = "Hello world 🙂 ❤️"
        self.assertEqual(handler.handle(text), expected)

    def test_remove_strategy(self):
        handler = EmojiHandler(strategy='remove')
        text = "This is bad 😡 :("
        # The :( becomes ☹️, both emojis removed
        expected = "This is bad  " # Note: Space left behind is fine, Normalizer handles it later
        self.assertEqual(handler.handle(text), expected)

    def test_replace_descriptive_strategy(self):
        handler = EmojiHandler(strategy='replace_descriptive')
        text = "LMAO 😂"
        # 😂 is face_with_tears_of_joy
        expected = "LMAO <emoji_face_with_tears_of_joy>"
        self.assertEqual(handler.handle(text), expected)

    def test_replace_sentiment_strategy(self):
        handler = EmojiHandler(strategy='replace_sentiment')
        text = "I love this ❤️ and hate this 😡 :D"
        # ❤️ -> love, 😡 -> anger, :D -> 😃 -> laughter
        expected = "I love this <emoji_love> and hate this <emoji_anger> <emoji_laughter>"
        self.assertEqual(handler.handle(text), expected)

    def test_mixed_emoji_sequence(self):
        handler = EmojiHandler(strategy='replace_sentiment')
        # Multiple emojis in a row
        text = "Omg 😱😭👍"
        expected = "Omg <emoji_surprise> <emoji_sadness> <emoji_positive>"
        self.assertEqual(handler.handle(text), expected)

    def test_dataset_statistics(self):
        handler = EmojiHandler(strategy='preserve')
        dataset = [
            "Nice pic 😊",
            "Wow 😮",
            "I am so angry 🤬 :(",
            "No emojis here."
        ]
        stats = handler.get_statistics(dataset)
        
        self.assertEqual(stats["total_emojis_found"], 4)
        self.assertEqual(stats["classes"]["positive"], 1) # 😊
        self.assertEqual(stats["classes"]["surprise"], 1) # 😮
        self.assertEqual(stats["classes"]["anger"], 1)    # 🤬
        self.assertEqual(stats["classes"]["negative"], 1) # :( -> ☹️
        self.assertEqual(stats["classes"]["love"], 0)

if __name__ == '__main__':
    unittest.main()
