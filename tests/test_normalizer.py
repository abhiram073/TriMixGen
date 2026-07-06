import unittest
from src.features.normalizer import TextNormalizer

class TestTextNormalizer(unittest.TestCase):
    
    def setUp(self):
        # Default fully-enabled normalizer
        self.normalizer = TextNormalizer()

    def test_whitespace_normalization(self):
        text = "  This   has  way \n too \t much   whitespace.  "
        expected = "this has way too much whitespace."
        self.assertEqual(self.normalizer.normalize(text), expected)

    def test_case_normalization(self):
        # Should lowercase English but leave Telugu intact
        text = "Hello WORLD! ఇది బాగుంది."
        expected = "hello world! ఇది బాగుంది."
        self.assertEqual(self.normalizer.normalize(text), expected)

    def test_repeated_chars(self):
        # Testing max threshold of 2
        text = "I am soooo happppy to seeee youuuu"
        expected = "i am soo happy to see youu"
        self.assertEqual(self.normalizer.normalize(text), expected)

    def test_repeated_punctuation(self):
        # Preserves sentence boundaries but reduces spam
        text = "Wait... what??? NO WAY!!!"
        expected = "wait. what? no way!"
        self.assertEqual(self.normalizer.normalize(text), expected)

    def test_url_handling(self):
        text = "Check out my video http://youtube.com/watch and www.google.com"
        expected = "check out my video <URL> and <URL>"
        self.assertEqual(self.normalizer.normalize(text), expected)

    def test_mention_handling(self):
        text = "Hey @super_user123, did you see what @friend said?"
        expected = "hey <USER>, did you see what <USER> said?"
        self.assertEqual(self.normalizer.normalize(text), expected)

    def test_hashtag_handling(self):
        text = "This is trending #TriMixGen #NLP"
        expected = "this is trending trimixgen nlp"
        self.assertEqual(self.normalizer.normalize(text), expected)

    def test_html_cleanup(self):
        text = "Me &amp; you &lt;3"
        expected = "me & you <3"
        self.assertEqual(self.normalizer.normalize(text), expected)

    def test_unicode_normalization(self):
        # \u0065 is 'e', \u0301 is combining acute accent. NFC composes them to \u00e9
        decomposed = "r\u0065\u0301sum\u0065\u0301" 
        normalized = self.normalizer.normalize(decomposed)
        self.assertEqual(normalized, "résumé")
        
    def test_disable_features(self):
        # Test configurability
        custom_norm = TextNormalizer(
            lowercase_latin=False,
            url_placeholder=None,
            remove_hashtag_symbol=False,
            max_repeated_chars=0 # Disable
        )
        text = "HEY @user, #Cool http://link.com soooo!!!"
        # Mentions are replaced by default, URLs left intact, hashtags left, case left, repeated chars left, punct reduced
        expected = "HEY <USER>, #Cool http://link.com soooo!"
        self.assertEqual(custom_norm.normalize(text), expected)

if __name__ == '__main__':
    unittest.main()
