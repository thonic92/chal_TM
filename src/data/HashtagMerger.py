import spacy
from spacy.tokens import Token
from spacy.matcher import Matcher

class HashtagMerger(object):
    def __init__(self, nlp):
        
        Token.set_extension('is_hashtag', default=None)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add('HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        for span in spans:
            span.merge()
            for token in span:
                token._.is_hashtag = True
        return doc
