# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This module provides the primary functions of the Citron quote extraction 
and attribution system and a web server supporting a REST API.
"""

from .data import Quote
from .cue import CueClassifier
from .content import ContentClassifier
from .content import ContentResolver
from .source import SourceResolver
from .source import SourceClassifier
from .coreference import CoreferenceResolver
from . import utils
from . import metrics
from .logger import logger

APPLICATION_NAME = "citron-extractor"
DESIRED_LABELS = {"GPE", "PERSON", "NORP", "ORG"}

class Citron():
    """
    Class providing methods to extract quotes from documents.
    """

    def __init__(self, model_path, nlp=None):
        """
        Constructor.

        Args:
            model_path: The path (string) to the Citron model directory.
            nlp: A spaCy Language object, or None.
        """
        if nlp is None:
            self.nlp = utils.get_parser()
        else:
            self.nlp = nlp

        logger.info("Loading Citron model: %s", model_path)
        self.cue_classifier = CueClassifier(model_path)
        self.content_classifier = ContentClassifier(model_path)
        self.source_classifier = SourceClassifier(model_path)
        self.content_resolver = ContentResolver(model_path)
        self.source_resolver = SourceResolver(model_path)
        self.coreference_resolver = CoreferenceResolver(model_path)

        self.source = {
            "application": APPLICATION_NAME,
            "model:": self.cue_classifier.model["timestamp"] 
        }


    def extract(self, text, resolve_coreferences=True):
        """
        Extract quotes from the supplied text.
        
        Args:
            text: The text (string)
            resolve_coreferences: A boolean flag indicating whether to resolve coreferences.
            
        Returns:
            A JSON serialisable object containing the extracted quotes.
        """
        
        doc = self.nlp(text)
        
        quotes = self.get_quotes(doc, resolve_coreferences)
        quotes_json = []
        
        for quote in quotes:
            quotes_json.append(quote.to_json())

        entities = self.get_entities(doc)
        
        return { 
            "quotes": quotes_json, 
            "source": self.source,
            "entities": entities,
         }
    
    def get_entities(self, doc):
        seen = {}
        results = []
        for ee in doc.ents:
            entityName = ee.text.strip()
            if ee.label_ in DESIRED_LABELS and entityName not in seen:
                should_add = True
                for text in seen.keys():
                    # If this is a substring, skip it
                    if entityName in text:
                        should_add = False
                        break
                seen[entityName] = True

                if should_add:
                    result_entity = { 
                            "Label": ee.label_,
                            "Text": entityName,
                            "Start": ee.start,
                            }   
                    results.append(result_entity)

        return results
    
    def get_quotes(self, doc, resolve_coreferences=True):
        """
        Extract quotes from a spaCy Doc.
        
        Args:
            doc: A spaCy Doc object.
            resolve_coreferences: A boolean flag indicating whether to resolve coreferences.
        
        Returns:
            A list of citron.data.Quote objects.
        """
        # First find quote-cues.
        cue_spans, cue_labels = self.cue_classifier.predict_cues_and_labels(doc)
        
        if len(cue_spans) == 0:
            return []
        
        # Identify source and content spans.
        content_spans, content_labels = self.content_classifier.predict_contents_and_labels(doc, cue_labels)
        source_spans = self.source_classifier.predict_sources_and_labels(doc, cue_labels, content_labels)[0]
        
        if len(content_spans) == 0:
            return []
        
        if len(source_spans) == 0:
            return []
        
        # Identify the quote-cue associated with each source and content span.
        cue_to_contents_map = self.content_resolver.resolve_contents(content_spans, cue_spans)
        sentence_section_labels =  utils.get_sentence_section_labels(doc)
        cue_to_sources_map  = self.source_resolver.resolve_sources(source_spans, cue_spans, sentence_section_labels)
        
        # Join source and content spans which share the same quote-cue.
        quotes = []
        for cue in cue_spans:
            key = (cue.start, cue.end)
            
            if key in cue_to_contents_map and key in cue_to_sources_map:
                contents = cue_to_contents_map[key]
                source = cue_to_sources_map[key]
                confidence = source._.probability
                
                for content in contents:
                    confidence = confidence * content._.probability
                
                quote = Quote(cue, [source], contents, confidence=confidence)
                quotes.append(quote)
        
        if resolve_coreferences and len(quotes) > 0:
            self.coreference_resolver.resolve_document(doc, quotes, source_spans, content_spans, content_labels)
        
        return quotes
    
    
    def evaluate(self, test_path):
        """
        Evaluate Citron using the supplied test data.
        
        Args:
            test_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
        """
        
        metrics.evaluate(self, test_path)