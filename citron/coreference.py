# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This module provides methods to identify coreferences for quote sources.

A "mention" in this code refers to either a quote source, a named entity or a pronoun
that has been mentioned in the text. They are represented by a spaCy span.

"""

from datetime import datetime
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from nltk import corpus

from .data import DataSource
from . import utils
from . import metrics
from .logger import logger
from . import gender

PREFIX_GENDERS = {
    "mr": "male",
    "sir": "male",
    "lord": "male",
    "duke": "male",
    "marquess": "male",
    "count": "male",
    "viscount": "male",
    "baron": "male",
    "prince": "male",
    "king": "male",
    "mrs": "female",
    "miss": "female",
    "ms": "female",
    "lady": "female",
    "duchess": "female",
    "marchioness": "female",
    "countess": "female",
    "viscountess": "male",
    "baroness": "female",
    "princess": "female",
    "queen": "female",
    "dr": "unknown",
    "doctor": "unknown",
    "judge": "unknown",
    "justice": "unknown",
    "prof": "unknown",
    "professor": "unknown",
    "rep": "unknown",
    "sen": "unknown",
    "official": "unknown",
    "aide-de-camp": "unknown",
    "alderman": "unknown",
    "ambassador": "unknown",
    "appraiser": "unknown",
    "attaché": "unknown",
    "bureaucrat": "unknown",
    "cabinet secretary": "unknown",
    "chief agricultural negotiator": "unknown",
    "chief of district": "unknown",
    "chief of local state administration": "unknown",
    "chief of protocol": "unknown",
    "city attorney": "unknown",
    "manager": "unknown",
    "managers": "unknown",
    "city remembrancer": "unknown",
    "municipal clerk": "unknown",
    "commissioner of official languages": "unknown",
    "commissioner of the republic": "unknown",
    "coordinating secretary": "unknown",
    "council architect": "unknown",
    "council ranger": "unknown",
    "county executive": "unknown",
    "county administrator": "unknown",
    "county surveyor": "unknown",
    "courtier": "unknown",
    "cultural attaché": "unknown",
    "department secretary": "unknown",
    "deputy mayor": "unknown",
    "deputy prime minister": "unknown",
    "director of communications": "unknown",
    "director of the u.s. government publishing office": "unknown",
    "drain commissioner": "unknown",
    "diak": "unknown",
    "fence viewer": "unknown",
    "first civil service commissioner": "unknown",
    "general register office": "unknown",
    "gold commissioner": "unknown",
    "government agent": "unknown",
    "governor": "unknown",
    "gov": "unknown",
    "guardian minister": "unknown",
    "hardship post": "unknown",
    "high bailiff": "unknown",
    "historiographer Royal": "unknown",
    "hofmeister": "unknown",
    "interim leader": "unknown",
    "keeper of the privy seal of scotland": "unknown",
    "keeper of the seals": "unknown",
    "king of arms": "unknown",
    "language commissioner": "unknown",
    "marshal of the sejm": "unknown",
    "mayor": "unknown",
    "member of congress": "unknown",
    "minister": "unknown",
    "min": "unknown",
    "ministerial diary secretary": "unknown",
    "municipal commissioner": "unknown",
    "obersthofmeister": "unknown",
    "official secretary to the governor": "unknown",
    "ombudsman": "unknown",
    "parliamentary state secretary": "unknown",
    "permanent representative": "unknown",
    "permanent secretary": "unknown",
    "portreeve": "unknown",
    "postal worker": "unknown",
    "postmaster": "unknown",
    "prefect": "unknown",
    "president of the council of ministers": "unknown",
    "president of the municipal chamber": "unknown",
    "press secretary": "unknown",
    "prosecutor": "unknown",
    "pursuivant": "unknown",
    "reading clerk": "unknown",
    "receiver general": "unknown",
    "recorder of deeds": "unknown",
    "royal secretary": "unknown",
    "secretary": "unknown",
    "sec": "unknown",
    "secretary of state": "unknown",
    "secretary of the government": "unknown",
    "sheriff": "unknown",
    "sovereign": "unknown",
    "speaker": "unknown",
    "spokesman": "male",
    "spokeswoman": "female",
    "spokesperson": "unknown",
    "state architect": "unknown",
    "sub-divisional magistrate": "unknown",
    "tax collector": "unknown",
    "supervisor": "unknown",
    "treasurer": "unknown",
    "undersecretary": "unknown",
    "wazira": "unknown",
    "whip": "unknown",
    "president": "unknown",
    "director": "unknown",
    "ceo": "unknown",
    "chief": "unknown",
    "attorney": "unknown",
    "district attorney": "unknown",
    "doctor": "unknown",
    "dr": "unknown",
    "general": "unknown",
    "gen": "unknown",
    "senator": "unknown",
    "sen": "unknown",
    "leader": "unknown",
    "representative": "unknown",
    "rep": "unknown",
    "director-general": "unknown",
    "chair": "unknown",
    "chairman": "unknown",
    "chairwoman": "unknown",
    "administrator": "unknown",
    "adm": "unknown",
    "chancellor": "unknown",
    "judge": "unknown",
    "commissioner": "unknown",
    "superintendent": "unknown",
    "secretary-general": "unknown",
    "cardinal": "unknown",
    "undersheriff": "unknown",
    "president-elect": "unknown",
    "lt": "unknown",
    "lieutenant": "unknown",
    "pastor": "unknown",
    "historian": "unknown",
    "coroner": "unknown",
    "sgt": "unknown",
    "sergeant": "unknown",
    "priest": "unknown",
    "justice": "unknown",
    "politician": "unknown",
    "mayor-elect": "unknown",
    "founder": "unknown",
    "adviser": "unknown",
    "advisor": "unknown",
    "biologist": "unknown",
    "qb": "unknown",
    "democrat": "unknown",
    "republican": "unknown",
    "bishop": "unknown",
    "principal": "unknown",
    "filmmaker": "unknown",
    "organizer": "unknown",
    "archbishop": "unknown",
    "col": "unknown",
    "colonel": "unknown",
    "historian": "unknown",
    "co-ceo": "unknown",
    "journalist": "unknown",
    "sheikh": "unknown",
    "officer": "unknown",
    "fire marshall": "unknown",
    "u.s. marshal": "unknown",
    "midshipman": "unknown",
    "comic": "unknown",
    "coroner": "unknown",
    "sister": "female",
    "brother": "male",
    "psychotherapist": "unknown",
    "wrangler": "unknown",
    "archaeologist": "unknown",
    "rescuer": "unknown",
    "trustee": "unknown",
    "adm": "unknown",
    "admiral": "unknown",
}

def get_gender(name, gender_resolver):
    """
    Get the gender of a name.
    
    Args:
        name: a spaCy Span object.
        gender_resolver: A citron.gender.ForenameGenderClassifier object.
    
    Returns:
        "male", "female", "neutral" or "unknown".
    """ 
    if is_pronoun(name):
        return get_pronoun_gender(name)
    elif utils.is_person(name):
        # References to earlier full names in the article are left as "unknown"
        # They are resolved to earlier matches.
        if len(name) > 1:
            prefix, rest_of_name = split_on_longest_prefix(name)
            if prefix is not None:
                gender = get_prefix_gender(prefix)
                if gender != "unknown":
                    return gender
            
            first_name = rest_of_name.split(" ")[0]
            return gender_resolver.get_forename_gender(first_name)
        else:
            return "unknown"
    elif utils.is_organisation(name):
        return "neutral"
    else:
        return "unknown"

PRONOUN_GENDERS = {
    "he": "male",
    "she": "female",
    "it": "neutral",
    "they": "neutral",
    "we": "neutral",
}

def get_pronoun_gender(pronoun):
    """
    Get the gender associated with a pronoun.
    
    Args:
        pronoun: a spaCy Span object.
    
    Returns:
        "male", "female", "neutral" or "unknown".
    """   
    text = pronoun.text.lower()
    if text in PRONOUN_GENDERS:
        return PRONOUN_GENDERS[text]
    return "unknown"

def split_on_longest_prefix(name):
    """
    Splits the input on the longest prefix and returns the prefix and name

    Args:
        name: a spaCy Span object.
    """
    lowered = name.text.lower()
    for key in PREFIX_GENDERS.keys():
        if lowered.startswith(key):
            prefix_removed = name.text[len(key):].strip()
            if prefix_removed.startswith("."):
                prefix_removed = prefix_removed[1:]
            return key, prefix_removed.strip()

    return None, name.text

def split_on_rightmost_prefix(name):
    """
    Splits the input on the rightmost prefix and returns the prefix and name

    Args:
        name: a spaCy Span object.
    """
    for end_idx in range(len(name), 0, -1):
        for start_idx in range(0, end_idx):
            span = name[start_idx:end_idx]
            lower_chunk = span.text.lower()
            if lower_chunk in PREFIX_GENDERS:
                prefix = name[:end_idx]
                suffix = name[end_idx:]
                return prefix, suffix
    return [], name

def get_prefix_gender(prefix):
    """
    Get the gender associated with a prefix.
    
    Args:
        prefix: a string representing the prefix.
    
    Returns:
        "male", "female", or "unknown"
    """
    if prefix.endswith("."):
        prefix = prefix[: -1]

    lowered = prefix.lower()
    
    if lowered in PREFIX_GENDERS:
        return PREFIX_GENDERS[lowered]
    return "unknown"


def is_proper_noun(span):
    """
    Check whether the span is a proper noun. 
    """
    for token in span:
        if token.pos_ != "PROPN":
            return False
    
    return True


def get_mention_type(mention):
    """
    Get the type of the mention. A mention may be a name or a pronoun.
    
    Args:
        mention: A spaCy Span object.
    
    Returns:
        The type (string) or "unknown".
    """
    
    if is_pronoun(mention):    
        return "pronominal"
    
    elif is_proper_noun(mention):
        return "proper"
    
    elif is_noun(mention):
        return "nominal"
    
    else:
        return "unknown"


def is_pronoun(span):
    """
    Check whether the span is a pronoun.
    
    Args:
        span: A spaCy Span object.
    
    Returns:
        A boolean value.
    """
    
    return len(span) == 1 and span[0].pos_ == "PRON"


def is_plural_pronoun(span):
    """
    Check whether the span is a plural pronoun.
    
    Args:
        span: A spaCy Span object.
    
    Returns:
        A boolean value.
    """
    
    text = span.text.lower()
    
    if text == "it":
        # Assume "it" is an organisation.
        return True
    
    elif text == "they": 
        return True
    
    elif text == "we": 
        return True
    
    else:
        return False


def is_noun(span):
    """
    Check whether the span is a noun.
    
    Args:
        span: A spaCy Span object.
    
    Returns:
        A boolean value.
    """
    
    return span[0].pos_ == "NOUN"

 
def is_plural(span):
    """
    Check whether the span is a plural mention.
    
    Args:
        span: A spaCy Span object.
    
    Returns:
        A boolean value.
    """
    
    if span[0].tag_ == "NNS":
        return True
    
    elif span[0].tag_ == "NNPS": 
        return True
     
    elif span[0].pos_ == "PRON" and span[0].text.lower() == "they":
        return True
    
    elif span[0].ent_iob_ != "O":
        return span[0].ent_type_ in ("ORG", "NORP")
    
    else:
        return False


class CoreferenceResolver():    
    """
    Class which identifies coreferences for quote sources.
    
    Approach:
      For each document:
        - Find all the person and organisation names mentioned in the document.
        - Create a table mapping each name to the earliest instance of its longest form.
        - Working from left to right:
          - resolve each pronoun back to a name.
    """
    
    MODEL_FILENAME = "coreference-resolver.pickle"
    PROBABILITY_THRESHOLD = 0.1
    PREVIOUS_N = 2
    
    
    def __init__(self, model_path):
        """
        Constructor.
        
        Args:
            model_path: The path (string) to the Citron model.
        """
        
        filename = os.path.join(model_path, self.MODEL_FILENAME)
        logger.debug("Loading Coreference Resolver model: %s", filename)
        
        with open(filename, "rb") as infile:
            self._model = pickle.load(infile)
    
    def resolve_document(self, doc, gender_resolver, quotes, sources, contents, content_labels):
        """
        Find the primary coreferences of the quote sources in a document.
        
        Args:
            doc: A spaCy Doc object.
            gender_resolver: A citron.gender.ForenameGenderClassifier object.
            quotes: A list of spaCy Span objects.
            sources: A list of spaCy Span objects.
            contents: A list of spaCy Span objects.
            content_labels: A list containing an IOB label for each token in the document.
        """
        coreference_table = CoreferenceTable(doc, gender_resolver, quotes, content_labels)

        logger.debug("Resolve document: %s", contents)
        
        for quote in quotes:
            self._resolve_quote(doc, gender_resolver, coreference_table, quote, sources, contents)
    
    
    def _resolve_quote(self, doc, gender_resolver, coreference_table, quote, sources, contents):
        """
        Find the primary coreferences of the sources of a quote.
        
        Args:
            doc: A spaCy Doc object.
            gender_resolver: A citron.gender.ForenameGenderClassifier object.
            coreference_table: A citron.coreference.CoreferenceTable object.
            quote: A citron.Data.Quote object (the quote to resolve).
            sources: A list of spaCy Span objects.
            contents: A list of spaCy Span objects.
        """
        coreferences = []
        logger.debug("Resolve quote: %s: %s", quote.contents, quote.sources)
        for source in quote.sources:
            coreference = self._resolve_coreference_chain(doc, gender_resolver, coreference_table, quote, source, sources, contents)
            
            if coreference is not None and coreference.text != source.text:
                coreferences.append(coreference)
            
        quote.coreferences = coreferences
    
    
    def _resolve_coreference_chain(self, doc, gender_resolver, coreference_table, quote, span, sources, contents):
        """
        Find the primary coreference of a span by recursively resolving a coreference chain.
        
        Args:
            doc: A spaCy Doc object.
            gender_resolver: A citron.gender.ForenameGenderClassifier object.
            coreference_table: A citron.coreference.CoreferenceTable object.
            quote: A citron.data.Quote object (the quote containing the span).
            span: A spaCy Span object (the span to resolve).
            sources: A list of spaCy Span objects.
            contents: A list of spaCy Span objects.
        """
        chain = []
        coreference = self._resolve_coreference(doc, gender_resolver, coreference_table, quote, span, chain, sources, contents)
        
        for mention in chain:
            if not coreference_table.contains(mention):
                coreference_table.add_entry(mention, coreference)
        
        return coreference
    
    
    def _resolve_coreference(self, doc, gender_resolver, coreference_table, quote, span, chain, sources, contents):
        """
        Find the coreference for a span.
        
        Args:
            doc: A spaCy Doc object.
            gender_resolver: A citron.gender.ForenameGenderClassifier object.
            coreference_table: A citron.coreference.CoreferenceTable object.
            quote: A citron.data.Quote object (the quote containing the span).
            span: A spaCy Span object (the span to resolve).
            chain: A list spaCy Span objects.
            sources: A list of spaCy Span objects.
            contents: A list of spaCy Span objects.
        
        Returns:
            A spaCy Span object.
            
        """   
        coreference = coreference_table.resolve(span)
        logger.debug("Resolve span: %s: %s", span, coreference)
        
        if coreference is not None:                
            return coreference
        
        elif is_pronoun(span):
            chain.append(span)
            coreference = self._resolve_pronoun(gender_resolver, coreference_table, quote, span, sources, contents)
            
            if coreference is None:
                return span      
            else:
                return self._resolve_coreference(doc, gender_resolver, coreference_table, quote, coreference, chain, sources, contents)
        
        else:
            return span
    
    
    def _resolve_pronoun(self, gender_resolver, coreference_table, quote, pronoun, sources, contents):
        """
        Find the coreference for a pronoun.
        
        Args:
            gender_resolver: A citron.gender.ForenameGenderClassifier object.
            coreference_table: A citron.coreference.CoreferenceTable object.
            quote: A citron.data.Quote object (the quote referencing the pronoun).
            pronoun: A spaCy Span object (the pronoun to resolve)
            sources: A list of spaCy Span objects.
            contents: A list of spaCy Span objects.
        
        Returns:
            predicted_coreference: A spaCy Span object, or None.
            
        """
        
        candidate_mentions = coreference_table.get_closest_preceding_mentions(pronoun, self.PREVIOUS_N, sources, contents, quote=quote)
        logger.debug("Candidate mentions: %s, %s", candidate_mentions, pronoun)
        
        if len(candidate_mentions) == 0:
            return None
        
        features = []
        
        for mention_index, mention in enumerate(candidate_mentions):
            candidate_features = self._get_features(gender_resolver, coreference_table, mention, mention_index, pronoun)
            features.append(candidate_features)

        logger.debug("Features: %s", features)

        test_vectors = self._model["vectorizer"].transform(features)
        predicted_probabilities = self._model["classifier"].predict_proba(test_vectors)
        predicted_index, probability = utils.get_index_of_max(predicted_probabilities)
        
        if probability < self.PROBABILITY_THRESHOLD:
            return None
        
        predicted_coreference = candidate_mentions[predicted_index]
        predicted_coreference._.probability = probability
        logger.debug("Predicted coreference: %s, %s", predicted_coreference, probability)
        return predicted_coreference
    
    
    def evaluate(self, gender_resolver, nlp, test_path):
        """
        Evaluate the Coreference Classifier by measuring the ability to identify 
        the correct coreference group for pronoun sources in each document.
        
        For a true positive the predicted coreference must be in the same coreference
        group as the pronoun. Ignore documents and quotes where coreference data is 
        not available in test data. 
        
        Args:
            nlp: A spaCy Language object.
            test_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
        """
        
        logger.info("Evaluating Coreference Resolver using: %s", test_path)
        document_count = 0
        pronoun_count = 0
        tp = 0
        fp = 0
        fn = 0
        
        for doc, quotes, coref_groups in DataSource(nlp, test_path):
            # Ignore test data which does not support coreference data.
            if coref_groups is None:
                continue
            
            sources   = utils.get_sources(quotes)
            contents  = utils.get_contents(quotes)
            content_labels = utils.get_iob_labels_for_spans(doc, contents)
            
            self.resolve_document(doc, gender_resolver, quotes, sources, contents, content_labels)
            document_count += 1
            
            for quote in quotes:
                for source in quote.sources:
                    if is_pronoun(source):
                        pronoun_count += 1
                        coref_group = self._get_coreference_group(source, coref_groups)
                        
                        if coref_group is None:
                            continue
                        
                        if len(quote.coreferences) == 0:
                            fn += 1
                        
                        else:
                            same_group = False
                            
                            for predicted_coref in quote.coreferences:                                
                                if self._get_label(predicted_coref, coref_group) == 1:
                                    same_group = True
                                    break
                            
                            if same_group:
                                tp += 1
                            else:
                                fp += 1
        
        print("Document count:", document_count)
        print("Pronoun count: ", pronoun_count) 
        print()
        print("--------  Metrics  --------")
        exact_scores = metrics.get_exact_scores(tp, fp, fn)
        metrics.print_metrics(*exact_scores)
    
    
    @staticmethod
    def build_model(nlp, gender_resolver, train_path, model_path):
        """
        Build and save a Coreference Resolver model.
        
        Args:
            nlp: A spaCy Language object.
            train_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
            model_path: The path (string) to the Citron model directory.          
        """
        
        logger.info("Building Coreference Resolver model using: %s", train_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        features, labels = CoreferenceResolver._get_features_and_labels(nlp, gender_resolver, train_path)
        
        logger.info("Vectorising training data")
        vectorizer = DictVectorizer()
        train_vectors = vectorizer.fit_transform(features)
        logger.debug("train_vectors.shape: %s", train_vectors.shape)
        
        logger.info("Training Coreference Resolver model.")
        classifier = LogisticRegression(solver="liblinear")
        classifier.fit(train_vectors, labels)
        
        model = {}
        model["classifier"] = classifier
        model["vectorizer"] = vectorizer
        model["forenamesTable"] = gender_resolver
        model["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        filename = os.path.join(model_path, CoreferenceResolver.MODEL_FILENAME)
        logger.info("Saving Coreference Resolver model: %s", filename)
        
        try:
            with open(filename, "wb") as outfile:
                pickle.dump(model, outfile)
        
        except IOError:
            logger.error("Unable to save Coreference Resolver model: %s", filename)
    
    
    @staticmethod
    def _get_features_and_labels(nlp, gender_resolver, input_path):
        """
        Get features and labels for all pronouns in the coreference groups of each document in a corpus.
        
        Args:
            input_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
        
        Returns:
            A tuple containing:
                features: A list of feature dicts.
                labels: A list of binary labels.
        """
        
        features = []
        labels = []
        document_count = 0
        
        for doc, quotes, coref_groups in DataSource(nlp, input_path):
            # Ignore test data which does not support coreference groups.
            if coref_groups is None:
                continue
            
            document_count += 1
            sources   = utils.get_sources(quotes)
            contents  = utils.get_contents(quotes)
            content_labels = utils.get_iob_labels_for_spans(doc, contents)
            coreference_table = CoreferenceTable(doc, gender_resolver, quotes, content_labels)
            
            for coref_group in coref_groups:
                for span in coref_group:
                    if is_pronoun(span):
                        pronoun = span
                        
                        # Ignore pronouns inside content spans.
                        if utils.is_inside_spans(pronoun.start, contents):
                            continue
                        
                        # Add features and labels for each candidate mention
                        candidate_mentions = coreference_table.get_closest_preceding_mentions(pronoun, 5, sources, contents)
                        
                        for mention_index, mention in enumerate(candidate_mentions):            
                            candidate_features = CoreferenceResolver._get_features(gender_resolver, coreference_table, mention, mention_index, pronoun)
                            
                            if candidate_features is not None:                
                                features.append(candidate_features)
                                label = CoreferenceResolver._get_label(candidate_mentions[mention_index], coref_group)
                                labels.append(label)
        
        logger.info("Document count: %s", document_count)
        return features, labels
    
    
    @staticmethod
    def _get_features(gender_resolver, coreference_table, mention, mention_index, pronoun):
        """
        Get the features of a mention (a name or pronoun) in relation to the specified pronoun.
        
        Args:
            gender_resolver: A citron.gender.ForenameGenderClassifier object.
            coreference_table: A CoreferenceTable object.
            mention: A spaCy Span object.
            mention_index: An index (int) representing the distance (in mentions) from the pronoun.
            pronoun: A spaCy Span object.
        
        Returns:
            A features dict.
        """
        
        mention_type = get_mention_type(mention)
        
        if mention_type == "pronominal":
            mention_type_text = mention.text
        else:
            mention_type_text = mention_type
        
        pronoun_is_plural = is_plural_pronoun(pronoun)
        mention_is_plural = mention._.is_plural
        mention_head = mention.root
        pronoun_gender = get_pronoun_gender(pronoun)
        
        if mention._.gender is None:
            mention._.gender = get_gender(mention, gender_resolver)
        
        mention_gender = mention._.gender
        
        # Add features
        features = {}
        
        # MENTION FEATURES      
        features["mention_type_text"]   = mention_type_text
        features["mention_entity_type"] = mention[0].ent_type_
        features["mention_pos"]        = mention[0].pos_
        
        # PRONOUN FEATURES
        features["pronoun_text"] = pronoun.text
        
        # PAIR FEATURES
        features["gender_match"] = pronoun_gender == mention_gender
        features["plurality_match"] = pronoun_is_plural == mention_is_plural
                
        # CONJUNCTIONS
        features["mention_distance_pronoun_type"] = str(mention_index) + pronoun.text + mention_type_text
        features["mention_head_pronoun_type"] = mention_head.text + pronoun.text + mention_type_text
        return features
    
    
    @staticmethod
    def _get_coreference_group(pronoun, coref_groups):
        """
        Get the coreference group containing the specified pronoun. 
        
        Args:
            pronoun: A spaCy Span object.
            coref_groups: A list of coreference groups. Each coreference group is a 
                list of spaCy Span objects.
        
        Returns:
            coref_group: A list of spaCy Span objects, or None.
        """
        
        for coref_group in coref_groups:
            for span in coref_group:
                if span.start == pronoun.start:
                    return coref_group
        
        return None
    
    
    @staticmethod
    def _get_label(span, coreference_group):
        """
        Get a binary label indicating whether a span is the coreference_group.
        
        Args:
            span: A spaCy Span object.
            coreference_group: A list of spaCy Span objects.
        
        Returns:
            Return 1 if the candidate_mention is in the group. Otherwise 0.
        """
        
        for other_span in coreference_group:
            if utils.are_overlapping_spans(span, other_span):
                return 1
        
        return 0


class CoreferenceTable():
    """
    Class which provides a list of all mentions (names and pronouns) found in 
    a document and a table which maps each mention with the earliest, longest 
    matching name found earlier in the text. The table is initially built 
    with names and then entries are added for pronouns as these are resolved.
    """
    
    def __init__(self, doc, gender_resolver, quotes=None, content_labels=None):
        """
        Constructor.
        
        Args:
            doc: A spaCy Doc object.
            gender_resolver: A citron.gender.ForenameGenderClassifier object.
            quotes: A list of citron.data.Quote objects.
            content_labels: A list containing an IOB label for each token in the document.
        """     
        self.doc = doc
        
        # Get all names in the document
        names = self.get_names(doc, gender_resolver, quotes, content_labels)
        logger.debug("Names: %s", names)
        
        # Dict which maps mentions to their primary (longest, earliest) coreference
        self.coreference_map = self._build_name_table(names)
        
        # Combine names and pronouns to create a list of mentions
        for token in doc:
            if content_labels is None or content_labels[token.i] == "O":
                if token.pos_ == "PRON":
                    pronoun_span = doc[token.i : token.i + 1]
                    pronoun_span._.gender = get_pronoun_gender(pronoun_span)
                    pronoun_span._.is_plural = is_plural_pronoun(pronoun_span)
                    names.append(pronoun_span)

        filtered = [split_on_rightmost_prefix(name)[1] for name in names if not is_pronoun(name)]
        logger.debug("Filtered: %s", filtered)
        
        # A sorted list of all names and pronouns in the document
        self.mentions = sorted(filtered, key=lambda x: x.start)
    
    
    def add_entry(self, mention, root_mention):
        """
        Add an entry to the table, associating a mention with a root mention
        (the earliest or primary coreference).
        
        Args:
            mention: A spaCy Span object.
            root_mention: A spaCy Span object.
        """
        
        mention_span_tuple = (mention.start, mention.end) 
        self.coreference_map[mention_span_tuple] = root_mention
    
    
    def contains(self, mention):
        """
        Determine whether a mention is present in the coreference table.
        
        Args:
            mention: A spaCy Span object.
        
        Returns:
            True if present, otherwise False.
        """
        
        mention_span_tuple = (mention.start, mention.end)
        return mention_span_tuple in self.coreference_map
    
    
    def resolve(self, mention):
        """
        Get the primary coreference for this mention, if one exists.
        
        Args:
            mention: A spaCy Span object.
        
        Returns:
            A spaCy Span object or None.
        """
        
        start = None
        end = None
        
        for token in mention:
            if start is None:
                if token.ent_iob_ != "O":
                    start = token.i
            
            elif end is None:
                if token.ent_iob_ == "O":
                    end = token.i
        
        if start is None:
            start = mention.start
        
        if end is None:
            end = mention.end
        
        logger.debug("Resolve mention: %s (%d-%d): (%d-%d)", mention, mention.start, mention.end, start, end)
        entity_tuple = (start, end)
        
        if entity_tuple in self.coreference_map:
            return self.coreference_map[entity_tuple]
        
        else:
            return None
    
    
    def get_closest_preceding_mentions(self, pronoun, closest_n, sources, contents, quote=None):
        """
        Get the closest preceding mentions for a pronoun.        
        
        Args:
            pronoun:  A spaCy Span object.
            closest_n: The maximum number of mentions to return (int)
            sources: A list of spaCy Span objects.
            contents: A list of spaCy Span objects.
            quote: A citron.data.Quote object, or None.
            
        Returns:
            A list of spaCy Span objects.
            
        """
        preceding_mentions = []
        pronoun_is_gendered = pronoun._.gender not in {"unknown", "neutral"}
        for i in range(len(self.mentions) - 1, -1, -1):
            candidate_mention = self.mentions[i]
            # logger.debug("Candidate mention: %s (%d-%d) (%d-%d)", candidate_mention, candidate_mention.start, candidate_mention.end, pronoun.start, pronoun.end)

            # Ignore mentions that occur after the pronoun
            if candidate_mention.end > pronoun.start:
                continue

            # Ignore candidates which are in content spans
            if utils.are_overlapping_span_lists([candidate_mention], contents):
                logger.debug("Overlapping span: %s", candidate_mention)
                continue

            # Ignore candidates which are in the quote (a pronoun reference would likely not have their name in the quote)
            if quote is not None and mention_in_quote(candidate_mention, quote):
                logger.debug("Skipping mention for being in quote: %s", candidate_mention)
                continue

            # If candidate is a proper noun, and gender is unknown, include it
            # Ignore gender mismatches
            mention_matches_gender = candidate_mention._.gender == pronoun._.gender
            mention_is_noun_or_proper = is_proper_noun(candidate_mention) or is_noun(candidate_mention)
            if (pronoun_is_gendered and not mention_matches_gender) and not (mention_is_noun_or_proper and candidate_mention._.gender == "unknown"):
                logger.debug("Bad gender match: %s %s - %s", candidate_mention, candidate_mention._.gender, pronoun._.gender)
                continue
                
            # Add source span, if it contains the candidate
            for source in sources:
                if utils.are_overlapping_spans(candidate_mention, source):
                    preceding_mentions.append(source)
                    break
            else:
                preceding_mentions.append(candidate_mention)
                
            if len(preceding_mentions) >= closest_n:
                break
        return preceding_mentions
    
    def _build_name_table(self, names):
        """
        Build a table mappping each name span in the document with the earliest, longest matching name
        found earlier in the document.        
        
        Args:
            names: A list of spaCy Span objects.
        
        Returns:
            name_table: a dict mapping names to the longest matching name in the document.
                The names are represented by a tuple containing the start and end index.
        """
        
        name_table = {}
        
        for idx, name in enumerate(names):
            longest_match = None
            
            prefix, _ = split_on_longest_prefix(name)
            if name[-1].ent_type_ == "PERSON" or prefix is not None or is_proper_noun(name):
                # Use earliest, longer instance with matching surname
                for candidate_idx in range(0, idx):             
                    candidate_name = names[candidate_idx]
                    if name.text in candidate_name.text:
                        if longest_match is None or candidate_name.text > longest_match.text:
                            longest_match = candidate_name
                            break
                
                if longest_match is None:
                    # Use earliest, longer instance with matching forename(s)
                    for candidate_name in names:
                        if self.is_longer_with_matching_forenames(candidate_name, name):
                            if longest_match is None or candidate_name.text > longest_match.text:
                                longest_match = candidate_name
                                break

            if longest_match is None:
                # Use earliest instance of matching name
                for candidate_name in names:
                    if name.text == candidate_name.text:
                        longest_match = candidate_name
                        break
            
            # Theoretically this should never be called            
            if longest_match is None:
                longest_match = name
            
            key = (name.start, name.end)
            
            if longest_match != name:  
                if name._.gender == "unknown":
                    name._.gender = longest_match._.gender
                   
                if name._.is_plural is None:
                    name._.is_plural = longest_match._.is_plural 
                
                name_table[key] = longest_match
                
        return name_table
    
    
    def get_names(self, doc, gender_resolver, quotes=None, content_labels=None):
        """
        Get a list of all the name spans within the document.
        
        Args:
            doc: a spaCy Doc object.
            gender_resolver: A citron.gender.ForenameGenderClassifier object.
            quotes: A list of citron.data.Quote objects.
            content_labels: A list containing an IOB label for each token in the document.
        
        Returns:
            names: A list of spaCy Span objects.
        """
        
        # Build list of names for all sources and entities
        names = []
        name_labels = [0] * len(doc)
        
        # TODO: Determine if we actually need this, we should trust the spacy model
        # Add all persons and organisations in the quote sources 
        # if quotes is not None:     
        #     for quote in quotes:
        #         for source in quote.sources:
        #             print("Source", source.start, source.end, source)
        #             if utils.is_person(source) or utils.is_organisation(source):
        #                 names.append(source)
        #                 for i in range(source.start, source.end):
        #                     name_labels[i] = 1

        for sentence in doc.sents:
            for start, stop in utils.get_quoted_text_indices(sentence):
                for i in range(start, stop):
                    name_labels[i] = 1

        # Add all persons and organisations in the document's entities, if they are outside existing names and content labels
        for entity in doc.ents:
            if entity.label_ in ("PERSON", "ORG"):
                for i in range(entity.start, entity.end):
                    if name_labels[i] == 1:
                        break
                    
                    elif content_labels is not None and content_labels[i] != "O":
                        break
                else:           
                    names.append(entity)
        

        # A sorted list of all names and pronouns in the document
        names = sorted(names, key=lambda x: x.start)

        # Determine plurality and gender
        for idx, name in enumerate(names):
            if utils.is_person(name):
                name._.gender = get_gender(name, gender_resolver)
                name._.is_plural = False           
            elif name.label_ == "ORG":
                name._.gender = "neutral"
                name._.is_plural = True

            lower_name = utils.strip_possessive(name.text.lower())
            for name_idx in range(idx - 1, 0, -1):
                lower_match = names[name_idx].text.lower()
                if lower_name in lower_match:
                    name._.gender = names[name_idx]._.gender
                    break
                
        return names
    
    
    # def is_longer_with_matching_surname(self, candidate_name, name):
    #     """
    #     Test whether the candidate name is a longer version of the name with
    #     a matching surname.
        
    #     Args:
    #         candidate_name: a spaCy Span object.
    #         name: a spaCy Span object.
        
    #     Returns:
    #         True, if the name is a longer match, otherwise False.
    #     """
        
    #     if len(candidate_name.text) > len(name.text):
    #         if name[-1].text == candidate_name[-1].text:
    #             return True
        
    #     return False
    
    
    def is_longer_with_matching_forenames(self, candidate_name, name):
        """
        Test whether the candidate name is a longer version of the name with
        a matching forenames.
        
        Args:
            candidate_name: a spaCy Span object.
            name: a spaCy Span object.
        
        Returns:
           True, if the name is a longer match. Otherwise False.
        """
        
        if len(candidate_name.text) > len(name.text) and len(candidate_name) > len(name):
            for i in range(0, len(name)):
                if candidate_name[i].text != name[i].text:
                    return False
            
            return True        
        return False

def mention_in_quote(mention, quote):
    """
    Check whether a mention is in a quote.
    
    Args:
        mention: A spaCy Span object.
        quote: A citron.data.Quote object.
    
    Returns:
        A boolean value.
    """
    for q in quote.contents:
        quote_text = str(q)
        quotes_to_check = utils.get_quoted_text(quote_text)

        if len(quotes_to_check) == 0:
            quotes_to_check = [quote_text]

        for quote_text in quotes_to_check:
            if mention.text in quote_text:
                return True
            for token in mention:
                if (token.ent_type_ == "PERSON" or token.ent_type_ == "ORG") and token.text in quote_text:
                    return True
        
    return False

class ForenamesTable():
    """
    Class which provides gender information about forenames.
    
    The  information is obtained from NLTK but may be supplemented by 
    adding names to two files (male.txt and female.txt) in:
    
        citron/etc/forenames/
    """
    
    def __init__(self):
        """
        Constructor.
        """
        
        self.male_forenames   = set(corpus.names.words("male.txt"))
        self.female_forenames = set(corpus.names.words("female.txt"))
        
        male_filename = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "etc/forenames/male.txt"
        )
        
        if os.path.exists(male_filename):
            self.add_forenames(self.male_forenames, male_filename)
        
        female_filename = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "etc/forenames/female.txt"
        )
        
        if os.path.exists(female_filename):
            self.add_forenames(self.female_forenames, female_filename)
        
        logger.debug("Male forenames:   %s", len(self.male_forenames))
        logger.debug("Female forenames: %s", len(self.female_forenames))
    
    
    def get_forename_gender(self, forename):
        """
        Get the gender of a forename. Ambiguous and unrecognised forenames 
        return "unknown".
        
        Args:
            forename: A string.
        
        Returns:
            Either "male", "female" or "unknown".
        """
        
        is_male   = forename in self.male_forenames
        is_female = forename in self.female_forenames
        
        if is_male and not is_female:
            return "male"
        
        elif is_female and not is_male:
            return "female"
        
        else:
            # Unknown or ambiguous forename
            return "unknown"
    
    
    def add_forenames(self, forenames, filename):
        """
        Add the forenames from a file to a set.
        
        Args:
            forenames: A set of forenames (strings)
            filename: The path (string) of the file.
        """
        
        with open(filename, encoding = "utf-8") as infile:
            for line in infile:
                forename = line.strip()
                
                if len(forename) > 0:
                    forenames.add(forename)