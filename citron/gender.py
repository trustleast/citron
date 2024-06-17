import nltk
from nltk.corpus import names
import os
import random

from .logger import logger

class ForenameGenderClassifier():
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
        self.male_forenames   = set(names.words("male.txt"))
        self.female_forenames = set(names.words("female.txt"))
        
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
        
        self.classifier = self.train()
    
    
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
        
        predicted = self.classifier.classify(self._get_features(forename))
        logger.debug("Predicted gender for %s: %s", forename, predicted)
        return predicted
    
    
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
    
    def train(self):
        """
        Train the classifier.
        """
        labeled = ([(name, 'male') for name in self.male_forenames] + 
                   [(name, 'female') for name in self.female_forenames])
        random.shuffle(labeled)
        feature_sets = [(self._get_features(n), gender) for (n, gender) in labeled]
        return nltk.NaiveBayesClassifier.train(feature_sets)

    def _get_features(self, name):
        """
        Get the features of a name.

        Args:
            name: A string representing the forename.
        """
        if len(name) == 0:
            return {}
        
        lower_name = name.lower()
        
        # May want to get rid of this
        male_prefix = False
        male_suffix = False
        for hardcode_name in self.male_forenames:
            lower_hardcode = hardcode_name.lower()
            if lower_name.startswith(lower_hardcode):
                male_prefix = True
            if lower_name.endswith(lower_hardcode):
                male_suffix = True

        female_prefix = False
        female_suffix = False
        for hardcode_name in self.female_forenames:
            lower_hardcode = hardcode_name.lower()
            if lower_name.startswith(lower_hardcode):
                female_prefix = True
            if lower_name.endswith(lower_hardcode):
                female_suffix = True
        
        return {
            'suffix1': lower_name[-1:],
            'suffix2': lower_name[-2:],
            'suffix3': lower_name[-3:],
            'prefix1': lower_name[:1],
            'prefix2': lower_name[:2],
            'prefix3': lower_name[:3],
            'male_prefix': male_prefix,
            'male_suffix': male_suffix,
            'female_prefix': female_prefix,
            'female_suffix': female_suffix,
        }
    
