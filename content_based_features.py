"""
This file contains the functions that can be used to extract content-based features from a lecture transcript string and
lecture title
"""
from collections import Counter
from _text_utils import shallow_word_segment
from textatistic import Textatistic
from scipy.stats import entropy
from datetime import datetime
import re
STOPWORDS = frozenset(
    [u'all', 'show', 'anyway', 'fifty', 'four', 'go', 'mill', 'find', 'seemed', 'one', 'whose', 're', u'herself',
     'whoever', 'behind', u'should', u'to', u'only', u'under', 'herein', u'do', u'his', 'get', u'very', 'de',
     'none', 'cannot', 'every', u'during', u'him', u'did', 'cry', 'beforehand', u'these', u'she', 'thereupon',
     u'where', 'ten', 'eleven', 'namely', 'besides', u'are', u'further', 'sincere', 'even', u'what', 'please',
     'yet', 'couldnt', 'enough', u'above', u'between', 'neither', 'ever', 'across', 'thin', u'we', 'full',
     'never', 'however', u'here', 'others', u'hers', 'along', 'fifteen', u'both', 'last', 'many', 'whereafter',
     'wherever', u'against', 'etc', u's', 'became', 'whole', 'otherwise', 'among', 'via', 'co', 'afterwards',
     'seems', 'whatever', 'alone', 'moreover', 'throughout', u'from', 'would', 'two', u'been', 'next', u'few',
     'much', 'call', 'therefore', 'interest', u'themselves', 'thru', u'until', 'empty', u'more', 'fire',
     'latterly', 'hereby', 'else', 'everywhere', 'former', u'those', 'must', u'me', u'myself', u'this', 'bill',
     u'will', u'while', 'anywhere', 'nine', u'can', u'of', u'my', 'whenever', 'give', 'almost', u'is', 'thus',
     u'it', 'cant', u'itself', 'something', u'in', 'ie', u'if', 'inc', 'perhaps', 'six', 'amount', u'same',
     'wherein', 'beside', u'how', 'several', 'whereas', 'see', 'may', u'after', 'upon', 'hereupon', u'such', u'a',
     u'off', 'whereby', 'third', u'i', 'well', 'rather', 'without', u'so', u'the', 'con', u'yours', u'just',
     'less', u'being', 'indeed', u'over', 'move', 'front', 'already', u'through', u'yourselves', 'still', u'its',
     u'before', 'thence', 'somewhere', u'had', 'except', u'ours', u'has', 'might', 'thereafter', u'then', u'them',
     'someone', 'around', 'thereby', 'five', u'they', u'not', u'now', u'nor', 'hereafter', 'name', 'always',
     'whither', u't', u'each', 'become', 'side', 'therein', 'twelve', u'because', 'often', u'doing', 'eg',
     u'some', 'back', u'our', 'beyond', u'ourselves', u'out', u'for', 'bottom', 'since', 'forty', 'per',
     'everything', u'does', 'three', 'either', u'be', 'amoungst', 'whereupon', 'nowhere', 'although', 'found',
     'sixty', 'anyhow', u'by', u'on', u'about', 'anything', u'theirs', 'could', 'put', 'keep', 'whence', 'due',
     'ltd', 'hence', 'onto', u'or', 'first', u'own', 'seeming', 'formerly', u'into', 'within', u'yourself',
     u'down', 'everyone', 'done', 'another', 'thick', u'your', u'her', u'whom', 'twenty', 'top', u'there',
     'system', 'least', 'anyone', u'their', u'too', 'hundred', u'was', u'himself', 'elsewhere', 'mostly', u'that',
     'becoming', 'nobody', u'but', 'somehow', 'part', u'with', u'than', u'he', 'made', 'whether', u'up', 'us',
     'nevertheless', u'below', 'un', u'were', 'toward', u'and', 'describe', u'am', 'mine', u'an', 'meanwhile',
     u'as', 'sometime', u'at', u'have', 'seem', u'any', 'fill', u'again', 'hasnt', u'no', 'latter', u'when',
     'detail', 'also', u'other', 'take', u'which', 'becomes', u'you', 'towards', 'though', u'who', u'most',
     'eight', 'amongst', 'nothing', u'why', u'don', 'noone', 'sometimes', 'together', 'serious', u'having',
     u'once'])

CONJ_WORDS = frozenset(["and", "but", "or", "yet", "nor"])

NORM_SUFFIXES = ["tion", "ment", "ence", "ance"]

TO_BE_VERBS = frozenset(["be", "being", "was", "were", "been", "are", "is"])

PREPOSITION_WORDS = frozenset(
    ["aboard", "about", "above", "according to", "across from", "after", "against", "alongside", "alongside of",
     "along with", "amid", "among", "apart from", "around", "aside from", "at", "away from", "back of", "because of",
     "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond", "but", "by means of",
     "concerning", "considering", "despite", "down", "down from", "during", "except", "except for", "excepting for",
     "from among", "from between", "from under", "in addition to", "in behalf of", "in front of", "in place of",
     "in regard to", "inside of", "inside", "in spite of", "instead of", "into", "like", "near to", "off ",
     "on account of", "on behalf of", "onto", "on top of", "on", "opposite", "out of", "out", "outside", "outside of",
     "over to", "over", "owing to", "past", "prior to", "regarding", "round about", "round", "since", "subsequent to",
     "together", "with", "throughout", "through", "till", "toward", "under", "underneath", "until", "unto", "up",
     "up to", "upon", "with", "within", "without", "across", "along", "by", "of", "in", "to", "near", "of", "from"])

AUXILIARY_VERBS = frozenset(
    ["will", "shall", "cannot", "may", "need to", "would", "should", "could", "might", "must", "ought", "ought to",
     "can’t", "can"])

PRONOUN_WORDS = frozenset(
    ["i", "me", "we", "us", "you", "he", "him", "she", "her", "it", "they", "them", "thou", "thee", "ye", "myself",
     "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves", "oneself", "my", "mine",
     "his", "hers", "yours", "ours", "theirs", "its", "our", "that", "their", "these", "this", "those", "you"])


def get_stopwords(additional_stopwords=set()):
    """returns the default stopword set aggregated to a custom stopword set provided.

    Args:
        additional_stopwords ({str}): set of additional stopwords to be added to the default stopword set.

    Returns:
        {str}: frozenset of stopwords
    """
    return STOPWORDS | additional_stopwords


def word_count(s):
    """returns word count of a string

    Args:
        s (str): string to be word counted.

    Returns:
        (int): number of words in the string

    """
    return len(shallow_word_segment(s))


def title_word_count(doc):
    """returns word count of a title

    Args:
        title (str): title string to be word counted.

    Returns:
        (int): number of words in the title

    """
    first_line = doc.split('\n')[0]

    # Counting the number of words in the first line
    return len(first_line.split())


def compute_entropy(word_list):
    """Computes document entropy of a transcript calculated
    according to https://people.cs.umass.edu/~yanlei/publications/wsdm11.pdf

        Args:
            word_list ([str]): list of words in the transcript.

        Returns:
            (float): document entropy value
    """

    word_histogram = Counter(word_list)
    total_word_count = float(len(word_list))

    word_probs = []
    for _, count in word_histogram.items():
        mle_pw = count / float(total_word_count)
        word_probs.append(mle_pw)

    return entropy(word_probs, base=2)



def count_sentences(text):
    # Simple approach to count the number of sentences
    return len(re.findall(r'[.!?]+', text))

def count_words(text):
    # Counts words assuming they are separated by whitespace
    return len(text.split())

def estimate_syllable_count(word):
    # Count vowel groups as a rough estimate of syllable count
    return len(re.findall(r'[aeiouy]+', word, re.IGNORECASE))

def count_syllables(text):
    # Estimate syllables in each word
    words = text.split()
    syllable_count = sum(estimate_syllable_count(word) for word in words)
    return syllable_count

def get_readability_features(text):
    """Get FK easiness readability score from a text
    calculated according to https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests

    Args:
        text (str): text string of the transcript

    Returns:
        easiness (float): FK ease readability score for the text.
    """
    try:
        total_sentences = count_sentences(text)
        total_words = count_words(text)
        total_syllables = count_syllables(text)

        # Flesch-Kincaid readability formula
        easiness = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    except ZeroDivisionError:
        easiness = 100.0

    return easiness


def compute_stop_word_presence_rate(word_list, stop_word_set=get_stopwords()):
    """returns the stopword presence rate
    calculated according to fracStops in https://people.cs.umass.edu/~yanlei/publications/wsdm11.pdf

    Args:
        word_list ([str]): list of words in the transcript.
        stop_word_set ({set}): set of stopwords to be used

    Returns:
        (float): stopword presence rate
    """
    word_count = float(len(word_list))

    # if no words, return 0
    if word_count == 0:
        return 0.

    stopwords_count = 0.

    for w in word_list:
        if w in stop_word_set:
            stopwords_count += 1

    return stopwords_count / word_count


def compute_stop_word_coverage_rate(word_list, stop_word_set=get_stopwords()):
    """returns the stopword coverage rate
    calculated according to stopCover in https://people.cs.umass.edu/~yanlei/publications/wsdm11.pdf

    Args:
        word_list ([str]): list of words in the transcript.
        stop_word_set ({set}): set of stopwords to be used

    Returns:
        (float): stopword coverage rate
    """
    word_count = float(len(word_list))

    # if no words, return 0
    if word_count == 0:
        return 0.

    stopwords_cardinality = float(len(stop_word_set))
    stopwords_present = set()

    for w in word_list:
        if w in stop_word_set:
            stopwords_present.add(w)

    return len(stopwords_present) / stopwords_cardinality


def compute_conjunction_rate(word_list):
    """Compute conjugation word rate
    calculated according to https://dl.acm.org/doi/pdf/10.1145/2063504.2063507

    Args:
        word_list ([str]): list of words in the transcript.

    Returns:
        (float): conjugation word rate
    """
    word_count = float(len(word_list))

    if word_count == 0:
        return 0.

    qualified_count = 0

    for word in word_list:
        if word.strip() in CONJ_WORDS:
            qualified_count += 1

    return qualified_count / word_count


def compute_normalization_rate(word_list):
    """Compute normalization suffix rate
    calculated according to https://dl.acm.org/doi/pdf/10.1145/2063504.2063507

        Args:
            word_list ([str]): list of words in the transcript.

        Returns:
            (float): normalization suffix rate
        """
    word_count = float(len(word_list))

    if word_count == 0:
        return 0.

    qualified_count = 0

    for word in word_list:
        _word_ = word.strip()
        for suffix in NORM_SUFFIXES:
            if _word_.endswith(suffix):
                qualified_count += 1
                break

    return qualified_count / word_count


def compute_preposition_rate(word_list):
    """Compute preposition word rate
    calculated according to https://dl.acm.org/doi/pdf/10.1145/2063504.2063507

        Args:
            word_list ([str]): list of words in the transcript.

        Returns:
            (float): preposition word rate
        """
    word_count = float(len(word_list))

    if word_count == 0:
        return 0.

    qualified_count = 0

    for word in word_list:
        if word.strip() in PREPOSITION_WORDS:
            qualified_count += 1

    return qualified_count / word_count


def compute_tobe_verb_rate(word_list):
    """Compute to-be verb word rate
    calculated according to https://dl.acm.org/doi/pdf/10.1145/2063504.2063507

        Args:
            word_list ([str]): list of words in the transcript.

        Returns:
            (float): to-be verb word
        """
    word_count = float(len(word_list))

    if word_count == 0:
        return 0.

    qualified_count = 0

    for word in word_list:
        if word.strip() in TO_BE_VERBS:
            qualified_count += 1

    return qualified_count / word_count


def compute_auxiliary_verb_rate(word_list):
    """Compute auxiliary verb word rate
    calculated according to https://dl.acm.org/doi/pdf/10.1145/2063504.2063507

        Args:
            word_list ([str]): list of words in the transcript.

        Returns:
            (float): auxiliary verb word rate
        """
    word_count = float(len(word_list))

    if word_count == 0:
        return 0.

    qualified_count = 0

    for word in word_list:
        if word.strip() in AUXILIARY_VERBS:
            qualified_count += 1

    return qualified_count / word_count


def compute_pronouns_rate(word_list):
    """Compute pronoun word rate
    calculated according to https://dl.acm.org/doi/pdf/10.1145/2063504.2063507

        Args:
            word_list ([str]): list of words in the transcript.

        Returns:
            (float): pronoun word rate
        """
    word_count = float(len(word_list))

    if word_count == 0:
        return 0.

    qualified_count = 0

    for word in word_list:
        if word.strip() in PRONOUN_WORDS:
            qualified_count += 1

    return qualified_count / word_count

def compute_freshness(publication_date_str):
    """Compute the freshness based on the publication date.

    Args:
        publication_date_str (str): Publication date in the format 'YYYY-MM-DD'.

    Returns:
        int: Number of days since January 1, 1970.
    """
    publication_date = datetime.strptime(publication_date_str, '%Y-%m-%d')
    reference_date = datetime(1970, 1, 1)
    return (publication_date - reference_date).days