from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
# remove ' character from exclude set because we'd like to keep track of
# phrases like "I'm, someone's"
exclude = set(string.punctuation)
exclude.remove("'")
tknzr = TweetTokenizer()
stopwords_list = stopwords.words('english')


def remove_p(ch):
    if ch in exclude:
        ch = ' '
    return ch


def string2words(s, remove_stopwords=True, remove_digits=True,
                 tolower=True, stem=False):
    if s is None:
        return []
    s = str(s)
    s = s.encode('ascii', errors='ignore').strip().decode('utf-8')
    if tolower:
        s = s.lower()
    # remove \n and \t
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    # remove punctuation
    s = ''.join(remove_p(ch) for ch in s)
    # s = ''.join(ch for ch in s if ch not in exclude)
    words = tknzr.tokenize(s)
    if remove_digits:
        words = [w for w in words if not w.isdigit()]
    if remove_stopwords:
        words = [w for w in words if w not in stopwords_list]
    if stem:
        words = [stemmer.stem(w) for w in words]
    return words


# def pre_process_string(s, tolower=True):
#     s = unicode(s).encode('ascii', 'ignore').strip()
#     if tolower:
#         s = s.lower()
#     # remove \n and \t
#     s = s.replace("\n", " ")
#     s = s.replace("\t", " ")
#     return s


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.
       Same elements will be added
    '''
    z = x.copy()
    z.update(y)
    return z
