import re
import math
from typing import List, Tuple, Dict, Iterable
from pydoc import doc

from utils import timer
from text_processing import TextProcessing
from mongo_db import insert_doc_len_index, insert_vs_index, query_doc, query_vs_index

from scipy import spatial

N = 0

text_processor = TextProcessing.from_nltk()

class InvertedIndex:
    """Inverted Index Data Structure to map tokens to their postings."""
    def __init__(self):
        self.index = []
        self.appearances_dict = dict()
        
    def index_document(self, document: dict) -> None:
        """Process a given document and update the appearances dict."""
        terms = text_processor.get_normalized_tokens(document['title'], document['content_str'])
        for term in terms:
            if term in self.appearances_dict:
                self.appearances_dict[term].append((document['id'], text_processor.tf(get_tf_value(term, document['content_str']))))
            else:
                self.appearances_dict[term] = [(document['id'], text_processor.tf(get_tf_value(term, document['content_str'])))]
    
    def load_index_postings_list(self) -> None:
        for term in self.appearances_dict:
            self.index.append({
                'token': term,
                'doc_tf_index': self.appearances_dict[term]
            })

    @classmethod
    def get_index(self) -> List:
        return self.index



def get_doc_vec_norm(term_tfs: List[float]) -> float:
    """
    helper function, should be called in build_inverted_index
    compute the length of a document vector
    :param term_tfs: a list of term weights (log tf) for one document
    :return:
    """
    return (1/math.sqrt(sum(term_tfs)))
    

@timer
def build_inverted_index(wapo_docs: Iterable) -> None:
    """
    load wapo_pa4.jl to build two indices:
        - "vs_index": for each normalized term as a key, the value should be a list of tuples; each tuple stores the doc id this term appears in and the term weight (log tf) == THIS IS EVERY INDEX OF THE INV INDEX
        - "doc_len_index": for each doc id as a key, the value should be the "length" of that document vector
    insert the indices by using mongo_db.insert_vs_index and mongo_db.insert_doc_len_index method
    """
    global N 
    N = len(wapo_docs)
    inv_ind = InvertedIndex()
    for doc_image in wapo_docs:

        ##! Weight document terms using log TF formula with cosine (length) normalization
        insert_doc_len_index({'doc_id': doc_image['doc_id'], 'length': get_doc_vec_norm([text_processor.tf(get_tf_value(term, doc_image['content_str'])) for term in doc_image['content_str']])})

        inv_ind.index_document(doc_image)
    inv_ind.load_index_postings_list()
    insert_vs_index(inv_ind.get_index())



def parse_query(query: str) -> Tuple[List[str], List[str], List[str]]:
    """
    helper function, should be called in query_inverted_index
    given each query, return a list of normalized terms, a list of stop words and a list of unknown words separately
    """
    normalized_query = text_processor.get_normalized_tokens(query)
    query_list = query.split(' ')
    query_list = re.findall(r"[\w']+|[.,!?;]", query)

    stop_words = {token for token in query_list if not text_processor.normalize(token) in normalized_query}
    unknown_words = {token for token in query_list if isinstance(query_vs_index(text_processor.normalize(token)), type(None)) and not token in stop_words}

    return (query_list, stop_words, unknown_words)
    ##! Weight query terms using logarithmic TF*IDF formula without length normalization

def top_k_docs(doc_scores: Dict[int, float], k: int) -> List[Tuple[float, int]]:
    """
    helper function, should be called in query_inverted_index method
    given the doc_scores, return top k doc ids and corresponding scores using a heap
    :param doc_scores: a dictionary where doc id is the key and cosine similarity score (TO THE PARSED QUERY) is the value
    :param k:
    :return: a list of tuples, each tuple contains (score, doc_id)
    """
    #! use the below to calc cos similarity of two vectors of nums
    # result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
    ##? HERE IS MY BIG QUESTION
    ##? how am I getting the param for this method, is it the doc_len_index? - where is the method for cosine similarity/length normalization supposed to go?
    pass


def query_inverted_index(query: str, k: int) -> Tuple[List[Tuple[float, int]], List[str], List[str]]:
    """
    disjunctive query over the vs_index with the help of mongo_db.query_vs_index, mongo_db.query_doc_len_index methods
    return a list of matched documents (output from the function top_k_docs), a list of stop words and a list of unknown words separately
    """

    #1. get user query
    #2. parse user query
    #3. search index for query's tokens
    #4. get list of postings list
    #5. compute the tf-idf of each term in query
    #6. iterate over the doc ids that were returned
    #7. compute the cosine similarity (tf-idf term * doc tf) in a nested loop over each term - for each term do each doc id
        #s7. (tf-idf*tf)/(query-length * doc-length)
    parsed_query, stop_words, unknown_words = parse_query(query)
    doc_scores = Dict()
    for term in parsed_query:
        postings_list = query_vs_index(term)['doc_tf_index']
        for doc_tf_tuple in postings_list:
            term_tf_idf_score = text_processor.tf(get_tf_value(term, query_doc(doc_tf_tuple[0])['content_str'])) * text_processor.idf(N, len(postings_list))

            cosine_similarity = term_tf_idf_score * doc_tf_tuple[1]

    
def get_tf_value(term: str, content: str) -> int:
    """ This is a method that takes a str representing an individual token and a str representing the set of tokens and calculates
    the number of appearances of the given term in the content """
    return len([token for token in content if token==term])

def cosine_sim(tfidf_term: float, tf_doc: float, query_length: int, doc_length: float) -> float:
    """ This is a method to get the cosine similarity between objects of given information. Namely, this method computed the cosine similarity
    for a given tfidf score, a given tf score, and two vector lengths """
    return (tfidf_term*tf_doc)/(query_length*doc_length)

if __name__ == "__main__":
    pass
