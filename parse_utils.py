"""
Copied from textacy
"""
import pathlib
import re
import itertools

import numpy as np
from spacy import attrs
from spacy.language import Language
from spacy.symbols import NOUN, PROPN, VERB
from spacy.tokens import Doc, Span, Token


from __future__ import annotations

import collections
from operator import attrgetter
from typing import Iterable, List, Optional, Pattern, Tuple

from cytoolz import itertoolz
from spacy.symbols import (
    AUX, VERB,
    agent, attr, aux, auxpass, csubj, csubjpass, dobj, neg, nsubj, nsubjpass, obj, pobj, xcomp,
)
from spacy.tokens import Doc, Span, Token
from parse_utils import *

_NOMINAL_SUBJ_DEPS = {nsubj, nsubjpass}
_CLAUSAL_SUBJ_DEPS = {csubj, csubjpass}
_ACTIVE_SUBJ_DEPS = {csubj, nsubj}
_VERB_MODIFIER_DEPS = {aux, auxpass, neg}

SVOTriple: Tuple[List[Token], List[Token], List[Token]] = collections.namedtuple(
    "SVOTriple", ["subject", "verb", "object"]
)
SSSTriple: Tuple[List[Token], List[Token], List[Token]] = collections.namedtuple(
    "SSSTriple", ["entity", "cue", "fragment"]
)
DQTriple: Tuple[List[Token], List[Token], Span] = collections.namedtuple(
    "DQTriple", ["speaker", "cue", "content"]
)


DEFAULT_DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"

NUMERIC_ENT_TYPES = {"ORDINAL", "CARDINAL", "MONEY", "QUANTITY", "PERCENT", "TIME", "DATE"}
SUBJ_DEPS = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS = {"attr", "dobj", "dative", "oprd"}
AUX_DEPS = {"aux", "auxpass", "neg"}

REPORTING_VERBS = {
    "according", "accuse", "acknowledge", "add", "admit", "agree",
    "allege", "announce", "argue", "ask", "assert", "believe", "blame",
    "charge", "cite", "claim", "complain", "concede", "conclude",
    "confirm", "contend", "criticize", "declare", "decline", "deny",
    "describe", "disagree", "disclose", "estimate", "explain", "fear",
    "hope", "insist", "maintain", "mention", "note", "observe", "order",
    "predict", "promise", "recall", "recommend", "reply", "report", "say",
    "state", "stress", "suggest", "tell", "testify", "think", "urge", "warn",
    "worry", "write"
}

MATCHER_VALID_OPS = {"!", "+", "?", "*"}

POS_REGEX_PATTERNS = {
    "en": {
        "NP": r"<DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN> <PART>?)+",
        "PP": r"<ADP> <DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN> <PART>?)+",
        "VP": r"<AUX>* <ADV>* <VERB>",
    }
}

RE_MATCHER_TOKPAT_DELIM = re.compile(r"\s+")
RE_MATCHER_SPECIAL_VAL = re.compile(
    r"^(int|bool)\([^: ]+\)$",
    flags=re.UNICODE)

RE_ACRONYM = re.compile(
    r"(?:^|(?<=\W))"
    r"(?:"
    r"(?:(?:(?:[A-Z]\.?)+[a-z0-9&/-]?)+(?:[A-Z][s.]?|\ds?))"
    r"|"
    r"(?:\d(?:\-?[A-Z])+)"
    r")"
    r"(?:$|(?=\W))",
    flags=re.UNICODE)

RE_LINEBREAK = re.compile(r"(\r\n|[\n\v])+")
RE_NONBREAKING_SPACE = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)

# regexes for cleaning up crufty terms
RE_DANGLING_PARENS_TERM = re.compile(
    r"(?:\s|^)(\()\s{1,2}(.*?)\s{1,2}(\))(?:\s|$)", flags=re.UNICODE)
RE_LEAD_TAIL_CRUFT_TERM = re.compile(
    r"^[^\w(-]+|[^\w).!?]+$", flags=re.UNICODE)
RE_LEAD_HYPHEN_TERM = re.compile(
    r"^-([^\W\d_])", flags=re.UNICODE)
RE_NEG_DIGIT_TERM = re.compile(
    r"(-) (\d)", flags=re.UNICODE)
RE_WEIRD_HYPHEN_SPACE_TERM = re.compile(
    r"(?<=[^\W\d]) (-[^\W\d])", flags=re.UNICODE)
RE_WEIRD_APOSTR_SPACE_TERM = re.compile(
    r"([^\W\d]+) ('[a-z]{1,2}\b)", flags=re.UNICODE)


def expand_noun(tok: Token) -> List[Token]:
    """Expand a noun token to include all associated conjunct and compound nouns."""
    tok_and_conjuncts = [tok] + list(tok.conjuncts)
    compounds = [
        child
        for tc in tok_and_conjuncts
        for child in tc.children
        # TODO: why doesn't compound import from spacy.symbols?
        if child.dep_ == "compound"
    ]
    return tok_and_conjuncts + compounds


def expand_verb(tok: Token) -> List[Token]:
    """Expand a verb token to include all associated auxiliary and negation tokens."""
    verb_modifiers = [
        child for child in tok.children if child.dep in _VERB_MODIFIER_DEPS
    ]
    return [tok] + verb_modifiers


def is_acronym(token, exclude=None):
    """
    Pass single token as a string, return True/False if is/is not valid acronym.
    Args:
        token (str): Single word to check for acronym-ness
        exclude (Set[str]): If technically valid but not actually good acronyms
            are known in advance, pass them in as a set of strings; matching
            tokens will return False.
    Returns:
        bool
    """
    # exclude certain valid acronyms from consideration
    if exclude and token in exclude:
        return False
    # don't allow empty strings
    if not token:
        return False
    # don't allow spaces
    if " " in token:
        return False
    # 2-character acronyms can't have lower-case letters
    if len(token) == 2 and not token.isupper():
        return False
    # acronyms can't be all digits
    if token.isdigit():
        return False
    # acronyms must have at least one upper-case letter or start/end with a digit
    if not any(char.isupper() for char in token) and not (
        token[0].isdigit() or token[-1].isdigit()
    ):
        return False
    # acronyms must have between 2 and 10 alphanumeric characters
    if not 2 <= sum(1 for char in token if char.isalnum()) <= 10:
        return False
    # only certain combinations of letters, digits, and '&/.-' allowed
    if not RE_ACRONYM.match(token):
        return False
    return True


def merge_spans(spans, doc):
    """
    Merge spans into single tokens in ``doc``, *in-place*.
    Args:
        spans (Iterable[:class:`spacy.tokens.Span`])
        doc (:class:`spacy.tokens.Doc`)
    """
    try:  # retokenizer was added to spacy in v2.0.11
        with doc.retokenize() as retokenizer:
            string_store = doc.vocab.strings
            for span in spans:
                retokenizer.merge(
                    doc[span.start : span.end],
                    attrs=attrs.intify_attrs({"ent_type": span.label}, string_store),
                )
    except AttributeError:
        spans = [(span.start_char, span.end_char, span.label) for span in spans]
        for start_char, end_char, label in spans:
            doc.merge(start_char, end_char, ent_type=label)


def preserve_case(token):
    """
    Return True if ``token`` is a proper noun or acronym; otherwise, False.
    Args:
        token (:class:`spacy.tokens.Token`)
    Returns:
        bool
    Raises:
        ValueError: If parent document has not been POS-tagged.
    """
    if token.doc.is_tagged is False:
        raise ValueError(
            'parent doc of token "{}" has not been POS-tagged'.format(token)
        )
    if token.pos == PROPN or is_acronym(token.text):
        return True
    else:
        return False


def get_normalized_text(span_or_token):
    """
    Get the text of a spaCy span or token, normalized depending on its characteristics.
    For proper nouns and acronyms, text is returned as-is; for everything else,
    text is lemmatized.
    Args:
        span_or_token (:class:`spacy.tokens.Span` or :class:`spacy.tokens.Token`)
    Returns:
        str
    """
    if isinstance(span_or_token, Token):
        return (
            span_or_token.text if preserve_case(span_or_token) else span_or_token.lemma_
        )
    elif isinstance(span_or_token, Span):
        return " ".join(
            token.text if preserve_case(token) else token.lemma_
            for token in span_or_token
        )
    else:
        raise TypeError(
            'input must be a spaCy Token or Span, not "{}"'.format(type(span_or_token))
        )


def get_main_verbs_of_sent(sent):
    """Return the main (non-auxiliary) verbs in a sentence."""
    return [
        tok for tok in sent if tok.pos == VERB and tok.dep_ not in AUX_DEPS
    ]


def get_subjects_of_verb(verb):
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts if tok.dep_ in SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    return subjs


def get_objects_of_verb(verb):
    """
    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """
    objs = [tok for tok in verb.rights if tok.dep_ in OBJ_DEPS]
    # get open clausal complements (xcomp)
    objs.extend(tok for tok in verb.rights if tok.dep_ == "xcomp")
    # get additional conjunct objects
    objs.extend(tok for obj in objs for tok in _get_conjuncts(obj))
    return objs


def _get_conjuncts(tok):
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights if right.dep_ == "conj"]


def get_span_for_compound_noun(noun):
    """
    Return document indexes spanning all (adjacent) tokens
    in a compound noun.
    """
    min_i = noun.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ == "compound", reversed(list(noun.lefts))
        )
    )
    return (min_i, noun.i)


def get_span_for_verb_auxiliaries(verb):
    """
    Return document indexes spanning all (adjacent) tokens
    around a verb that are auxiliary verbs or negations.
    """
    min_i = verb.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in AUX_DEPS, reversed(list(verb.lefts))
        )
    )
    max_i = verb.i + sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in AUX_DEPS, verb.rights
        )
    )
    return (min_i, max_i)




def subject_verb_object_triples(doclike: types.DocLike) -> Iterable[SVOTriple]:
    """
    Extract an ordered sequence of subject-verb-object triples from a document
    or sentence.
    Args:
        doclike
    Yields:
        Next SVO triple as (subject, verb, object), in approximate order of appearance.
    """
    if isinstance(doclike, Span):
        sents = [doclike]
    else:
        sents = doclike.sents

    for sent in sents:
        # connect subjects/objects to direct verb heads
        # and expand them to include conjuncts, compound nouns, ...
        verb_sos = collections.defaultdict(lambda: collections.defaultdict(set))
        for tok in sent:
            head = tok.head
            # ensure entry for all verbs, even if empty
            # to catch conjugate verbs without direct subject/object deps
            if tok.pos == VERB:
                _ = verb_sos[tok]
            # nominal subject of active or passive verb
            if tok.dep in _NOMINAL_SUBJ_DEPS:
                if head.pos == VERB:
                    verb_sos[head]["subjects"].update(expand_noun(tok))
            # clausal subject of active or passive verb
            elif tok.dep in _CLAUSAL_SUBJ_DEPS:
                if head.pos == VERB:
                    verb_sos[head]["subjects"].update(tok.subtree)
            # nominal direct object of transitive verb
            elif tok.dep == dobj:
                if head.pos == VERB:
                    verb_sos[head]["objects"].update(expand_noun(tok))
            # prepositional object acting as agent of passive verb
            elif tok.dep == pobj:
                if head.dep == agent and head.head.pos == VERB:
                    verb_sos[head.head]["objects"].update(expand_noun(tok))
            # open clausal complement, but not as a secondary predicate
            elif tok.dep == xcomp:
                if (
                    head.pos == VERB
                    and not any(child.dep == dobj for child in head.children)
                ):
                    # TODO: just the verb, or the whole tree?
                    # verb_sos[verb]["objects"].update(expand_verb(tok))
                    verb_sos[head]["objects"].update(tok.subtree)
        # fill in any indirect relationships connected via verb conjuncts
        for verb, so_dict in verb_sos.items():
            conjuncts = verb.conjuncts
            if so_dict.get("subjects"):
                for conj in conjuncts:
                    conj_so_dict = verb_sos.get(conj)
                    if conj_so_dict and not conj_so_dict.get("subjects"):
                        conj_so_dict["subjects"].update(so_dict["subjects"])
            if not so_dict.get("objects"):
                so_dict["objects"].update(
                    obj
                    for conj in conjuncts
                    for obj in verb_sos.get(conj, {}).get("objects", [])
                )
        # expand verbs and restructure into svo triples
        for verb, so_dict in verb_sos.items():
            if so_dict["subjects"] and so_dict["objects"]:
                yield SVOTriple(
                    subject=sorted(so_dict["subjects"], key=attrgetter("i")),
                    verb=sorted(expand_verb(verb), key=attrgetter("i")),
                    object=sorted(so_dict["objects"], key=attrgetter("i")),
                )