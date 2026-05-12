"""
nlp.py — Pure-Python Chat Processor
=====================================
Smart India Travel Guide Chatbot

This module handles the complete user-input pipeline **without** any
external NLP libraries (no spaCy, NLTK, or LLM calls).  Every stage —
cleaning, entity extraction, intent classification, and response
generation — relies on deterministic Python string manipulation and
pattern matching against the live semantic network.

Pipeline Stages
---------------
1. **Clean**   — normalise case, strip punctuation, collapse whitespace.
2. **Extract** — match n-grams against known graph nodes.
3. **Route**   — classify intent via keyword/pattern heuristics.
4. **Respond** — delegate to the appropriate ``SemanticNetwork`` method
   and format an explainable answer with a reasoning path.
"""

from __future__ import annotations

import logging
import random
import re
import string
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from network import SemanticNetwork

logger = logging.getLogger(__name__)


# ======================================================================
#  Constants
# ======================================================================

# Words too common to be meaningful entities
_STOP_WORDS: frozenset[str] = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "a", "an", "the", "is", "am", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "about", "between", "through", "after", "before",
    "above", "below", "and", "but", "or", "nor", "not", "so", "if",
    "then", "than", "too", "very", "just", "also", "how", "what",
    "where", "when", "which", "who", "whom", "this", "that", "these",
    "those", "all", "each", "every", "any", "some", "no", "there",
    "here", "up", "out", "off", "over", "under", "again", "further",
    "tell", "show", "give", "get", "go", "know", "like", "want",
    "need", "find", "visit", "travel", "trip", "place", "places",
    "thing", "things", "please", "thanks", "thank", "hey", "hi",
    "hello", "good", "best", "why", "suggest", "recommend",
})


def format_list_naturally(items: list[str]) -> str:
    """Join a list into natural, grammatically correct English.

    This is the **public helper** used throughout the chatbot to turn
    any Python list of strings into a phrase a 5-year-old could read.

    Examples
    --------
    >>> format_list_naturally(['Manali'])
    'Manali'
    >>> format_list_naturally(['Manali', 'Munnar'])
    'Manali and Munnar'
    >>> format_list_naturally(['Manali', 'Munnar', 'Coorg'])
    'Manali, Munnar, and Coorg'
    >>> format_list_naturally([])
    ''
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def generate_place_response(
    place: str,
    concepts_list: list,
    activities_list: list,
) -> str:
    """Build an enthusiastic, 5-year-old-friendly answer about a *place*.

    This is the **Forward Reasoning** response formatter.  Given a place
    and its associated concepts / activities (already collected by the
    semantic-network traversal), it picks a random conversational
    template and fills it in with natural-language lists.

    Parameters
    ----------
    place : str
        The destination name (e.g. ``"Goa"``).
    concepts_list : list
        High-level concepts linked to the place
        (e.g. ``["Beaches", "Nightlife", "Portuguese Heritage"]``).
    activities_list : list
        Concrete activities available at the place
        (e.g. ``["Scuba Diving", "Parasailing", "Beach Parties"]``).

    Returns
    -------
    str
        A complete, ready-to-display response string.

    Examples
    --------
    >>> random.seed(0)
    >>> generate_place_response("Goa", ["Beaches"], ["Scuba Diving"])
    "If you go to Goa, it's like a giant playground! ..."
    """
    concepts = format_list_naturally(concepts_list)
    activities = format_list_naturally(activities_list)

    templates: list[str] = [
        (
            f"If you go to {place}, it's like a giant playground! "
            f"You get to see amazing {concepts} and play by doing "
            f"{activities}. It's going to be awesome! \U0001F389"
        ),
        (
            f"{place} is wonderful! \U0001F929 Make sure you look at the "
            f"{concepts} and you absolutely have to try {activities}. "
            f"It's the best!"
        ),
        (
            f"Oh wow, {place}! \U0001F30D That place has the coolest "
            f"{concepts} ever! And guess what? You can do {activities} "
            f"there — how fun is that?!"
        ),
        (
            f"You're going to LOVE {place}! \U00002728 It's packed with "
            f"{concepts}, and for fun you can do {activities}. "
            f"Seriously, it doesn't get better than this!"
        ),
        (
            f"Guess what? {place} is like a treasure chest! \U0001F4AB "
            f"Inside you'll find {concepts}, and the best part is you "
            f"get to enjoy {activities}. Let's go!"
        ),
    ]

    return random.choice(templates)


def generate_activity_response(
    activity: str,
    places_list: list,
) -> str:
    """Build an enthusiastic, 5-year-old-friendly answer about an *activity*.

    This is the **Reverse Reasoning** response formatter.  Given an
    activity and the list of destinations that offer it, it picks a
    random conversational template and fills it in.

    Parameters
    ----------
    activity : str
        The activity or concept the user asked about
        (e.g. ``"Trekking"``, ``"Scuba Diving"``).
    places_list : list
        Destinations where the activity is available
        (e.g. ``["Manali", "Munnar", "Coorg"]``).

    Returns
    -------
    str
        A complete, ready-to-display response string.

    Examples
    --------
    >>> random.seed(0)
    >>> generate_activity_response("Trekking", ["Manali", "Munnar"])
    "Oh, you like Trekking? ..."
    """
    places = format_list_naturally(places_list)

    templates: list[str] = [
        (
            f"Oh, you like {activity}? That sounds like a big adventure! "
            f"\U0001F3AF You should definitely visit {places}. "
            f"They are famous for it!"
        ),
        (
            f"Wow, {activity} is so much fun! \U0001F60D The absolute best "
            f"places to go for that are {places}. You will love it!"
        ),
        (
            f"Looking for {activity}? Great taste! \U0001F31F Head to "
            f"{places} \u2014 they're the top spots in India for that!"
        ),
        (
            f"{activity}? Oh, you're going to have a blast! \U0001F525 "
            f"Check out {places} \u2014 each one is an incredible experience!"
        ),
        (
            f"You want to try {activity}? YES! \U0001F929 The best places "
            f"for that are {places}. Trust me, you'll have the time of "
            f"your life!"
        ),
    ]

    return random.choice(templates)


# ======================================================================
#  Small Talk Responses
# ======================================================================

_SMALL_TALK: dict[str, list[str]] = {
    "hello": [
        "Namaste! \U0001F64F I'm your friendly travel guide! Where would you like to go in India, or what fun activities do you want to do?",
        "Hey there, explorer! \U0001F30D Ready to discover the magic of India? Ask me about any destination or activity!",
    ],
    "hi": [
        "Namaste! \U0001F64F I'm your friendly travel guide! Where would you like to go in India, or what fun activities do you want to do?",
        "Hi there! \U0001F44B I know all about 20 amazing Indian destinations. What are you curious about?",
    ],
    "hey": [
        "Hey! \U0001F44B Ready for an adventure? Tell me a place you want to explore or an activity you love!",
    ],
    "thanks": [
        "You're so welcome! \U0001F60A Happy to help you plan an amazing trip. Got more questions?",
        "Anytime, friend! That's what I'm here for. Ask me anything else about India! \U0001F1EE\U0001F1F3",
    ],
    "thank you": [
        "You're very welcome! \U0001F60A Wishing you the most amazing travels!",
    ],
    "bye": [
        "Happy travels! \U0001F30D Come back anytime you need travel inspiration. Phir milenge! \U0001F44B",
    ],
    "help": [
        "Of course! Here's what I can do:\n\n\U0001F3D6 Ask about a place: \"Tell me about Goa\"\n\U000026F0 Find an activity: \"Where can I do Trekking?\"\n\U0001F9E0 Explore why: \"Why visit Manali for Better Health?\"\n\U00002744 Filter by season: \"Best places in Winter\"\n\U0001F331 Find similar spots: \"Places similar to Kerala\"",
    ],
}


# ======================================================================
#  Conversational Templates
# ======================================================================

_FORWARD_TEMPLATES: list[str] = [
    "Oh, {place} is absolutely wonderful! \U0001F929 You can enjoy {activities} there. Plus, you'll get to experience {features}. It's going to be an unforgettable trip!",
    "Great choice! {place} is a gem! \U0001F48E You should definitely try {activities}, and make sure to check out the {features}. You'll love every moment!",
    "If you go to {place}, you're going to have the BEST time! \U0001F389 There's {activities} to enjoy, and the {features} are simply magical!",
    "{place}? Amazing pick! \U00002728 It's famous for its {features}, and you can have a blast doing {activities}. Pack your bags!",
]

_REVERSE_TEMPLATES: list[str] = [
    "Oh, you like {activity}? That sounds like a big adventure! \U0001F3AF You should definitely visit {places}. They are famous for it!",
    "Wow, {activity} is so much fun! \U0001F60D The absolute best places to go for that are {places}. You will love it!",
    "Looking for {activity}? Great taste! \U0001F31F Head to {places} — they're the top spots in India for that!",
    "{activity}? Oh, you're going to have a blast! \U0001F525 Check out {places} — each one is an incredible experience!",
]

_PATH_TEMPLATES: list[str] = [
    "Great question! Here's exactly why {start} connects to {end} — it's like a chain of awesomeness! \U0001F9E0\n\n{steps}\n\nPretty cool, right? Each step leads naturally to the next! \U0001F31F",
    "Ooh, I love this one! Let me show you the magical connection between {start} and {end}! \U00002728\n\n{steps}\n\nSee? It all connects beautifully! \U0001F4AB",
]

_ATTRIBUTE_TEMPLATES: list[str] = [
    "Looking for the best {attr} destinations? Here you go! \U0001F389\n\n{list}\n\nAny of these would make an amazing trip! \U0001F60D",
    "Great timing! Here are the perfect {attr} destinations in India! \U0001F1EE\U0001F1F3\n\n{list}\n\nWant me to tell you more about any of them? \U00002728",
]

_RELATED_TEMPLATES: list[str] = [
    "If you love {place}, you're going to adore these similar destinations! \U0001F496\n\n{list}\n\nWant to explore any of them?",
    "Oh, a fan of {place}? Excellent taste! \U0001F31F Here are places with a similar vibe:\n\n{list}\n\nShall I tell you more about any of these?",
]


# ======================================================================
#  Intent Enum
# ======================================================================

class Intent(Enum):
    """Recognised user-intent categories."""

    FORWARD = auto()    # "Tell me about X"
    REVERSE = auto()    # "Where can I do X?"
    PATH = auto()       # "Why visit X for Y?"
    ATTRIBUTE = auto()  # "What places are good in Winter?"
    RELATED = auto()    # "Places similar to X"
    UNKNOWN = auto()    # Fallback


# ======================================================================
#  Structured Response
# ======================================================================

@dataclass
class ChatResponse:
    """Immutable container returned by ``ChatProcessor.process``.

    Attributes
    ----------
    message : str
        The human-readable answer string.
    reasoning_path : list[tuple[str, str, str]]
        The ordered ``(source, predicate, target)`` hops that justify
        the answer.  Empty when no traversal was needed.
    intent : Intent
        The classified intent of the user query.
    matched_entities : list[str]
        Entities extracted from the user's raw input.
    """

    message: str
    reasoning_path: list[tuple[str, str, str]] = field(default_factory=list)
    intent: Intent = Intent.UNKNOWN
    matched_entities: list[str] = field(default_factory=list)


# ======================================================================
#  Intent-Detection Patterns
# ======================================================================

# Each entry: (compiled regex, Intent, description)
# Patterns are tried in order; first match wins.

_INTENT_PATTERNS: list[tuple[re.Pattern[str], Intent]] = [
    # PATH — two-entity patterns
    (re.compile(r"\bwhy\b.*\bfor\b", re.I), Intent.PATH),
    (re.compile(r"\bhow\b.*\bconnect", re.I), Intent.PATH),
    (re.compile(r"\bpath\b.*\bbetween\b", re.I), Intent.PATH),
    (re.compile(r"\blink\b.*\bbetween\b", re.I), Intent.PATH),
    (re.compile(r"\b(?:relat(?:e|ion)|similar)\b", re.I), Intent.RELATED),

    # REVERSE — activity / concept → place lookups
    (re.compile(r"\bwhere\b.*\b(?:can|do|find|try|enjoy)\b", re.I), Intent.REVERSE),
    (re.compile(r"\bwhich\b.*\b(?:place|dest)", re.I), Intent.REVERSE),
    (re.compile(r"\bwho\b.*\boffer", re.I), Intent.REVERSE),
    (re.compile(r"\b(?:place|dest)\w*\b.*\b(?:offer|have|provide|for|with)\b", re.I), Intent.REVERSE),
    # NEW — catch "suggest/show/find/list places" patterns
    (re.compile(r"\b(?:suggest|show|find|list|give)\b.*\b(?:place|dest)", re.I), Intent.REVERSE),
    (re.compile(r"\b(?:place|dest)\w*\b.*\b(?:with|having|featuring)\b", re.I), Intent.REVERSE),
    (re.compile(r"\bwhere\b.*\b(?:is|are|have)\b", re.I), Intent.REVERSE),
    (re.compile(r"\b(?:find|get|see)\b.*\b(?:place|dest)", re.I), Intent.REVERSE),
    (re.compile(r"\b(?:suggest|recommend)\b", re.I), Intent.REVERSE),

    # ATTRIBUTE — season / budget / type queries
    (re.compile(r"\b(?:season|month|when|budget|cost|cheap|expensive)\b", re.I), Intent.ATTRIBUTE),
    (re.compile(r"\b(?:winter|summer|monsoon)\b", re.I), Intent.ATTRIBUTE),
    (re.compile(r"\b(?:low|moderate|high)\s*budget\b", re.I), Intent.ATTRIBUTE),

    # FORWARD — destination-centric (broad catch)
    (re.compile(r"\btell\b.*\babout\b", re.I), Intent.FORWARD),
    (re.compile(r"\bwhat\b.*\b(?:offer|do|see|explore|about)\b", re.I), Intent.FORWARD),
    (re.compile(r"\b(?:explore|discover|describe|info)\b", re.I), Intent.FORWARD),
]


# ======================================================================
#  ChatProcessor Class
# ======================================================================

class ChatProcessor:
    """Deterministic NLP pipeline that converts raw user text into
    structured answers backed by semantic-network reasoning.

    Parameters
    ----------
    network : SemanticNetwork, optional
        A pre-built network instance.  If *None*, a new one is
        constructed from ``data.RELATIONSHIPS`` automatically.

    Examples
    --------
    >>> proc = ChatProcessor()
    >>> resp = proc.process("Tell me about Goa")
    >>> print(resp.message)
    >>> print(resp.reasoning_path)
    """

    def __init__(self, network: Optional[SemanticNetwork] = None) -> None:
        self._net: SemanticNetwork = network or SemanticNetwork()

        # Pre-compute a lookup of lowercased node → original-case node,
        # sorted longest-first so multi-word entities match before
        # single-word sub-strings (e.g. "Rann of Kutch" before "Kutch").
        self._node_lookup: list[tuple[str, str]] = sorted(
            ((n.lower(), n) for n in self._net.all_nodes),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        logger.info(
            "ChatProcessor ready — %d matchable entities loaded.",
            len(self._node_lookup),
        )

    # ------------------------------------------------------------------
    #  Stage 1 — Text Cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(raw: str) -> str:
        """Normalise user input for downstream matching.

        Steps performed:
        1. Convert to lowercase.
        2. Strip leading/trailing whitespace.
        3. Remove punctuation (except hyphens inside words).
        4. Collapse multiple spaces into one.

        Parameters
        ----------
        raw : str
            The unprocessed user message.

        Returns
        -------
        str
            The cleaned string.
        """
        text = raw.lower().strip()
        # Remove punctuation but keep intra-word hyphens
        text = re.sub(r"[^\w\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    #  Stage 2 — Entity Extraction
    # ------------------------------------------------------------------

    def extract_entities(self, cleaned: str) -> list[str]:
        """Identify semantic-network nodes mentioned in *cleaned* text.

        The method performs **greedy longest-match** scanning: it checks
        multi-word node names first (e.g. ``"rann of kutch"``) so they
        are not broken into partial hits.

        Parameters
        ----------
        cleaned : str
            Pre-cleaned (lowercased) user text.

        Returns
        -------
        list[str]
            Original-case node names found, ordered by first appearance.
        """
        found: list[str] = []
        remaining = f" {cleaned} "  # pad for whole-word boundary checks

        for lower_node, original_node in self._node_lookup:
            # Skip trivially short / stop-word nodes
            if lower_node in _STOP_WORDS or len(lower_node) < 2:
                continue

            # Whole-word boundary search (case-insensitive by design:
            # both `lower_node` and `remaining` are already lowercase)
            pattern = rf"(?<!\w){re.escape(lower_node)}(?!\w)"
            if re.search(pattern, remaining):
                found.append(original_node)
                # Remove matched span to prevent sub-matches
                remaining = re.sub(pattern, " ", remaining, count=1)

        # ── DEBUG: visible in terminal so you can verify extraction ──
        print(f"[DEBUG] Entities found: {found}")
        logger.debug("Extracted entities: %s", found)
        return found

    # ------------------------------------------------------------------
    #  Stage 3 — Intent Classification
    # ------------------------------------------------------------------

    def classify_intent(
        self,
        cleaned: str,
        entities: list[str],
    ) -> Intent:
        """Determine the user's intent from cleaned text and entities.

        Intent is resolved by testing the input against an ordered list
        of regex patterns.  If no pattern matches, a heuristic based on
        entity count is applied as a fallback.

        Parameters
        ----------
        cleaned : str
            Pre-cleaned user text.
        entities : list[str]
            Entities already extracted from the text.

        Returns
        -------
        Intent
            The best-matching intent category.
        """
        for pattern, intent in _INTENT_PATTERNS:
            if pattern.search(cleaned):
                print(f"[DEBUG] Intent matched pattern → {intent.name}")
                logger.debug("Intent matched pattern → %s", intent.name)
                return intent

        # Fallback heuristics
        if len(entities) >= 2:
            return Intent.PATH
        if len(entities) == 1:
            # Smart fallback: if the entity is NOT a destination
            # (i.e. it has few outgoing edges, like "Mountains" or
            # "Trekking"), the user likely wants to find PLACES that
            # offer it → REVERSE.  If it IS a destination (like
            # "Goa"), they want to know what it offers → FORWARD.
            entity = entities[0]
            outgoing = self._net.get_outgoing_edges(entity)
            if len(outgoing) >= 3:
                print(f"[DEBUG] Fallback → FORWARD ('{entity}' is a destination with {len(outgoing)} edges)")
                return Intent.FORWARD
            else:
                print(f"[DEBUG] Fallback → REVERSE ('{entity}' is a concept with {len(outgoing)} edges)")
                return Intent.REVERSE
        return Intent.UNKNOWN

    # ------------------------------------------------------------------
    #  Stage 4 — Dispatch & Response Generation
    # ------------------------------------------------------------------

    def process(self, user_input: str) -> ChatResponse:
        """Run the full pipeline: clean → extract → route → respond.

        This is the **single entry-point** that external callers should
        use.

        Parameters
        ----------
        user_input : str
            Raw text typed by the user.

        Returns
        -------
        ChatResponse
            A structured response containing the answer message,
            reasoning path, detected intent, and matched entities.
        """
        cleaned = self.clean_text(user_input)
        print(f"[DEBUG] Cleaned input: '{cleaned}'")

        # ── Small-talk early exit ────────────────────────────────
        # Check for greetings / pleasantries *before* doing any
        # heavy entity extraction or intent classification.
        small_talk_reply = self._check_small_talk(cleaned)
        if small_talk_reply is not None:
            logger.info("Processing — small-talk shortcut for %r", cleaned)
            return small_talk_reply

        # ── Full NLP pipeline ────────────────────────────────────
        entities = self.extract_entities(cleaned)
        intent = self.classify_intent(cleaned, entities)

        logger.info(
            "Processing — cleaned=%r  entities=%s  intent=%s",
            cleaned, entities, intent.name,
        )

        # Dispatch to the appropriate handler
        handler = {
            Intent.FORWARD: self._handle_forward,
            Intent.REVERSE: self._handle_reverse,
            Intent.PATH: self._handle_path,
            Intent.ATTRIBUTE: self._handle_attribute,
            Intent.RELATED: self._handle_related,
            Intent.UNKNOWN: self._handle_unknown,
        }[intent]

        return handler(cleaned, entities)

    # ------------------------------------------------------------------
    #  Intent Handlers
    # ------------------------------------------------------------------

    def _handle_forward(
        self, cleaned: str, entities: list[str],
    ) -> ChatResponse:
        """Handle forward-reasoning queries (e.g. 'Tell me about Goa').

        Classifies BFS results into *concepts* (depth-1, typically
        ``has`` edges like Beaches, Mountains) and *activities*
        (depth-2, typically ``enable`` edges like Trekking, Scuba
        Diving), then passes them through
        ``generate_place_response()`` for a friendly answer.
        """
        if not entities:
            return self._no_entity_response(Intent.FORWARD)

        target = entities[0]
        results = self._net.forward_reasoning(target, max_depth=3)

        if not results:
            return ChatResponse(
                message=f"I know about {target}, but I couldn't find "
                        f"detailed connections for it in my knowledge base.",
                intent=Intent.FORWARD,
                matched_entities=entities,
            )

        # ── Classify graph nodes into concepts vs. activities ─────
        #  Depth-1 nodes ("has" edges)    → concepts / features
        #  Depth-2 nodes ("enable" edges) → activities
        #  Anything else is ignored for the friendly message but
        #  kept in the reasoning path for transparency.
        concepts: list[str] = []
        activities: list[str] = []
        all_chains: list[tuple[str, str, str]] = []

        # Predicates that mark metadata, not interesting features
        _META_PREDICATES = {
            "best_season", "budget_level", "located_in", "type",
            "also_known_as", "known_for",
        }

        for node, chain in results.items():
            all_chains.extend(chain)
            depth = len(chain)
            last_predicate = chain[-1][1] if chain else ""

            # Skip metadata edges (season, budget, region, etc.)
            if last_predicate in _META_PREDICATES:
                continue

            if depth == 1:
                concepts.append(node)
            elif depth == 2 and last_predicate == "enable":
                activities.append(node)

        # Deduplicate the reasoning path
        unique_path = list(dict.fromkeys(all_chains))

        # ── Build the friendly response ──────────────────────────
        if concepts or activities:
            message = generate_place_response(
                target,
                concepts if concepts else ["some wonderful sights"],
                activities if activities else ["lots of fun things"],
            )
        else:
            # Extremely unlikely fallback
            message = (
                f"I know about {target}, but I couldn't neatly "
                f"separate its features. Here's the raw data: "
                f"{', '.join(results.keys())}"
            )

        return ChatResponse(
            message=message,
            reasoning_path=unique_path,
            intent=Intent.FORWARD,
            matched_entities=entities,
        )

    def _handle_reverse(
        self, cleaned: str, entities: list[str],
    ) -> ChatResponse:
        """Handle reverse-reasoning queries (e.g. 'Where can I trek?').

        Collects destination-level nodes from the reverse BFS
        results and passes them through
        ``generate_activity_response()`` for a friendly answer.
        """
        if not entities:
            return self._no_entity_response(Intent.REVERSE)

        target = entities[0]
        results = self._net.reverse_reasoning(target, max_depth=3)

        if not results:
            return ChatResponse(
                message=f"I couldn't find any places connected to "
                        f"'{target}' in my knowledge base.",
                intent=Intent.REVERSE,
                matched_entities=entities,
            )

        # Separate destinations (rich outgoing edges) from concepts
        destinations: list[str] = []
        all_chains: list[tuple[str, str, str]] = []

        for node, chain in results.items():
            all_chains.extend(chain)
            # A node is likely a "destination" if it has many outgoing edges
            if len(self._net.get_outgoing_edges(node)) >= 3:
                destinations.append(node)

        unique_path = list(dict.fromkeys(all_chains))

        # ── Build the friendly response ──────────────────────────
        if destinations:
            message = generate_activity_response(target, destinations)
        else:
            # Fallback: list whatever we found
            all_nodes = list(results.keys())[:10]
            message = generate_activity_response(target, all_nodes)

        return ChatResponse(
            message=message,
            reasoning_path=unique_path,
            intent=Intent.REVERSE,
            matched_entities=entities,
        )

    def _handle_path(
        self, cleaned: str, entities: list[str],
    ) -> ChatResponse:
        """Handle path-finding queries (e.g. 'Why Manali for trekking?').

        Uses ``_PATH_TEMPLATES`` with ``random.choice()`` for a
        friendly explanation.
        """
        if len(entities) < 2:
            return ChatResponse(
                message="I need two concepts to trace a reasoning path. "
                        "Try: 'Why visit Manali for Better Health?'",
                intent=Intent.PATH,
                matched_entities=entities,
            )

        start, end = entities[0], entities[1]

        # Try both directions
        path = self._net.find_path(start, end)
        if path is None:
            path = self._net.find_path(end, start)
            if path is not None:
                start, end = end, start

        if path is None or len(path) == 0:
            return ChatResponse(
                message=f"I couldn't find a reasoning path between "
                        f"'{entities[0]}' and '{entities[1]}'.",
                intent=Intent.PATH,
                matched_entities=entities,
            )

        # Build a numbered step list for the template
        step_lines: list[str] = []
        for i, (src, pred, tgt) in enumerate(path, 1):
            step_lines.append(f"  {i}. {src}  \u2500[{pred}]\u2500\u25B6  {tgt}")
        steps_str = "\n".join(step_lines)

        message = random.choice(_PATH_TEMPLATES).format(
            start=start, end=end, steps=steps_str,
        )

        return ChatResponse(
            message=message,
            reasoning_path=path,
            intent=Intent.PATH,
            matched_entities=entities,
        )

    def _handle_attribute(
        self, cleaned: str, entities: list[str],
    ) -> ChatResponse:
        """Handle attribute queries (e.g. 'best places in Winter').

        Uses ``_ATTRIBUTE_TEMPLATES`` with ``random.choice()`` for a
        friendly presentation.
        """
        # Detect attribute type from keywords
        attr_map: list[tuple[str, str, list[str]]] = [
            ("best_season", "Winter", ["winter"]),
            ("best_season", "Summer", ["summer"]),
            ("best_season", "Monsoon", ["monsoon", "rain"]),
            ("budget_level", "Low Budget", ["low budget", "cheap", "budget friendly"]),
            ("budget_level", "Moderate Budget", ["moderate budget", "mid range"]),
            ("budget_level", "High Budget", ["high budget", "expensive", "luxury"]),
        ]

        predicate: Optional[str] = None
        value: Optional[str] = None

        # First check if any entity is a known attribute value
        for ent in entities:
            for pred, val, _ in attr_map:
                if ent == val:
                    predicate, value = pred, val
                    break

        # Fallback: keyword scan
        if predicate is None:
            for pred, val, keywords in attr_map:
                if any(kw in cleaned for kw in keywords):
                    predicate, value = pred, val
                    break

        if predicate is None or value is None:
            return ChatResponse(
                message="I can filter destinations by season (Winter, Summer, "
                        "Monsoon) or budget (Low, Moderate, High). "
                        "Try: 'Best places in Winter'.",
                intent=Intent.ATTRIBUTE,
                matched_entities=entities,
            )

        matches = self._net.get_destinations_by_attribute(predicate, value)

        if not matches:
            return ChatResponse(
                message=f"No destinations found for {predicate}={value}.",
                intent=Intent.ATTRIBUTE,
                matched_entities=entities,
            )

        # Build a bullet-point list and pick a friendly template
        items = "\n".join(f"  \U00002B50 {m}" for m in matches)
        friendly_attr = value  # e.g. "Winter", "Low Budget"
        message = random.choice(_ATTRIBUTE_TEMPLATES).format(
            attr=friendly_attr, list=items,
        )

        return ChatResponse(
            message=message,
            reasoning_path=[(m, predicate, value) for m in matches],
            intent=Intent.ATTRIBUTE,
            matched_entities=entities,
        )

    def _handle_related(
        self, cleaned: str, entities: list[str],
    ) -> ChatResponse:
        """Handle similarity queries (e.g. 'Places similar to Goa').

        Uses ``_RELATED_TEMPLATES`` with ``random.choice()`` for a
        friendly presentation.
        """
        if not entities:
            return self._no_entity_response(Intent.RELATED)

        target = entities[0]
        try:
            related = self._net.get_related_destinations(target)
        except KeyError:
            return ChatResponse(
                message=f"'{target}' is not a known destination.",
                intent=Intent.RELATED,
                matched_entities=entities,
            )

        if not related:
            return ChatResponse(
                message=f"I couldn't find destinations similar to {target}.",
                intent=Intent.RELATED,
                matched_entities=entities,
            )

        # Build a bullet-point list with shared features
        lines: list[str] = []
        for dest, shared in related[:7]:
            shared_str = format_list_naturally(shared)
            lines.append(f"  \U0001F4CD {dest}  (shared: {shared_str})")
        list_str = "\n".join(lines)

        message = random.choice(_RELATED_TEMPLATES).format(
            place=target, list=list_str,
        )

        return ChatResponse(
            message=message,
            intent=Intent.RELATED,
            matched_entities=entities,
        )

    def _handle_unknown(
        self, cleaned: str, entities: list[str],
    ) -> ChatResponse:
        """Fallback when intent cannot be determined."""
        if entities:
            # Default: attempt forward reasoning on the first entity
            return self._handle_forward(cleaned, entities)

        return ChatResponse(
            message=(
                "I'm your Smart India Travel Guide! Try asking:\n"
                "  - 'Tell me about Goa'\n"
                "  - 'Where can I do Trekking?'\n"
                "  - 'Why visit Manali for Better Health?'\n"
                "  - 'Best places in Winter'\n"
                "  - 'Places similar to Kerala'"
            ),
            intent=Intent.UNKNOWN,
            matched_entities=entities,
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_small_talk(cleaned: str) -> ChatResponse | None:
        """Return a friendly reply if *cleaned* is a simple greeting.

        Scans the ``_SMALL_TALK`` dictionary for a keyword match.
        Returns ``None`` when the input is **not** small talk so the
        caller can continue with the full NLP pipeline.
        """
        # Strip to the bare words so "hi!" or "hey there" still match
        words = cleaned.split()
        if not words:
            return None

        # Try the full cleaned text first, then just the first word
        for candidate in (cleaned, words[0]):
            if candidate in _SMALL_TALK:
                reply = random.choice(_SMALL_TALK[candidate])
                logger.debug("Small-talk hit on %r", candidate)
                return ChatResponse(
                    message=reply,
                    intent=Intent.UNKNOWN,
                )
        return None

    @staticmethod
    def _no_entity_response(intent: Intent) -> ChatResponse:
        """Return a helpful prompt when no entities were extracted."""
        return ChatResponse(
            message="I couldn't identify a specific place or activity in "
                    "your message. Could you mention a destination (e.g. "
                    "Goa, Manali) or an activity (e.g. Trekking, Scuba "
                    "Diving)?",
            intent=intent,
        )

    def __repr__(self) -> str:
        return (
            f"ChatProcessor(network={self._net!r}, "
            f"matchable_entities={len(self._node_lookup)})"
        )


# ======================================================================
#  Module-level Demo
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    proc = ChatProcessor()
    separator = "=" * 64

    demo_queries = [
        "Tell me about Goa",
        "Where can I do Trekking?",
        "Why visit Manali for Better Health?",
        "Best places to visit in Winter",
        "Places similar to Kerala",
        "What does Varanasi offer?",
        "Where can I find Scuba Diving?",
        "Hello, what can you do?",
    ]

    print(f"\n{separator}")
    print("  CHAT PROCESSOR  -  Demo Run")
    print(separator)

    for query in demo_queries:
        print(f"\n  USER: {query}")
        print("  " + "-" * 60)
        resp = proc.process(query)
        print(f"  Intent  : {resp.intent.name}")
        print(f"  Entities: {resp.matched_entities}")
        print(f"  Path    : {len(resp.reasoning_path)} hop(s)")
        print(f"\n{resp.message}")
        print(separator)
