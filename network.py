"""
network.py — Semantic Network Reasoning Engine
=================================================
Smart India Travel Guide Chatbot

This module implements the core graph-based reasoning engine that powers
the chatbot.  It consumes raw relationship triples from ``data.py`` and
constructs an in-memory directed graph (adjacency-list representation).
Graph traversal algorithms — primarily Breadth-First Search (BFS) — are
then used to perform **multi-hop symbolic inference** without any external
LLM or API dependency.

Key Capabilities
-----------------
* **Forward Reasoning** — Given a destination, traverse outward to
  discover every feature, activity, requirement, and outcome it offers.
* **Reverse Reasoning** — Given an activity or concept, walk backward
  along inverted edges to locate all destinations that ultimately
  provide it.
* **Path Finding** — Compute the shortest reasoning chain between any
  two nodes, returning both the nodes visited and the predicates that
  connect them — i.e. the *explainable "Why"*.
* **Neighbor Inspection** — Retrieve the immediate outgoing or incoming
  connections for any node.

Design Decisions
-----------------
* The graph is stored as **two adjacency dictionaries** — one for
  forward edges and one for inverted (reverse) edges — so both
  directions of traversal run in O(V + E) without on-the-fly inversion.
* Every edge carries its predicate label, enabling the engine to
  produce human-readable reasoning traces (e.g.
  ``Goa ─has→ Beaches ─enable→ Water Sports``).
* All public methods use **Python type hints** (``typing`` module) and
  follow PEP-257 docstring conventions.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Optional

from data import RELATIONSHIPS

# ──────────────────────────────────────────────────────────────────────
#  Module-level logger
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Type Aliases
# ──────────────────────────────────────────────────────────────────────
Triple = tuple[str, str, str]
"""A single (Subject, Predicate, Object) relationship triple."""

Edge = tuple[str, str]
"""An adjacency-list entry: (neighbor_node, predicate_label)."""

ReasoningChain = list[tuple[str, str, str]]
"""
An ordered sequence of (source, predicate, target) hops that forms a
human-readable explanation path through the graph.
"""


# ======================================================================
#  SemanticNetwork Class
# ======================================================================

class SemanticNetwork:
    """A directed, labeled graph built from (Subject, Predicate, Object)
    triples, with BFS-powered symbolic reasoning methods.

    The network maintains two parallel adjacency lists:

    * ``_forward_graph``  — maps each node to its **outgoing** edges.
    * ``_reverse_graph``  — maps each node to its **incoming** edges
      (edges stored with the *inverted* predicate direction).

    This dual-graph design allows both forward and reverse multi-hop
    traversal in optimal O(V + E) time.

    Parameters
    ----------
    triples : list[Triple], optional
        A list of ``(Subject, Predicate, Object)`` tuples.  If *None*,
        the network auto-loads ``data.RELATIONSHIPS`` at construction
        time.

    Examples
    --------
    >>> net = SemanticNetwork()          # loads default data
    >>> net.forward_reasoning("Goa")     # what does Goa offer?
    >>> net.reverse_reasoning("Trekking")  # which places have trekking?
    >>> net.find_path("Manali", "Better Health")  # reasoning chain
    """

    # ------------------------------------------------------------------
    #  Construction & Graph Building
    # ------------------------------------------------------------------

    def __init__(self, triples: Optional[list[Triple]] = None) -> None:
        """Initialize the semantic network and build internal graphs.

        Parameters
        ----------
        triples : list[Triple] or None
            Raw relationship data.  Defaults to ``data.RELATIONSHIPS``
            when *None*.
        """
        self._forward_graph: dict[str, list[Edge]] = defaultdict(list)
        self._reverse_graph: dict[str, list[Edge]] = defaultdict(list)
        self._all_nodes: set[str] = set()
        self._all_predicates: set[str] = set()
        self._triple_count: int = 0

        source_triples: list[Triple] = triples if triples is not None else RELATIONSHIPS
        self._build_graph(source_triples)
        logger.info(
            "SemanticNetwork initialized — %d triples, %d nodes, %d predicate types",
            self._triple_count,
            len(self._all_nodes),
            len(self._all_predicates),
        )

    def _build_graph(self, triples: list[Triple]) -> None:
        """Populate forward and reverse adjacency lists from raw triples.

        For every triple ``(S, P, O)`` two edges are created:

        * **Forward**: ``S → O``  (labeled *P*)
        * **Reverse**: ``O → S``  (labeled *P*, direction inverted)

        Parameters
        ----------
        triples : list[Triple]
            The source relationship data to ingest.

        Raises
        ------
        ValueError
            If any triple does not contain exactly three elements.
        """
        for triple in triples:
            if len(triple) != 3:
                raise ValueError(
                    f"Expected a 3-element triple, got {len(triple)} elements: {triple}"
                )

            subject, predicate, obj = triple

            # Forward edge: Subject ──predicate──▶ Object
            self._forward_graph[subject].append((obj, predicate))

            # Reverse edge: Object ──predicate──▶ Subject  (inverted direction)
            self._reverse_graph[obj].append((subject, predicate))

            # Book-keeping
            self._all_nodes.update((subject, obj))
            self._all_predicates.add(predicate)
            self._triple_count += 1

    # ------------------------------------------------------------------
    #  Graph Statistics & Inspection
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Return the total number of unique nodes in the graph."""
        return len(self._all_nodes)

    @property
    def edge_count(self) -> int:
        """Return the total number of directed edges (triples) ingested."""
        return self._triple_count

    @property
    def predicate_types(self) -> set[str]:
        """Return the set of unique predicate labels used in the graph."""
        return set(self._all_predicates)

    @property
    def all_nodes(self) -> set[str]:
        """Return a copy of all unique node identifiers."""
        return set(self._all_nodes)

    def has_node(self, node: str) -> bool:
        """Check whether a node exists in the network.

        Parameters
        ----------
        node : str
            The node identifier to look up.

        Returns
        -------
        bool
            *True* if the node is present, *False* otherwise.
        """
        return node in self._all_nodes

    def get_outgoing_edges(self, node: str) -> list[Edge]:
        """Return all outgoing (forward) edges from *node*.

        Parameters
        ----------
        node : str
            The source node.

        Returns
        -------
        list[Edge]
            A list of ``(neighbor, predicate)`` tuples.  Returns an
            empty list if the node has no outgoing edges or does not
            exist.
        """
        return list(self._forward_graph.get(node, []))

    def get_incoming_edges(self, node: str) -> list[Edge]:
        """Return all incoming (reverse) edges leading *to* a node.

        Parameters
        ----------
        node : str
            The target node.

        Returns
        -------
        list[Edge]
            A list of ``(source_node, predicate)`` tuples.  Returns an
            empty list if no edges point to the node.
        """
        return list(self._reverse_graph.get(node, []))

    # ------------------------------------------------------------------
    #  Core BFS Traversal
    # ------------------------------------------------------------------

    def _bfs_traversal(
        self,
        start: str,
        graph: dict[str, list[Edge]],
        max_depth: Optional[int] = None,
    ) -> dict[str, list[tuple[str, str, str]]]:
        """Perform a Breadth-First Search from *start* over *graph*.

        This is the **foundational traversal method** used by all
        higher-level reasoning functions.  It records, for every
        reachable node, the complete chain of ``(source, predicate,
        target)`` hops from *start* to that node.

        Parameters
        ----------
        start : str
            The node from which traversal begins.
        graph : dict[str, list[Edge]]
            The adjacency list to traverse (forward or reverse).
        max_depth : int or None
            Maximum number of hops from *start*.  ``None`` means
            unlimited (traverse the full reachable component).

        Returns
        -------
        dict[str, list[tuple[str, str, str]]]
            A mapping from each **visited node** to the ordered list of
            ``(from_node, predicate, to_node)`` edges forming the
            shortest path from *start* to that node.

        Notes
        -----
        The BFS guarantees that the *first* path discovered to any node
        is the **shortest** (fewest hops).  The algorithm runs in
        O(V + E) time and O(V) space.
        """
        if start not in self._all_nodes:
            logger.warning("BFS start node '%s' not found in the network.", start)
            return {}

        # visited maps each reached node to its reasoning chain
        visited: dict[str, list[tuple[str, str, str]]] = {start: []}
        queue: deque[tuple[str, int]] = deque([(start, 0)])

        while queue:
            current_node, depth = queue.popleft()

            # Depth guard
            if max_depth is not None and depth >= max_depth:
                continue

            for neighbor, predicate in graph.get(current_node, []):
                if neighbor not in visited:
                    # Build reasoning chain by extending parent's chain
                    chain = visited[current_node] + [
                        (current_node, predicate, neighbor)
                    ]
                    visited[neighbor] = chain
                    queue.append((neighbor, depth + 1))

        return visited

    # ------------------------------------------------------------------
    #  Forward Reasoning  (What does a place offer?)
    # ------------------------------------------------------------------

    def forward_reasoning(
        self,
        entity: str,
        max_depth: Optional[int] = None,
    ) -> dict[str, ReasoningChain]:
        """Discover everything reachable *from* an entity via forward edges.

        Starting at *entity*, the method performs a BFS along outgoing
        edges and returns every reachable concept along with the
        reasoning chain that connects it back to the source.

        This answers the question: **"What does <entity> offer?"**

        Parameters
        ----------
        entity : str
            The starting node (typically a destination name like
            ``"Goa"`` or ``"Manali"``).
        max_depth : int or None
            Maximum traversal depth.  ``None`` = unlimited.

        Returns
        -------
        dict[str, ReasoningChain]
            Keys are all reachable nodes; values are the ordered
            reasoning chains from *entity* to that node.

        Raises
        ------
        KeyError
            If *entity* is not a recognized node in the network.

        Examples
        --------
        >>> net = SemanticNetwork()
        >>> results = net.forward_reasoning("Goa")
        >>> for node, chain in results.items():
        ...     arrows = " → ".join(
        ...         f"{s} ─{p}→ {t}" for s, p, t in chain
        ...     )
        ...     print(f"  {node}: {arrows}")
        """
        self._validate_node(entity)
        logger.debug("Forward reasoning from '%s' (max_depth=%s)", entity, max_depth)

        results = self._bfs_traversal(entity, self._forward_graph, max_depth)

        # Remove the start node itself from the result set
        results.pop(entity, None)
        return results

    # ------------------------------------------------------------------
    #  Reverse Reasoning  (What places offer a specific thing?)
    # ------------------------------------------------------------------

    def reverse_reasoning(
        self,
        entity: str,
        max_depth: Optional[int] = None,
    ) -> dict[str, ReasoningChain]:
        """Discover every node that eventually *leads to* an entity.

        Starting at *entity*, the method performs a BFS along **reversed**
        edges, walking backward through the knowledge graph.

        This answers the question:
        **"What places / concepts lead to <entity>?"**

        Parameters
        ----------
        entity : str
            The target concept (e.g. ``"Trekking"``, ``"Low Budget"``).
        max_depth : int or None
            Maximum traversal depth.  ``None`` = unlimited.

        Returns
        -------
        dict[str, ReasoningChain]
            Keys are all nodes from which *entity* is reachable; values
            are the **reverse** reasoning chains.

        Raises
        ------
        KeyError
            If *entity* is not a recognized node in the network.

        Examples
        --------
        >>> net = SemanticNetwork()
        >>> sources = net.reverse_reasoning("Trekking")
        >>> for src in sources:
        ...     print(src)   # e.g. "Mountains", "Manali", "Munnar", …
        """
        self._validate_node(entity)
        logger.debug("Reverse reasoning to '%s' (max_depth=%s)", entity, max_depth)

        results = self._bfs_traversal(entity, self._reverse_graph, max_depth)

        # Remove the start node itself from the result set
        results.pop(entity, None)
        return results

    # ------------------------------------------------------------------
    #  Path Finding  (The "Why")
    # ------------------------------------------------------------------

    def find_path(
        self,
        start_node: str,
        end_node: str,
    ) -> Optional[ReasoningChain]:
        """Find the shortest reasoning path between two nodes.

        Uses BFS on the forward graph to locate the shortest hop
        sequence from *start_node* to *end_node*.  The returned chain
        serves as the **explainable "Why"** — a step-by-step logical
        justification linking the two concepts.

        Parameters
        ----------
        start_node : str
            The origin node.
        end_node : str
            The destination node.

        Returns
        -------
        ReasoningChain or None
            An ordered list of ``(source, predicate, target)`` hops
            forming the shortest path.  Returns ``None`` if no path
            exists between the two nodes.

        Raises
        ------
        KeyError
            If either *start_node* or *end_node* is not present in the
            network.

        Examples
        --------
        >>> net = SemanticNetwork()
        >>> path = net.find_path("Manali", "Better Health")
        >>> if path:
        ...     for src, pred, tgt in path:
        ...         print(f"  {src} ─{pred}→ {tgt}")
        # Output:
        #   Manali ─has→ Mountains
        #   Mountains ─enable→ Trekking
        #   Trekking ─requires→ High Fitness
        #   High Fitness ─improves→ Better Health
        """
        self._validate_node(start_node)
        self._validate_node(end_node)

        if start_node == end_node:
            logger.debug("find_path: start and end are identical ('%s').", start_node)
            return []

        logger.debug("Finding path: '%s' → '%s'", start_node, end_node)
        visited = self._bfs_traversal(start_node, self._forward_graph)

        if end_node in visited:
            chain = visited[end_node]
            logger.debug(
                "Path found (%d hops): %s",
                len(chain),
                " → ".join(f"{s} ─{p}→ {t}" for s, p, t in chain),
            )
            return chain

        logger.debug("No path exists from '%s' to '%s'.", start_node, end_node)
        return None

    # ------------------------------------------------------------------
    #  Filtered / Specialized Queries
    # ------------------------------------------------------------------

    def get_destinations_by_attribute(
        self,
        predicate: str,
        value: str,
    ) -> list[str]:
        """Find all destinations matching a specific attribute value.

        Scans the reverse graph at *value* for edges whose predicate
        matches *predicate*, returning the list of source nodes.

        Parameters
        ----------
        predicate : str
            The relationship type to filter on (e.g. ``"best_season"``,
            ``"budget_level"``, ``"type"``).
        value : str
            The desired attribute value (e.g. ``"Winter"``,
            ``"Low Budget"``, ``"Adventure Destination"``).

        Returns
        -------
        list[str]
            Destination names (or any nodes) that have the specified
            ``(predicate, value)`` relationship.

        Examples
        --------
        >>> net = SemanticNetwork()
        >>> net.get_destinations_by_attribute("best_season", "Winter")
        ['Goa', 'Jaipur', 'Kerala', 'Varanasi', ...]
        """
        matches: list[str] = []
        for source_node, edge_predicate in self._reverse_graph.get(value, []):
            if edge_predicate == predicate:
                matches.append(source_node)

        logger.debug(
            "Attribute query [%s = %s] → %d result(s)",
            predicate,
            value,
            len(matches),
        )
        return matches

    def get_related_destinations(
        self,
        destination: str,
        max_shared: Optional[int] = None,
    ) -> list[tuple[str, list[str]]]:
        """Find destinations that share features with *destination*.

        The method collects all Level-1 outgoing neighbors of
        *destination*, then checks which other destinations also connect
        to those same nodes — revealing conceptual similarity.

        Parameters
        ----------
        destination : str
            The reference destination (e.g. ``"Goa"``).
        max_shared : int or None
            If given, only return destinations sharing at least
            *max_shared* common neighbors.

        Returns
        -------
        list[tuple[str, list[str]]]
            A list of ``(other_destination, shared_features)`` tuples,
            sorted by number of shared features (descending).

        Raises
        ------
        KeyError
            If *destination* is not in the network.

        Examples
        --------
        >>> net = SemanticNetwork()
        >>> related = net.get_related_destinations("Goa")
        >>> for dest, shared in related:
        ...     print(f"  {dest}: {shared}")
        """
        self._validate_node(destination)

        # Collect direct forward neighbors of the reference destination
        my_neighbors: set[str] = {
            neighbor for neighbor, _ in self._forward_graph.get(destination, [])
        }

        # For each neighbor, find other nodes that also connect to it
        overlap_map: dict[str, list[str]] = defaultdict(list)
        for neighbor in my_neighbors:
            for source, _ in self._reverse_graph.get(neighbor, []):
                if source != destination:
                    overlap_map[source].append(neighbor)

        # Filter and sort
        threshold = max_shared if max_shared is not None else 1
        results = [
            (dest, sorted(shared))
            for dest, shared in overlap_map.items()
            if len(shared) >= threshold
        ]
        results.sort(key=lambda item: len(item[1]), reverse=True)

        logger.debug(
            "Related destinations for '%s': %d found",
            destination,
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    #  Human-Readable Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_chain(chain: ReasoningChain) -> str:
        """Convert a reasoning chain into a human-readable arrow string.

        Parameters
        ----------
        chain : ReasoningChain
            An ordered list of ``(source, predicate, target)`` hops.

        Returns
        -------
        str
            A formatted string such as:
            ``Manali ─has→ Mountains ─enable→ Trekking``
        """
        if not chain:
            return "(empty chain)"

        parts: list[str] = [chain[0][0]]  # Start with the first source
        for source, predicate, target in chain:
            parts.append(f"─{predicate}→ {target}")
        return " ".join(parts)

    @staticmethod
    def format_results_table(
        results: dict[str, ReasoningChain],
        title: str = "Reasoning Results",
    ) -> str:
        """Format a reasoning result set as a readable multi-line table.

        Parameters
        ----------
        results : dict[str, ReasoningChain]
            Output from ``forward_reasoning`` or ``reverse_reasoning``.
        title : str
            A header title for the table.

        Returns
        -------
        str
            A nicely formatted string suitable for console output.
        """
        separator = "=" * 64
        lines: list[str] = [separator, f"  {title}", separator]

        if not results:
            lines.append("  (no results)")
        else:
            for idx, (node, chain) in enumerate(results.items(), start=1):
                depth = len(chain)
                chain_str = SemanticNetwork.format_chain(chain)
                lines.append(f"  {idx:>3}. [{depth}-hop] {node}")
                lines.append(f"       Path: {chain_str}")

        lines.append(separator)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Internal Helpers
    # ------------------------------------------------------------------

    def _validate_node(self, node: str) -> None:
        """Raise ``KeyError`` if *node* is not in the network.

        Parameters
        ----------
        node : str
            The node identifier to validate.

        Raises
        ------
        KeyError
            With a descriptive message listing the node that was not
            found.
        """
        if node not in self._all_nodes:
            raise KeyError(
                f"Node '{node}' not found in the semantic network.  "
                f"Available nodes ({len(self._all_nodes)}): "
                f"{sorted(self._all_nodes)[:10]}…"
            )

    # ------------------------------------------------------------------
    #  Dunder Methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"SemanticNetwork(nodes={self.node_count}, "
            f"edges={self.edge_count}, "
            f"predicates={len(self._all_predicates)})"
        )

    def __contains__(self, node: str) -> bool:
        """Support ``in`` operator for node membership checks.

        Examples
        --------
        >>> net = SemanticNetwork()
        >>> "Goa" in net
        True
        """
        return self.has_node(node)

    def __len__(self) -> int:
        """Return the number of nodes when ``len()`` is called."""
        return self.node_count


# ======================================================================
#  Module-level Demonstration
# ======================================================================

if __name__ == "__main__":
    # Configure basic logging for demo output
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 64)
    print("  SEMANTIC NETWORK — Interactive Demo")
    print("=" * 64)

    # 1. Build the network
    net = SemanticNetwork()
    print(f"\n  {net!r}\n")

    # ── Forward Reasoning Demo ──────────────────────────────────────
    print("-" * 64)
    print("  FORWARD REASONING: What does Goa offer?")
    print("-" * 64)
    fwd_results = net.forward_reasoning("Goa", max_depth=3)
    print(SemanticNetwork.format_results_table(fwd_results, "Goa — Forward Reasoning"))

    # ── Reverse Reasoning Demo ──────────────────────────────────────
    print("\n" + "-" * 64)
    print("  REVERSE REASONING: What places offer Trekking?")
    print("-" * 64)
    rev_results = net.reverse_reasoning("Trekking", max_depth=3)
    print(SemanticNetwork.format_results_table(rev_results, "Trekking — Reverse Reasoning"))

    # ── Path Finding Demo ───────────────────────────────────────────
    print("\n" + "-" * 64)
    print("  PATH FINDING: Manali → Better Health")
    print("-" * 64)
    path = net.find_path("Manali", "Better Health")
    if path:
        print(f"  Chain ({len(path)} hops):")
        print(f"    {SemanticNetwork.format_chain(path)}")
    else:
        print("  No path found.")

    # ── Attribute Query Demo ────────────────────────────────────────
    print("\n" + "-" * 64)
    print("  ATTRIBUTE QUERY: Destinations best in Winter")
    print("-" * 64)
    winter_spots = net.get_destinations_by_attribute("best_season", "Winter")
    for spot in winter_spots:
        print(f"    • {spot}")

    # ── Related Destinations Demo ───────────────────────────────────
    print("\n" + "-" * 64)
    print("  RELATED DESTINATIONS: Similar to Goa")
    print("-" * 64)
    related = net.get_related_destinations("Goa")
    for dest, shared in related[:5]:
        print(f"    • {dest} — shared: {', '.join(shared)}")

    print("\n" + "=" * 64)
    print("  Demo complete.")
    print("=" * 64 + "\n")
