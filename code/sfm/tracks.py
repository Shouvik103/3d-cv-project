"""Feature tracks built from geometrically verified pairwise matches."""

from __future__ import annotations

from collections import defaultdict


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        if node not in self.parent:
            self.parent[node] = node
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def build_tracks(verified_matches: dict[tuple[int, int], object]) -> dict[int, dict[int, int]]:
    """Return tracks after discarding components with duplicate views."""
    uf = UnionFind()
    for (left, right), pairs in verified_matches.items():
        for a, b in pairs:
            uf.union((left, int(a)), (right, int(b)))
    components: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for node in uf.parent:
        components[uf.find(node)].append(node)
    tracks: dict[int, dict[int, int]] = {}
    for nodes in components.values():
        views = {image for image, _ in nodes}
        if len(views) != len(nodes) or len(nodes) < 2:
            continue
        tracks[len(tracks)] = {image: feature for image, feature in nodes}
    return tracks
