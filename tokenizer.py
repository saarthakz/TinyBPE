import json
from pathlib import Path
from typing import List, Tuple, Dict, Set

from avl_tree import AVLTree
from linked_list import DoublyLinkedList, Node


class Tokenizer:
    """A Byte Pair Encoding (BPE) tokenizer.

    Supports training on a corpus of text, encoding text into token IDs,
    and decoding token IDs back into text.

    Data structures:
        - byte_pair_map: Ordered record of merge rules learned during training.
          Maps (token_a, token_b) -> merged_token_id.
        - vocab: Maps each token ID to the byte sequence it represents.
          IDs 0-255 are raw byte tokens; IDs >= 256 are merged tokens.
    """

    def __init__(self):
        self.byte_pair_map: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, List[int]] = {}

    def _build_vocab(self):
        """Reconstruct the vocab from byte_pair_map.

        Iterates in merge order so that when building the byte sequence for
        a merged token, its constituent tokens are already in the vocab.
        """
        self.vocab = {idx: [idx] for idx in range(256)}
        for byte_pair, token_val in self.byte_pair_map.items():
            first, second = byte_pair
            self.vocab[token_val] = self.vocab[first] + self.vocab[second]

    def save(self, path: str):
        """Persist the learned tokenizer state as JSON."""
        payload = {
            "byte_pair_map": [
                {"pair": [first, second], "token": token}
                for (first, second), token in self.byte_pair_map.items()
            ],
            "vocab": {str(token): byte_seq for token, byte_seq in self.vocab.items()},
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load a tokenizer previously saved with save()."""
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)

        tokenizer = cls()
        tokenizer.byte_pair_map = {
            tuple(item["pair"]): item["token"] for item in payload["byte_pair_map"]
        }
        tokenizer.vocab = {
            int(token): byte_seq for token, byte_seq in payload["vocab"].items()
        }
        return tokenizer

    # ── Shared helpers ──

    @staticmethod
    def _build_dll(tokens: List[int]) -> DoublyLinkedList:
        """Convert a flat list of token IDs into a doubly linked list."""
        dll = DoublyLinkedList()
        for tok in tokens:
            dll.insert(tok)
        return dll

    @staticmethod
    def _build_cache_map(dll: DoublyLinkedList) -> Dict[Tuple[int, int], Set[Node]]:
        """Walk the DLL and build the cache map.

        For each consecutive pair (node.val, node.next.val), stores the first
        node in the set for that pair. This allows O(1) lookup of all positions
        where a given pair occurs, and the pair's frequency is len(set).
        """
        cache_map: Dict[Tuple[int, int], Set[Node]] = {}
        itr = iter(dll)
        idx = 0
        while idx < dll.length:
            curr_node = next(itr)
            next_node = curr_node.next
            idx += 1
            if next_node is None:
                break
            pair = (curr_node.val, next_node.val)
            if pair not in cache_map:
                cache_map[pair] = {curr_node}
            else:
                cache_map[pair].add(curr_node)
        return cache_map

    @staticmethod
    def _dll_to_list(dll: DoublyLinkedList) -> List[int]:
        """Walk the DLL and collect all node values into a flat list."""
        output: List[int] = []
        itr = iter(dll)
        idx = 0
        while idx < dll.length:
            output.append(next(itr).val)
            idx += 1
        return output

    # ── Encode helpers (cache map only, no AVL tree) ──

    @staticmethod
    def _invalidate_pair(
        cache_map: Dict[Tuple[int, int], Set[Node]],
        pair: Tuple[int, int],
        node: Node,
    ):
        """Remove a node from a pair's set in the cache map.

        When a merge replaces two nodes, the old neighbor pairs become stale.
        This removes the specific node from the pair's set and cleans up
        empty entries.
        """
        node_set = cache_map[pair]
        node_set.remove(node)
        if len(node_set) == 0:
            del cache_map[pair]

    @staticmethod
    def _add_pair(
        cache_map: Dict[Tuple[int, int], Set[Node]],
        pair: Tuple[int, int],
        node: Node,
    ):
        """Add a node to a pair's set in the cache map.

        After a merge creates a new_node, it forms new pairs with its
        neighbors. This registers those new pairs (or increments their
        frequency if they already exist).
        """
        if pair not in cache_map:
            cache_map[pair] = {node}
        else:
            cache_map[pair].add(node)

    @staticmethod
    def _merge_pair(
        dll: DoublyLinkedList,
        cache_map: Dict[Tuple[int, int], Set[Node]],
        tok_tuple: Tuple[int, int],
        new_token: int,
    ):
        """Merge all occurrences of tok_tuple in the DLL into new_token.

        For each occurrence (first_node, second_node) of the pair:
          1. Create a new_node with value new_token
          2. Splice new_node into the DLL in place of first_node and second_node
          3. Invalidate old neighbor pairs (prev, first) and (second, next)
          4. Register new neighbor pairs (prev, new_node) and (new_node, next)
          5. Update DLL head/tail if the pair is at a boundary
          6. Detach old nodes to mark them as consumed
        """
        curr_tok_pair_nodes = cache_map[tok_tuple]

        # .copy() because neighbour invalidation can .remove() from this same set
        # when a neighbour pair equals tok_tuple, which would mutate the set during iteration.
        for first_node in curr_tok_pair_nodes.copy():
            second_node = first_node.next

            # A prior merge in this loop may have already consumed/detached this node
            # (e.g. overlapping pairs in [A,A,A]: merging node0+node1 detaches node1,
            # but node1 is still in the .copy() snapshot as a "first node" candidate).
            if second_node is None or (first_node.val, second_node.val) != tok_tuple:
                continue

            new_node = Node(new_token)

            # ── Splice left side ──
            if first_node.prev is not None:
                prev_to_first = first_node.prev
                new_node.prev = prev_to_first
                prev_to_first.next = new_node

                # Old pair (prev, first) is gone; new pair (prev, new_node) is formed
                Tokenizer._invalidate_pair(cache_map, (prev_to_first.val, first_node.val), prev_to_first)
                Tokenizer._add_pair(cache_map, (prev_to_first.val, new_node.val), prev_to_first)
            else:
                dll.head = new_node

            # ── Splice right side ──
            if second_node.next is not None:
                next_to_second = second_node.next
                new_node.next = next_to_second
                next_to_second.prev = new_node

                # Old pair (second, next) is gone; new pair (new_node, next) is formed
                Tokenizer._invalidate_pair(cache_map, (second_node.val, next_to_second.val), second_node)
                Tokenizer._add_pair(cache_map, (new_node.val, next_to_second.val), new_node)
            else:
                dll.tail = new_node

            # Detach old nodes — marks them as consumed so the stale node guard
            # (the check at the top of this loop) can skip them.
            first_node.next = None
            first_node.prev = None
            second_node.next = None
            second_node.prev = None

            dll.length -= 1

        if tok_tuple in cache_map:
            del cache_map[tok_tuple]

    # ── Train helpers (cache map + AVL tree) ──

    @staticmethod
    def _invalidate_pair_tracked(
        cache_map: Dict[Tuple[int, int], Set[Node]],
        pair: Tuple[int, int],
        node: Node,
        ordered_pairs: AVLTree[Tuple[int, Tuple[int, int]]],
    ):
        """Remove a node from a pair's set, updating the AVL tree.

        Same as _invalidate_pair but also maintains the AVL tree so that
        extract-max remains correct after the frequency change.
        """
        node_set = cache_map[pair]
        old_freq = len(node_set)
        node_set.remove(node)
        new_freq = len(node_set)

        ordered_pairs.discard((old_freq, pair))
        if new_freq > 0:
            ordered_pairs.add((new_freq, pair))
        else:
            del cache_map[pair]

    @staticmethod
    def _add_pair_tracked(
        cache_map: Dict[Tuple[int, int], Set[Node]],
        pair: Tuple[int, int],
        node: Node,
        ordered_pairs: AVLTree[Tuple[int, Tuple[int, int]]],
    ):
        """Add a node to a pair's set, updating the AVL tree.

        Same as _add_pair but also maintains the AVL tree so that
        extract-max remains correct after the frequency change.
        """
        if pair not in cache_map:
            cache_map[pair] = {node}
            ordered_pairs.add((1, pair))
        else:
            node_set = cache_map[pair]
            old_freq = len(node_set)
            node_set.add(node)
            ordered_pairs.discard((old_freq, pair))
            ordered_pairs.add((old_freq + 1, pair))

    @staticmethod
    def _merge_pair_tracked(
        dll: DoublyLinkedList,
        cache_map: Dict[Tuple[int, int], Set[Node]],
        tok_tuple: Tuple[int, int],
        new_token: int,
        ordered_pairs: AVLTree[Tuple[int, Tuple[int, int]]],
    ):
        """Merge all occurrences of tok_tuple, maintaining the AVL tree.

        Same splice/invalidate/register logic as _merge_pair, but calls
        the _tracked variants to keep the AVL tree in sync with frequency
        changes caused by the merge.
        """
        curr_tok_pair_nodes = cache_map[tok_tuple]

        for first_node in curr_tok_pair_nodes.copy():
            second_node = first_node.next

            if second_node is None or (first_node.val, second_node.val) != tok_tuple:
                continue

            new_node = Node(new_token)

            # ── Splice left side ──
            if first_node.prev is not None:
                prev_to_first = first_node.prev
                new_node.prev = prev_to_first
                prev_to_first.next = new_node

                Tokenizer._invalidate_pair_tracked(cache_map, (prev_to_first.val, first_node.val), prev_to_first, ordered_pairs)
                Tokenizer._add_pair_tracked(cache_map, (prev_to_first.val, new_node.val), prev_to_first, ordered_pairs)
            else:
                dll.head = new_node

            # ── Splice right side ──
            if second_node.next is not None:
                next_to_second = second_node.next
                new_node.next = next_to_second
                next_to_second.prev = new_node

                Tokenizer._invalidate_pair_tracked(cache_map, (second_node.val, next_to_second.val), second_node, ordered_pairs)
                Tokenizer._add_pair_tracked(cache_map, (new_node.val, next_to_second.val), new_node, ordered_pairs)
            else:
                dll.tail = new_node

            first_node.next = None
            first_node.prev = None
            second_node.next = None
            second_node.prev = None

            dll.length -= 1

        # Clean up the merged pair
        ordered_pairs.discard((len(cache_map.get(tok_tuple, set())), tok_tuple))
        if tok_tuple in cache_map:
            del cache_map[tok_tuple]

    # ── Train ──

    def train(self, texts: List[str], vocab_size: int, show_progress: bool = False):
        """Train the tokenizer on a list of text chunks using BPE.

        Concatenates all chunks into a single byte sequence, builds the DLL
        and cache map once, then performs merges using an AVL tree keyed by
        (frequency, pair) for O(log P) insert/delete/extract-max.

        Total time complexity: O(N log N) where N is the total byte count,
        compared to O(M * N) for the naive approach (M = number of merges).
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (the number of raw byte tokens)"

        progress_bar = None
        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError as exc:
                raise ImportError("show_progress=True requires tqdm to be installed") from exc

        num_merges = vocab_size - 256

        tokens: List[int] = []
        for text in texts:
            tokens.extend(text.encode("utf-8"))

        dll = self._build_dll(tokens)
        cache_map = self._build_cache_map(dll)

        # AVL tree keyed by (frequency, pair) for O(log P) max extraction.
        # Ties in frequency are broken by the pair tuple (arbitrary but deterministic).
        ordered_pairs: AVLTree[Tuple[int, Tuple[int, int]]] = AVLTree()
        for pair, node_set in cache_map.items():
            ordered_pairs.add((len(node_set), pair))

        self.byte_pair_map = {}

        if show_progress:
            progress_bar = tqdm(total=num_merges, desc="Training merges", unit="merge")

        for idx in range(num_merges):
            if not ordered_pairs:
                break

            # Pop the highest-frequency pair
            freq, pair = ordered_pairs.pop_max()
            new_token = 256 + idx
            self.byte_pair_map[pair] = new_token

            # Merge all occurrences — updates DLL, cache map, and AVL tree in place
            self._merge_pair_tracked(dll, cache_map, pair, new_token, ordered_pairs)
            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        self._build_vocab()

    # ── Decode ──

    def decode(self, tokens: List[int]) -> str:
        """Given a list of token IDs, return the decoded text string.

        Each token ID is expanded to its byte sequence via the vocab,
        then the concatenated bytes are decoded as UTF-8.
        """
        decoded_tokens: List[int] = []
        for token in tokens:
            decoded_tokens.extend(self.vocab[token])
        return bytes(decoded_tokens).decode("utf-8", errors="replace")

    # ── Encode ──

    def encode(self, text: str) -> List[int]:
        """Given a string, return the list of token IDs.

        Converts text to UTF-8 bytes, builds the DLL and cache map, then
        applies each merge from byte_pair_map in order. No AVL tree is
        needed here since the merge order is already determined by training.
        """
        byte_tokens = list(text.encode("utf-8"))
        dll = self._build_dll(byte_tokens)
        cache_map = self._build_cache_map(dll)

        for tok_tuple, new_token in self.byte_pair_map.items():
            if tok_tuple in cache_map:
                self._merge_pair(dll, cache_map, tok_tuple, new_token)

        return self._dll_to_list(dll)
