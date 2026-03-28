from typing import List, Tuple, Dict, Set
from linked_list import DoublyLinkedList, Node


class Tokenizer:
    def __init__(self):
        self.byte_pair_map: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, List[int]] = {}

    @staticmethod
    def _get_stats(tokens: List[int]) -> Dict[Tuple[int, int], int]:
        byte_pair_freq_map: Dict[Tuple[int, int], int] = {}
        for idx in range(len(tokens) - 1):
            pair = (tokens[idx], tokens[idx + 1])
            byte_pair_freq_map[pair] = byte_pair_freq_map.get(pair, 0) + 1
        return byte_pair_freq_map

    @staticmethod
    def _replace_pair_with_token(tokens: List[int], max_byte_pair: Tuple[int, int], new_token: int) -> List[int]:
        new_tokens = []
        idx = 0
        while idx < len(tokens):
            if idx != len(tokens) - 1 and (tokens[idx], tokens[idx + 1]) == max_byte_pair:
                new_tokens.append(new_token)
                idx += 2
            else:
                new_tokens.append(tokens[idx])
                idx += 1
        return new_tokens

    @staticmethod
    def _get_stats_multi(chunks: List[List[int]]) -> Dict[Tuple[int, int], int]:
        byte_pair_freq_map: Dict[Tuple[int, int], int] = {}
        for tokens in chunks:
            for idx in range(len(tokens) - 1):
                pair = (tokens[idx], tokens[idx + 1])
                byte_pair_freq_map[pair] = byte_pair_freq_map.get(pair, 0) + 1
        return byte_pair_freq_map

    def train(self, texts: List[str], vocab_size: int):
        assert vocab_size >= 256, "vocab_size must be at least 256 (the number of raw byte tokens)"
        num_merges = vocab_size - 256

        chunks = [list(text.encode("utf-8")) for text in texts]
        self.byte_pair_map = {}

        for idx in range(num_merges):
            byte_pair_freq_map = self._get_stats_multi(chunks)
            if not byte_pair_freq_map:
                break
            pair = max(byte_pair_freq_map, key=byte_pair_freq_map.get)
            new_token = 256 + idx
            chunks = [self._replace_pair_with_token(chunk, pair, new_token) for chunk in chunks]
            self.byte_pair_map[pair] = new_token

        self.vocab = {idx: [idx] for idx in range(256)}
        for byte_pair, token_val in self.byte_pair_map.items():
            first, second = byte_pair
            self.vocab[token_val] = self.vocab[first] + self.vocab[second]

    def decode(self, tokens: List[int]) -> str:
        decoded_tokens: List[int] = []
        for token in tokens:
            decoded_tokens.extend(self.vocab[token])
        return bytes(decoded_tokens).decode("utf-8", errors="replace")

    def encode(self, text: str) -> List[int]:
        byte_tokens = list(text.encode("utf-8"))
        linked_list = DoublyLinkedList()
        byte_pair_cache_map: Dict[Tuple[int, int], Set[Node]] = {}

        for byt in byte_tokens:
            linked_list.insert(byt)

        idx = 0
        itr = iter(linked_list)

        while idx < linked_list.length:
            curr_node = next(itr)
            next_node = curr_node.next
            idx += 1

            if next_node is None:
                break

            curr_byte_pair = (curr_node.val, next_node.val)

            if byte_pair_cache_map.get(curr_byte_pair) is None:
                byte_pair_cache_map[curr_byte_pair] = set([curr_node])
            else:
                byte_pair_cache_map[curr_byte_pair].add(curr_node)

        for tok_tuple, new_token in self.byte_pair_map.items():
            if tok_tuple in byte_pair_cache_map:
                curr_tok_pair_nodes = byte_pair_cache_map[tok_tuple]

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

                    if first_node.prev is not None:
                        prev_to_first = first_node.prev
                        new_node.prev = prev_to_first
                        prev_to_first.next = new_node

                        byte_pair_cache_map[(prev_to_first.val, first_node.val)].remove(prev_to_first)

                        if byte_pair_cache_map.get((prev_to_first.val, new_node.val)) is None:
                            byte_pair_cache_map[(prev_to_first.val, new_node.val)] = set([prev_to_first])
                        else:
                            byte_pair_cache_map[(prev_to_first.val, new_node.val)].add(prev_to_first)
                    else:
                        linked_list.head = new_node

                    if second_node.next is not None:
                        next_to_second = second_node.next
                        new_node.next = next_to_second
                        next_to_second.prev = new_node

                        byte_pair_cache_map[(second_node.val, next_to_second.val)].remove(second_node)

                        if byte_pair_cache_map.get((new_node.val, next_to_second.val)) is None:
                            byte_pair_cache_map[(new_node.val, next_to_second.val)] = set([new_node])
                        else:
                            byte_pair_cache_map[(new_node.val, next_to_second.val)].add(new_node)
                    else:
                        linked_list.tail = new_node

                    first_node.next = None
                    first_node.prev = None
                    second_node.next = None
                    second_node.prev = None

                    linked_list.length -= 1

                del byte_pair_cache_map[tok_tuple]

        itr = iter(linked_list)
        output_list: List[int] = []
        idx = 0
        while idx < linked_list.length:
            output_list.append(next(itr).val)
            idx += 1

        return output_list
