from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar


T = TypeVar("T")


@dataclass
class Node(Generic[T]):
    value: T
    left: Optional["Node[T]"] = None
    right: Optional["Node[T]"] = None
    height: int = 1


class AVLTree(Generic[T]):
    """Minimal AVL tree supporting the operations needed by tokenizer training."""

    def __init__(self):
        self.root: Optional[Node[T]] = None
        self.size = 0

    def __bool__(self) -> bool:
        return self.root is not None

    def __len__(self) -> int:
        return self.size

    def insert(self, value: T) -> "AVLTree[T]":
        self.root, inserted = self._insert(self.root, value)
        if inserted:
            self.size += 1
        return self

    def add(self, value: T) -> "AVLTree[T]":
        return self.insert(value)

    def discard(self, value: T) -> None:
        self.root, deleted = self._delete(self.root, value)
        if deleted:
            self.size -= 1

    def find(self, value: T) -> bool:
        curr = self.root
        while curr is not None:
            if value == curr.value:
                return True
            if value < curr.value:
                curr = curr.left
            else:
                curr = curr.right
        return False

    def pop_max(self) -> T:
        if self.root is None:
            raise IndexError("pop from empty AVLTree")
        self.root, max_value = self._pop_max(self.root)
        self.size -= 1
        return max_value

    def inorder(self) -> List[T]:
        vals: List[T] = []
        self._inorder(self.root, vals)
        return vals

    def _height(self, node: Optional[Node[T]]) -> int:
        if node is None:
            return 0
        return node.height

    def _update_height(self, node: Node[T]) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def get_balance(self, node: Optional[Node[T]]) -> int:
        if node is None:
            return 0
        return self._height(node.left) - self._height(node.right)

    def _rotate_left(self, node: Node[T]) -> Node[T]:
        assert node.right is not None
        new_root = node.right
        node.right = new_root.left
        new_root.left = node
        self._update_height(node)
        self._update_height(new_root)
        return new_root

    def _rotate_right(self, node: Node[T]) -> Node[T]:
        assert node.left is not None
        new_root = node.left
        node.left = new_root.right
        new_root.right = node
        self._update_height(node)
        self._update_height(new_root)
        return new_root

    def _balance(self, node: Node[T]) -> Node[T]:
        self._update_height(node)
        balance_factor = self.get_balance(node)

        if balance_factor > 1:
            assert node.left is not None
            if self.get_balance(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        if balance_factor < -1:
            assert node.right is not None
            if self.get_balance(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def _insert(self, curr: Optional[Node[T]], value: T) -> tuple[Node[T], bool]:
        if curr is None:
            return Node(value=value), True

        if value == curr.value:
            return curr, False

        if value < curr.value:
            curr.left, inserted = self._insert(curr.left, value)
        else:
            curr.right, inserted = self._insert(curr.right, value)

        return self._balance(curr), inserted

    def _delete(self, node: Optional[Node[T]], value: T) -> tuple[Optional[Node[T]], bool]:
        if node is None:
            return None, False

        if value < node.value:
            node.left, deleted = self._delete(node.left, value)
        elif value > node.value:
            node.right, deleted = self._delete(node.right, value)
        else:
            deleted = True
            if node.left is None:
                return node.right, True
            if node.right is None:
                return node.left, True

            successor = self._min_value_node(node.right)
            node.value = successor.value
            node.right, _ = self._delete(node.right, successor.value)

        if not deleted or node is None:
            return node, deleted

        return self._balance(node), True

    def _pop_max(self, node: Node[T]) -> tuple[Optional[Node[T]], T]:
        if node.right is None:
            return node.left, node.value

        node.right, max_value = self._pop_max(node.right)
        return self._balance(node), max_value

    def _min_value_node(self, node: Node[T]) -> Node[T]:
        curr = node
        while curr.left is not None:
            curr = curr.left
        return curr

    def _inorder(self, node: Optional[Node[T]], vals: List[T]) -> None:
        if node is None:
            return
        self._inorder(node.left, vals)
        vals.append(node.value)
        self._inorder(node.right, vals)
