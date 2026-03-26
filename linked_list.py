# Implementation of a Doubly Linked List
from __future__ import annotations
from typing import Optional

class Node:
    val: Optional[int]
    prev: Optional[Node]
    next: Optional[Node]

    def __init__(self, token: int, prev: Optional[None] = None, next: Optional[None] = None) -> None:
        self.val = token
        self.prev = prev
        self.next = next

class DoublyLinkedList:
    length: int = 0
    head: Optional[Node]
    tail: Optional[Node]

    def __init__(self) -> None:
        self.head = None
        self.tail = None

    # Inserts the given value as a Node and returns a reference to it
    def insert(self, token: int) -> Node:
        self.length += 1
        node = Node(token=token)

        # First insertion into the linked list
        if self.head is None and self.tail is None:
            self.head = self.tail = node
            return node
        
        # Not the first insertion

        # Pythonic checks in place
        assert self.tail is not None
        assert self.head is not None
        
        self.tail.next = node
        node.prev = self.tail
        self.tail = node
        return node
    
    def get_length(self) -> int:
        return self.length

    def __iter__(self):
        self.curr = self.head
        return self
    

    def __next__(self):
        pass