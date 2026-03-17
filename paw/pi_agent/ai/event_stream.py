from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from .types import AssistantMessage, AssistantMessageEvent

TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")


class EventStream(Generic[TEvent, TResult], Iterator[TEvent]):
    def __init__(
        self,
        producer: Callable[[], Iterator[TEvent]],
        is_complete: Callable[[TEvent], bool],
        extract_result: Callable[[TEvent], TResult],
    ) -> None:
        self._producer = producer
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._iterator = producer()
        self._done = False
        self._has_result = False
        self._result: TResult | None = None

    def __iter__(self) -> "EventStream[TEvent, TResult]":
        return self

    def __next__(self) -> TEvent:
        if self._done:
            raise StopIteration

        event = next(self._iterator)
        if self._is_complete(event):
            self._done = True
            self._has_result = True
            self._result = self._extract_result(event)
        return event

    def result(self) -> TResult:
        if self._has_result:
            return self._result  # type: ignore[return-value]

        for _ in self:
            pass

        if not self._has_result:
            raise RuntimeError("Stream ended without a final result event.")

        return self._result  # type: ignore[return-value]


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    def __init__(self, producer: Callable[[], Iterator[AssistantMessageEvent]]) -> None:
        super().__init__(
            producer=producer,
            is_complete=lambda event: event["type"] in {"done", "error"},
            extract_result=lambda event: event["message"] if event["type"] == "done" else event["error"],
        )
