from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Generic, TypeVar

from .types import AssistantMessage, AssistantMessageEvent, DoneEvent, ErrorEvent

TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")


class EventStream(Generic[TEvent, TResult], AsyncIterator[TEvent]):
    def __init__(
        self,
        producer: Callable[[], AsyncIterator[TEvent]],
        is_complete: Callable[[TEvent], bool],
        extract_result: Callable[[TEvent], TResult],
    ) -> None:
        self._producer = producer
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._iterator: AsyncIterator[TEvent] | None = None
        self._done = False
        self._has_result = False
        self._result: TResult | None = None

    def __aiter__(self) -> "EventStream[TEvent, TResult]":
        if self._iterator is None:
            self._iterator = self._producer()
        return self

    async def __anext__(self) -> TEvent:
        if self._done:
            raise StopAsyncIteration

        if self._iterator is None:
            self._iterator = self._producer()

        try:
            event = await anext(self._iterator)
        except StopAsyncIteration:
            self._done = True
            raise

        if self._is_complete(event):
            self._done = True
            self._has_result = True
            self._result = self._extract_result(event)

        return event

    async def result(self) -> TResult:
        if self._has_result:
            return self._result  # type: ignore[return-value]

        async for _ in self:
            pass

        if not self._has_result:
            raise RuntimeError("Stream ended without a final result event.")

        return self._result  # type: ignore[return-value]

    async def aclose(self) -> None:
        self._done = True
        iterator = self._iterator
        if iterator is None:
            return

        close = getattr(iterator, "aclose", None)
        if close is None:
            return

        result = close()
        if isinstance(result, Awaitable):
            await result

    async def __aenter__(self) -> "EventStream[TEvent, TResult]":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    def __init__(self, producer: Callable[[], AsyncIterator[AssistantMessageEvent]]) -> None:
        super().__init__(
            producer=producer,
            is_complete=lambda event: isinstance(event, (DoneEvent, ErrorEvent)),
            extract_result=lambda event: (
                event.message if isinstance(event, DoneEvent) else event.error
            ),
        )
