# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

import dataclasses
import threading
import sounddevice as sd
import numpy as np

if TYPE_CHECKING:
    import keyboard

__all__ = (
    "Waker",
    "PorcupineWaker",
    "HotkeyWaker",
)


class Waker:
    """Base class for all wakers.

    This class provides a common interface for implementing wake up
    mechanisms such as hotword detection, push-to-talk, etc.

    For now, Kevin only provides :class:`PorcupineWaker` however this
    class may be extended to implement support for forms of wake up
    mechanisms. All subclasses must implement the :meth:`.try_wake`
    method.

    .. note::

        This class provides an underlying "awake event" that can be controlled
        using the already implemented methods. To ensure the event is initialized,
        do not forget to call this class's constructor if you override the constructor
        in your implementation.
    """

    def __init__(self) -> None:
        self.__awake = threading.Event()

    def try_wake(self) -> None:
        """Attempts to wake based on any underlying condition.

        This method is to be implemented by subclasses.

        This method is called repeatedly by :class:`Kevin` while the
        awake state is not set. It is recommended for implementations
        to block the method caller until the awake state isn't set.

        See implementation of :class:`PorcupineWaker` for more details.
        """
        raise NotImplementedError

    def awake(self) -> bool:
        """Determines if the awake state is set."""
        return self.__awake.is_set()

    def wake(self) -> None:
        """Sets the awake state.

        This method should be called by :meth:`.try_wake` to set
        the awake state once the wake up condition is met e.g. hotword
        detected.
        """
        self.__awake.set()

    def sleep(self) -> bool:
        """Clears the awake state.

        This is typically called by :class:`Kevin` when the assistant
        is going to sleep.

        Returns true if awake state was cleared i.e. assistant has slept. If
        assistant was already asleep, this returns false.
        """
        was_set = self.__awake.is_set()
        self.__awake.clear()

        return was_set

    def wait_until_awake(self, *, timeout: float | None = None) -> None:
        """Blocks until awake state is set.

        This is only meant for niche cases and should not generally
        be used.

        Parameters
        ----------
        timeout: :class:`float` | None
            The number of seconds to wait for awake state to set
            before timing out.

            If set to None (default), the wait is indefinite.
        """
        self.__awake.wait(timeout)


@dataclasses.dataclass
class _AudioSpec:
    sample_rate: int = 16_000
    channels: int = 1
    dtype: str = "float32"
    frame_size: int | None = None


class PorcupineWaker(Waker):
    """Porcupine hotword engine based waker.

    This waker requires ``pvporcupine`` library to be installed.

    Parameters
    ----------
    access_key: :class:`str`
        The Porcupine access key.
    keywords: Sequence[:class:`str`]
        The list of hotwords supported by Porcupine.
    keyword_paths: Sequence[:class:`str`]
        The list of paths for custom keywords.
    porcupine_options: Dict[:class:`str`, Any]
        Additional options passed to :func:`pvporcupine.create` function.
    """

    def __init__(
        self,
        *,
        access_key: str | None = None,
        keywords: Sequence[str] | None = None,
        keyword_paths: Sequence[str] | None = None,
        porcupine_options: dict[str, Any] | None = None,
        device_index: int | None = None,
    ):
        try:
            import pvporcupine
        except ImportError:
            raise RuntimeError("pvporcupine must be installed to use PorcupineWaker")

        if porcupine_options is None:
            porcupine_options = {}

        porcupine_options.setdefault("access_key", access_key)
        porcupine_options.setdefault("keywords", keywords)
        porcupine_options.setdefault("keyword_paths", keyword_paths)

        self._porcupine = pvporcupine.create(**porcupine_options)
        self._device_index = device_index

        super().__init__()

    def _get_audio_spec(self) -> _AudioSpec:
        return _AudioSpec(
            sample_rate=self._porcupine.sample_rate,
            channels=1,
            dtype="int16",
            frame_size=self._porcupine.frame_length,
        )

    def _read_speech(self) -> None:
        spec = self._get_audio_spec()

        with sd.InputStream(
            samplerate=spec.sample_rate,
            blocksize=spec.frame_size,
            dtype=spec.dtype,
            channels=spec.channels,
            device=self._device_index,
        ) as stream:
            while not self.awake():
                data, _ = stream.read(spec.frame_size)

                if self._process_input(data):
                    self.wake()

    def _process_input(self, data: np.ndarray[np._AnyShapeT, np.dtype[Any]]) -> None:
        return self._porcupine.process(data[:, 0]) >= 0  # type: ignore

    def try_wake(self) -> None:
        self._read_speech()


class HotkeyWaker(Waker):
    """Waker based on single press of hotkey.

    Once the bound hotkey is pressed, the awake state is set. This
    is not the same as push-to-talk which would have required the
    hotkey to be held.

    This waker requires `keyboard` library to be installed.

    Parameters
    ----------
    hotkey: :class:`str`
        The hotkey to listen to. This can be given as a keyboard combination
        e.g. `ctrl + /`
    """
    def __init__(self, hotkey: str) -> None:
        try:
            import keyboard
        except Exception:
            raise RuntimeError("keyboard library must be installed to use HotkeyWaker")

        self.hotkey = hotkey
        self._wait = keyboard.wait

        super().__init__()

    def try_wake(self) -> None:
        self._wait(self.hotkey)
        self.wake()
