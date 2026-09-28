"""Read Guitar Pro ties using event identity during incremental parsing."""

from types import MethodType


def _tied_value(self, note):
    current = note.beat.voice.measure
    voice_index = next(i for i, voice in enumerate(current.voices) if voice is note.beat.voice)
    for measure in reversed(current.track.measures):
        voice = measure.voices[voice_index]
        beats = voice.beats
        if measure is current:
            index = next(i for i, beat in enumerate(beats) if beat is note.beat)
            beats = beats[:index]
        for beat in reversed(beats):
            if beat.status.name != 'empty':
                for previous in beat.notes:
                    if previous.string == note.string:
                        return previous.value
    return -1


def read_gp(stream, encoding='cp1252'):
    from guitarpro.io import _open

    reader, close = _open(None, stream, 'rb', encoding=encoding)
    # PyGuitarPro uses list.index while the current beat is only partly read.
    # Equal notes can then match an earlier beat whose duration differs, making
    # a later tie incorrectly inherit an older pitch. Identity is unambiguous.
    reader.getTiedNoteValue = MethodType(_tied_value, reader)
    try:
        return reader.readSong()
    finally:
        if close:
            reader.close()
