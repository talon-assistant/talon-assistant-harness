"""iTunes control talent — play, pause, skip, search, playlists, volume, etc.

Uses the iTunes COM API (Windows only). Requires iTunes to be installed.
win32com is already a project dependency (pywin32).
"""

import re
from talents.base import BaseTalent

try:
    import win32com.client
    _WIN32COM_AVAILABLE = True
except ImportError:
    _WIN32COM_AVAILABLE = False

# iTunes PlayerState constants
_STATE_STOPPED = 0
_STATE_PLAYING = 1
_STATE_PAUSED  = 2

# iTunes search kind constants
_SEARCH_ALL     = 1
_SEARCH_ARTISTS = 2
_SEARCH_ALBUMS  = 3
_SEARCH_SONGS   = 5


class ITunesTalent(BaseTalent):
    name = "itunes"
    description = "Control iTunes: play, pause, skip tracks, search songs, activate playlists, adjust volume, shuffle"
    keywords = ["itunes", "song", "music", "playlist", "play", "pause", "skip", "next track",
                "previous track", "shuffle", "volume"]
    examples = [
        "play music",
        "pause iTunes",
        "skip to the next song",
        "go back to the previous track",
        "play the song Bohemian Rhapsody",
        "play something by The Beatles",
        "play my workout playlist",
        "what song is playing",
        "turn the iTunes volume up",
        "set iTunes volume to 60",
        "shuffle on",
        "turn shuffle off",
        "stop the music",
        "resume",
    ]
    priority = 60

    # Intent labels the LLM will pick from
    _INTENTS = [
        "play", "pause", "stop", "resume", "next", "previous",
        "search_song", "search_artist", "search_album",
        "playlist", "volume_up", "volume_down", "volume_set",
        "shuffle_on", "shuffle_off", "shuffle_toggle",
        "now_playing", "unknown",
    ]

    def __init__(self):
        super().__init__()
        self._itunes = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def routing_available(self) -> bool:
        return _WIN32COM_AVAILABLE

    def _get_itunes(self):
        """Return a live iTunes COM object, (re)connecting as needed."""
        try:
            if self._itunes is None:
                self._itunes = win32com.client.Dispatch("iTunes.Application")
            # Ping it — raises if iTunes has been closed since last call
            _ = self._itunes.PlayerState
            return self._itunes
        except Exception:
            try:
                self._itunes = win32com.client.Dispatch("iTunes.Application")
                return self._itunes
            except Exception as e:
                raise RuntimeError(f"Cannot connect to iTunes: {e}")

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]

        if not _WIN32COM_AVAILABLE:
            return self._fail("pywin32 is not installed — cannot control iTunes.")

        try:
            it = self._get_itunes()
        except RuntimeError as e:
            return self._fail(str(e))

        # --- Classify intent ---
        intent = self._extract_arg(
            llm, command, "intent",
            options=self._INTENTS,
            fallback="unknown",
        )

        # --- Dispatch ---
        try:
            if intent == "play":
                return self._play(it, command, llm)

            elif intent == "pause":
                it.Pause()
                return self._ok("Paused.", "pause")

            elif intent == "resume":
                it.Play()
                return self._ok("Resumed.", "resume")

            elif intent == "stop":
                it.Stop()
                return self._ok("Stopped.", "stop")

            elif intent == "next":
                it.NextTrack()
                return self._ok("Skipped to next track.", "next_track")

            elif intent == "previous":
                it.PreviousTrack()
                return self._ok("Previous track.", "previous_track")

            elif intent in ("search_song", "search_artist", "search_album"):
                return self._search_and_play(it, command, intent, llm)

            elif intent == "playlist":
                return self._activate_playlist(it, command, llm)

            elif intent == "volume_up":
                vol = min(100, it.SoundVolume + 10)
                it.SoundVolume = vol
                return self._ok(f"Volume up to {vol}.", "volume_up")

            elif intent == "volume_down":
                vol = max(0, it.SoundVolume - 10)
                it.SoundVolume = vol
                return self._ok(f"Volume down to {vol}.", "volume_down")

            elif intent == "volume_set":
                return self._set_volume(it, command, llm)

            elif intent == "shuffle_on":
                it.CurrentPlaylist.Shuffle = True
                return self._ok("Shuffle on.", "shuffle_on")

            elif intent == "shuffle_off":
                it.CurrentPlaylist.Shuffle = False
                return self._ok("Shuffle off.", "shuffle_off")

            elif intent == "shuffle_toggle":
                current = it.CurrentPlaylist.Shuffle
                it.CurrentPlaylist.Shuffle = not current
                state = "on" if not current else "off"
                return self._ok(f"Shuffle {state}.", "shuffle_toggle")

            elif intent == "now_playing":
                return self._now_playing(it)

            else:
                return self._fail(
                    "I wasn't sure what you wanted to do in iTunes. "
                    "Try: play, pause, skip, previous, shuffle, or ask for a song or playlist."
                )

        except Exception as e:
            return self._fail(f"iTunes error: {e}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    # Words that on their own mean "just play" — not a search term
    _PLAY_NOISE = {"play", "music", "some", "a", "the", "me", "something", "anything",
                   "itunes", "please", "now", "on", "resume", "start"}

    def _play(self, it, command: str, llm) -> dict:
        """Generic play — if a search term is present, search first; otherwise resume."""
        # Strip noise words and see if anything meaningful remains
        words = re.sub(r"[^\w\s]", "", command.lower()).split()
        leftover = [w for w in words if w not in self._PLAY_NOISE]

        if leftover:
            # Let the LLM normalise the name (fixes misspellings like "zepplin" → "Led Zeppelin")
            term = self._extract_arg(llm, command, "artist or song name", max_length=40) \
                   or " ".join(leftover)
            for kind in (_SEARCH_ARTISTS, _SEARCH_ALL):
                try:
                    results = it.LibraryPlaylist.Search(term, kind)
                    if results and results.Count > 0:
                        track = results.Item(1)
                        track.Play()
                        return self._ok(
                            f"Playing \"{track.Name}\" by {track.Artist}.",
                            "play_track",
                            extra={"track": track.Name, "artist": track.Artist},
                        )
                except Exception:
                    continue
            return self._fail(f"No results found for \"{term}\" in your iTunes library.")

        # No search term — just resume/play
        state = it.PlayerState
        if state == _STATE_PAUSED:
            it.Play()
            return self._ok("Resumed.", "resume")
        elif state == _STATE_PLAYING:
            return self._ok("Already playing.", "play_noop")
        else:
            it.Play()
            return self._ok("Playing.", "play")

    def _search_and_play(self, it, command: str, intent: str, llm) -> dict:
        """Search the library and play the best match."""
        if intent == "search_artist":
            term = self._extract_arg(llm, command, "artist name", max_length=40)
            kind = _SEARCH_ARTISTS
            label = "artist"
        elif intent == "search_album":
            term = self._extract_arg(llm, command, "album name", max_length=40)
            kind = _SEARCH_ALBUMS
            label = "album"
        else:
            term = self._extract_arg(llm, command, "song title", max_length=40)
            kind = _SEARCH_SONGS
            label = "song"

        if not term:
            return self._fail(f"I couldn't find a {label} name in that command.")

        results = it.LibraryPlaylist.Search(term, kind)
        if not results or results.Count == 0:
            # Widen to all fields
            results = it.LibraryPlaylist.Search(term, _SEARCH_ALL)

        if not results or results.Count == 0:
            return self._fail(f"No results found for \"{term}\" in your iTunes library.")

        track = results.Item(1)
        track.Play()
        return self._ok(
            f"Playing \"{track.Name}\" by {track.Artist}.",
            "play_track",
            extra={"track": track.Name, "artist": track.Artist},
        )

    def _activate_playlist(self, it, command: str, llm) -> dict:
        """Find and play a named playlist."""
        name = self._extract_arg(llm, command, "playlist name", max_length=50)
        if not name:
            return self._fail("I couldn't find a playlist name in that command.")

        name_lower = name.lower()

        # Walk all sources (Library, devices, etc.)
        try:
            sources = it.Sources
            for src_idx in range(1, sources.Count + 1):
                try:
                    playlists = sources.Item(src_idx).Playlists
                    for pl_idx in range(1, playlists.Count + 1):
                        pl = playlists.Item(pl_idx)
                        if name_lower in pl.Name.lower():
                            pl.PlayFirstTrack()
                            return self._ok(
                                f"Playing playlist \"{pl.Name}\".",
                                "play_playlist",
                                extra={"playlist": pl.Name},
                            )
                except Exception:
                    continue
        except Exception:
            pass

        # Fallback: main library source only
        try:
            playlists = it.LibrarySource.Playlists
            for pl_idx in range(1, playlists.Count + 1):
                pl = playlists.Item(pl_idx)
                if name_lower in pl.Name.lower():
                    pl.PlayFirstTrack()
                    return self._ok(
                        f"Playing playlist \"{pl.Name}\".",
                        "play_playlist",
                        extra={"playlist": pl.Name},
                    )
        except Exception:
            pass

        return self._fail(f"No playlist matching \"{name}\" found.")

    def _set_volume(self, it, command: str, llm) -> dict:
        """Extract a numeric volume level and apply it."""
        raw = self._extract_arg(llm, command, "volume level as a number 0 to 100", max_length=6)
        if raw:
            m = re.search(r'\d+', raw)
            if m:
                vol = max(0, min(100, int(m.group())))
                it.SoundVolume = vol
                return self._ok(f"Volume set to {vol}.", "volume_set")
        return self._fail("I couldn't find a volume level in that command.")

    def _now_playing(self, it) -> dict:
        """Return info about the current track."""
        state = it.PlayerState
        if state == _STATE_STOPPED:
            return self._ok("iTunes is stopped — nothing is playing.", "now_playing")

        try:
            track = it.CurrentTrack
            status = "Playing" if state == _STATE_PLAYING else "Paused"
            duration_s = int(track.Duration)
            mins, secs = divmod(duration_s, 60)
            info = (
                f"{status}: \"{track.Name}\" by {track.Artist}"
                f" — {track.Album} ({mins}:{secs:02d})"
            )
        except Exception:
            info = "Something is playing but I couldn't read the track info."

        return self._ok(info, "now_playing")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ok(self, response: str, action: str, extra: dict | None = None) -> dict:
        act = {"action": action}
        if extra:
            act.update(extra)
        return {"success": True, "response": response, "actions_taken": [act]}

    def _fail(self, response: str) -> dict:
        return {"success": False, "response": response, "actions_taken": []}
