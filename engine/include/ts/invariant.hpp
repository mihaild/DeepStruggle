#pragma once

#include <cstdio>
#include <cstdlib>

#include "ts/game_state.hpp"

namespace ts {

// A violated engine invariant is a defect in the engine, not a state a game can be in. There
// is no correct way to carry on from one, and carrying on anyway is how a silently wrong
// result reaches a training set -- which is worse than a crash, because nothing downstream can
// tell it apart from a real game. So it fails where it happens, naming what broke.
//
// The fuzzers already report their own invariant violations this way and exit; this is the
// same thing one level down, for the invariants only the engine can see.
[[noreturn]] inline void invariant_failed(const char* what, int detail) noexcept {
    std::fprintf(stderr, "ts engine invariant violated: %s (%d)\n", what, detail);
    std::fflush(stderr);
    std::abort();
}

// A state that should not arise but can be continued from. Unlike invariant_failed this
// returns, because a training run that aborts on one rare board is worse than one that logs it
// and plays on -- provided the log is loud enough that nobody misses it, and complete enough
// to replay the board exactly.
//
// The whole GameState is dumped as hex. It is trivially copyable and under 4 KB, so those
// bytes are the position itself and can be restored with a memcpy; Serializer::to_json is a
// summary and would not be enough to reproduce the decision. Written to stderr unbuffered so
// it survives a training run that is watching stdout, and repeated for every distinct state so
// a loop cannot hide behind a single report -- an identical state seen twice in a row is
// reported once, since a mask may be generated several times for one decision.
void report_anomaly(const char* what, const GameState& state) noexcept;

} // namespace ts
