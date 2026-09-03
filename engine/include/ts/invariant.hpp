#pragma once

#include <cstdio>
#include <cstdlib>

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

} // namespace ts
