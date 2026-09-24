#!/usr/bin/env bash
#
# Install clang without root, on a Debian/Ubuntu box where `apt-get install clang` is not an
# option (a rented container running as an unprivileged user).
#
#     tools/scripts/install_clang_userspace.sh            # clang 21 into ~/.local/opt/llvm-21
#     LLVM_VERSION=20 PREFIX=/some/dir tools/scripts/install_clang_userspace.sh
#
# The engine is built with clang (see the root CMakeLists.txt, which refuses any other compiler).
# With root, `apt-get install clang` is the whole story and this script is not needed.
#
# `apt-get download` fetches the .deb files without installing them and `dpkg -x` unpacks them
# into PREFIX -- neither needs root. The compiler's own shared libraries (libLLVM, libclang-cpp)
# are then not on the loader path, so `clang`/`clang++` wrappers in ~/.local/bin set it for the
# compiler process only. CMake finds the wrapper there even when ~/.local/bin is not on PATH.
#
# Deliberately NOT installed: libomp. The engine does not use LLVM's OpenMP runtime; it calls
# GCC's libgomp, the one torch ships, so a process holds one thread pool (bindings/AGENTS.md).
# libomp-dev would also put a `libgomp.so` alias to libomp on clang's library path.
set -euo pipefail

V="${LLVM_VERSION:-21}"
PREFIX="${PREFIX:-$HOME/.local/opt/llvm-$V}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
# Compiler, its libraries, its resource headers, the LTO plugin, and compiler-rt (the
# sanitizer runtimes the ASan build needs).
PACKAGES=(
    "clang-$V" "libclang-cpp$V" "libllvm$V" "libclang1-$V"
    "libclang-common-$V-dev" "llvm-$V-linker-tools" "libclang-rt-$V-dev"
)

if ! command -v apt-get >/dev/null || ! command -v dpkg >/dev/null; then
    echo "install_clang_userspace: needs apt-get and dpkg (Debian/Ubuntu)." >&2
    exit 2
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
echo "install_clang_userspace: downloading ${PACKAGES[*]}"
(cd "$work" && apt-get download "${PACKAGES[@]}")

mkdir -p "$PREFIX" "$BIN_DIR"
for deb in "$work"/*.deb; do
    dpkg -x "$deb" "$PREFIX"
done

real_bin="$PREFIX/usr/lib/llvm-$V/bin"
libs="$PREFIX/usr/lib/x86_64-linux-gnu:$PREFIX/usr/lib/llvm-$V/lib"
if [[ ! -x "$real_bin/clang++" ]]; then
    echo "install_clang_userspace: $real_bin/clang++ missing after unpacking" >&2
    exit 2
fi

for tool in clang clang++; do
    cat > "$BIN_DIR/$tool" <<EOF
#!/usr/bin/env bash
# Written by tools/scripts/install_clang_userspace.sh: clang $V unpacked under $PREFIX.
export LD_LIBRARY_PATH="$libs\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "$real_bin/$tool" "\$@"
EOF
    chmod +x "$BIN_DIR/$tool"
done

# Prove it works end to end: compile, link and run against the system libstdc++.
probe="$work/probe"
printf '#include <vector>\n#include <cstdio>\nint main(){std::vector<int> v{1,2,3};std::printf("%%zu\\n",v.size());}\n' \
    > "$probe.cpp"
"$BIN_DIR/clang++" -std=c++20 -O2 "$probe.cpp" -o "$probe"
[[ "$("$probe")" == "3" ]] || { echo "install_clang_userspace: probe binary misbehaved" >&2; exit 2; }

echo "install_clang_userspace: $("$BIN_DIR/clang++" --version | head -1)"
echo "  wrappers: $BIN_DIR/clang, $BIN_DIR/clang++   (CMake looks there; add it to PATH for shells)"
echo "  now reconfigure the engine build: tools/scripts/check_engine_fresh.sh"
