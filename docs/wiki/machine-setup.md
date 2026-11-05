# Machine setup facts (dated)

Host facts that change tool speed or behaviour. Not repo code.

- 2026-11-05: 10 cores (8 P + 2 E), macOS. `/usr/bin/git` is the xcrun shim (20.2 ms/spawn). User approved `~/.local/bin/git` symlink to Xcode's git (7.0 ms/spawn). Agents and tests that inherit the user PATH use it. Related: [test-suite-speed.md](test-suite-speed.md).
- 2026-11-05 (later): the symlink broke `git fetch` over https (git derives its exec-path from the link: ~/.local/libexec/git-core, missing). Replaced by a 2-line exec wrapper script at ~/.local/bin/git: fetch OK, 10.7 ms/spawn (symlink 7.0, shim 20.2). Supersedes the symlink.
