# Contributing

Brazier is in its bootstrap phase. Before submitting changes:

1. run `cargo fmt --all --check`;
2. run `cargo test --workspace`;
3. run `pnpm check` and `pnpm test`;
4. document capability or API changes.

Engine adapters must report capabilities rather than silently accepting and
discarding unsupported model features. Build recipes must remain controlled by
Brazier and must never automatically invoke convenience installers from a fork.
