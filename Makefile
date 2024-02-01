.PHONY: build
build: setup
	SKIP_PROTO_GEN=1 cargo build --workspace

.PHONY: setup
setup:
	ci/setup.sh

.PHONY: clippy
clippy: setup
	SKIP_PROTO_GEN=1 cargo clippy --workspace --bins --examples --tests -- --no-deps

.PHONY: test
test: setup
	SKIP_PROTO_GEN=1 RUST_LOG=off cargo test --workspace
