use protobuf_codegen::Customize;

fn main() {
  if std::env::var("SKIP_PROTO_GEN").is_ok() {
    return;
  }

  println!("cargo:rerun-if-changed=proto/");

  std::fs::create_dir_all("src/protos").unwrap();
  protobuf_codegen::Codegen::new()
    .protoc()
    .customize(
      Customize::default()
        .gen_mod_rs(false)
        .tokio_bytes(true)
        .tokio_bytes_for_string(true),
    )
    .includes(["proto/", "../proto/src/"])
    .inputs(["proto/prometheus-types.proto"])
    .out_dir("src/proto")
    .capture_stderr()
    .run_from_script();
}
