use criterion::{criterion_group, criterion_main};

mod abs;
mod constants;

criterion_group!(
    benches,
    abs::bench_onnx
);

criterion_main!(benches);
