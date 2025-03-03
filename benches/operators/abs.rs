use super::constants::{INPUT_SIZE, RANDOM_SEED};
use criterion::{black_box, BenchmarkId, Criterion};
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub fn bench_onnx(c: &mut Criterion) {
    let mut group = c.benchmark_group("abs_ONNX");

    let env = Environment::builder().build().unwrap();
    let session_cpu = env
        .new_session_builder()
        .model_from_file("tests/models/abs.onnx")
        .unwrap();

    // Generate random input tensor
    let mut rng = StdRng::seed_from_u64(RANDOM_SEED);
    let input_data: Vec<f32> = (0..INPUT_SIZE).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let input = OrtOwnedTensor::from_slice(&input_data, &[INPUT_SIZE as i64]).unwrap();
    let mut outputs = vec![OrtOwnedTensor::new()];

    // Benchmark for CPU
    group.bench_with_input(
        BenchmarkId::new("abs", "onnx_cpu"),
        &input,
        |b, input| {
            b.iter(|| {
                session_cpu.run(vec![input.clone()], &mut outputs).unwrap();
                black_box(&outputs);
            });
        },
    );

    // Assuming you have a GPU session setup similarly
    #[cfg(feature = "gpu")]
    {
        let session_gpu = env
            .new_session_builder()
            .model_from_file("tests/models/abs.onnx")
            .unwrap();

        // Benchmark for GPU
        group.bench_with_input(
            BenchmarkId::new("abs", "onnx_gpu"),
            &input,
            |b, input| {
                b.iter(|| {
                    session_gpu.run(vec![input.clone()], &mut outputs).unwrap();
                    black_box(&outputs);
                });
            },
        );
    }

    group.finish();
}


// create onnx model
fn create_onnx_model() {
    let mut builder = ModelBuilder::new();
    let input = builder.add_input("input", InferenceType::new(InferenceShape::new(vec![1, 3, 224, 224])));
    let output = builder.add_output("output", InferenceType::new(InferenceShape::new(vec![1, 3, 224, 224])));
    let abs = builder.add_node("abs", "Abs", vec![input], vec![output]);
    let model = builder.build();
    model.save("tests/models/abs.onnx").unwrap();
}
