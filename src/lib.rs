use std::f64;
use std::fs::File;
use std::io::{BufWriter, Write};

use argminmax::ArgMinMax;
use serde::Serialize;
use tqdm::tqdm;

#[derive(Serialize)]
struct Result {
    number_to_sample: Vec<usize>,
    result: Vec<Vec<usize>>,
}

#[derive(serde::Serialize)]
struct Metadata {
    dtype: String,
    total_count: u64,
    shape: [usize; 2],
}

/// SAFETY: The following invariants must be ensured from the Python wrapper:
/// * `number_to_sample` must point to a buffer of length `n_datasets`.
/// * `cache_filename_stem` must point to a buffer of length `cache_filename_stem_len` containing valid UTF-8 data
#[no_mangle]
extern "C" fn sample(
    number_to_sample: *const u64,
    n_datasets: usize,
    cache_filename_stem: *const u8,
    cache_filename_stem_len: usize,
) {
    let number_to_sample = unsafe { std::slice::from_raw_parts(number_to_sample, n_datasets) };
    let cache_filename_stem_slice =
        unsafe { std::slice::from_raw_parts(cache_filename_stem, cache_filename_stem_len) };
    let cache_filename_stem = unsafe { std::str::from_utf8_unchecked(cache_filename_stem_slice) };
    sample_impl(number_to_sample, cache_filename_stem);
}

fn sample_impl(number_to_sample: &[u64], cache_filename_stem: &str) {
    let input_json_filepath = format!("{cache_filename_stem}.input.json");
    let input_json_writer = BufWriter::new(
        File::create(input_json_filepath)
            .expect("Failed to create blended data set index input.json file."),
    );
    serde_json::to_writer(input_json_writer, number_to_sample)
        .expect("Failed to write to blended data set index input.json file.");

    let mut number_sampled: Vec<u64> = vec![0; number_to_sample.len()];
    let mut proportion_sampled: Vec<f64> = vec![0.0; number_to_sample.len()];

    let total_count = number_to_sample.iter().copied().sum::<u64>() as usize;

    let bin_filename = format!("{cache_filename_stem}.bin");
    let mut result_writer = BufWriter::new(
        File::create(bin_filename).expect("Failed to create blended data set index bin file."),
    );

    for _ in tqdm(0..total_count) {
        // find smallest represenation
        let (dataset_index, _argmax) = proportion_sampled.argminmax();

        // add representation from dataset
        let sample_index = number_sampled[dataset_index];

        // update proportions
        number_sampled[dataset_index] += 1;
        proportion_sampled[dataset_index] =
            (number_sampled[dataset_index] as f64) / number_to_sample[dataset_index] as f64;

        result_writer
            .write_all(&(dataset_index as u64).to_le_bytes())
            .expect("Failed to write blended data set index bin file.");
        result_writer
            .write_all(&sample_index.to_le_bytes())
            .expect("Failed to write blended data set index bin file.");
    }

    assert!(total_count as u64 == number_sampled.iter().copied().sum::<u64>());

    number_to_sample
        .iter()
        .zip(number_sampled.iter())
        .for_each(|(a, b)| assert_eq!(a, b));

    result_writer
        .flush()
        .expect("Failed to flush blended data set index bin file.");

    // write metadata
    let meta_filename = format!("{cache_filename_stem}.meta.json");
    let meta_writer = BufWriter::new(
        File::create(meta_filename).expect("Failed to create blended data set meta.json file"),
    );
    serde_json::to_writer(
        meta_writer,
        &Metadata {
            total_count: total_count as u64,
            dtype: "uint64".to_string(),
            shape: [total_count, 2],
        },
    )
    .expect("Failed to write blended data set meta.json file");
}
