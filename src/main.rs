use std::f64;
use argminmax::ArgMinMax;  // import trait
use std::fs::File;
use std::io::{BufWriter, Write};
use tqdm::tqdm;

fn main() {
    let number_to_sample: Vec<usize> = vec![1, 2, 3, 4];
    let mut number_sampled: Vec<usize> = vec![0; number_to_sample.len()];
    let mut proportion_sampled: Vec<f64> = vec![0.0; number_to_sample.len()];

    let total_count: usize = number_to_sample.iter().copied().sum::<usize>();

    let mut result: Vec<Vec<usize>> = Vec::with_capacity(total_count);

    for _ in tqdm(0..total_count) {
        
        // find smallest represenation
        let (dataset_index, _argmax) = proportion_sampled.argminmax();  


        // add representation from dataset
        let index_next =  number_sampled[dataset_index];

        // update proportions
        number_sampled[dataset_index] += 1;
        proportion_sampled[dataset_index] = (number_sampled[dataset_index] as f64) / number_to_sample[dataset_index] as f64;

        result.push(vec![dataset_index, index_next]);

    }

    assert!(total_count == number_sampled.iter().copied().sum::<usize>());

    // write
    let file = File::create("result.json").unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &result).unwrap();
    writer.flush().unwrap();
   
}