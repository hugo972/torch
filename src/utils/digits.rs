use std::fs::File;
use std::io::Read;

pub const DIGIT_COUNT: usize = 1000;
pub const DIGIT_SIZE: usize = 28;
pub const DIGIT_BUFFER_SIZE: usize = DIGIT_SIZE.pow(2);

pub fn load_digit_data(path: &str) -> Vec<Vec<f32>> {
    let mut file = File::open(path).unwrap();
    let mut image_data = vec![0u8; DIGIT_COUNT * DIGIT_BUFFER_SIZE];
    file.read(&mut image_data).unwrap();
    image_data
        .chunks(DIGIT_BUFFER_SIZE)
        .map(|digit| {
            digit
                .iter()
                .map(|&d| d as f32 / 255.0)
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}
