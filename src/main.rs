mod utils;

use crate::utils::digits::{load_digit_data, DIGIT_COUNT};
use crate::utils::shuffle::ShuffleIterExt;
use tch::nn::{Module, OptimizerConfig};
use tch::Kind::Float;
use tch::{nn, Device, Tensor};

pub const TRAIN_DIGIT_COUNT: usize = 200;

fn main() {
    let digit_datas = [
        load_digit_data("./data/data0.bin"),
        load_digit_data("./data/data1.bin"),
        load_digit_data("./data/data2.bin"),
        load_digit_data("./data/data3.bin"),
        load_digit_data("./data/data4.bin"),
        load_digit_data("./data/data5.bin"),
        load_digit_data("./data/data6.bin"),
        load_digit_data("./data/data7.bin"),
        load_digit_data("./data/data8.bin"),
        load_digit_data("./data/data9.bin"),
    ];

    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    let net = nn::seq()
        .add(nn::linear(
            &vs.root() / "layer1",
            784,
            128,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            &vs.root() / "layer2",
            128,
            10,
            Default::default(),
        ))
        .add_fn(|xs| xs.softmax(0, Float));

    //train_network(&net, &digit_datas, &device, &vs, 100, 0.001);

    vs.load("./data/digit.tnn").unwrap();

    for digit in 0..=9 {
        let digit_variant = rand::random_range(TRAIN_DIGIT_COUNT..DIGIT_COUNT);

        let input = Tensor::from_slice(&digit_datas[digit][digit_variant]);
        let output = net.forward(&input);

        let mut result = [0.0f32; 10];
        output.copy_data(&mut result, 10);
        let predicted_digit = result
            .iter()
            .enumerate()
            .max_by_key(|(i, v)| (**v * 100.0) as i32)
            .unwrap()
            .0;
        println!(
            "Actual: {} [{}] Prediction: {}",
            digit, digit_variant, predicted_digit
        );
    }
}

fn train_network(
    neural_network: &nn::Sequential,
    digit_datas: &[Vec<Vec<f32>>; 10],
    device: &Device,
    var_store: &nn::VarStore,
    epochs: usize,
    learning_rate: f64,
) {
    let digit_train_data = digit_datas
        .iter()
        .enumerate()
        .flat_map(|(digit, digit_data)| {
            let mut output = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            output[digit] = 1.0;

            digit_data
                .iter()
                .take(TRAIN_DIGIT_COUNT)
                .map(|digit| {
                    let digit_input = Tensor::from_slice(&digit);
                    let label = Tensor::from_slice(&output);
                    (digit_input, label)
                })
                .collect::<Vec<_>>()
        })
        .shuffle()
        .collect::<Vec<_>>();

    let mut opt = nn::Adam::default().build(var_store, learning_rate).unwrap();
    for epoch in 1..=epochs {
        let mut error = Tensor::zeros(&[1], (Float, *device));

        for digit in 0..TRAIN_DIGIT_COUNT {
            let (digit_input, label) = &digit_train_data[digit];
            let output = neural_network.forward(digit_input);
            let loss = output.mse_loss(label, tch::Reduction::Mean);

            opt.backward_step(&loss);

            error += loss;
        }

        println!("epoch {:?} error: {:?}", epoch, error / 100.0);
    }

    var_store.save("./data/digit.tnn").unwrap();
}
