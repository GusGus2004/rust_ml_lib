mod matrix;
mod model;
mod rng;

use matrix::Matrix;
use model::{ModelContext, ModelTrainingDesc, ModelVarFlags};


fn draw_mnist_digit(data: &[f32]) {
    for y in 0..28usize {
        for x in 0..28usize {
            let num = data[x + y * 28];
            let col = 232 + (num * 23.0) as i32;
            print!("\x1b[48;5;{}m  ", col);
        }
        println!();
    }
    print!("\x1b[0m");
}


fn create_mnist_model(model: &mut ModelContext) -> Result<(), matrix::MatrixError> {
    let input = model.mv_create(784, 1, ModelVarFlags::INPUT)?;

    let w0 = model.mv_create(16, 784, ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER)?;
    let w1 = model.mv_create(16, 16, ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER)?;
    let w2 = model.mv_create(10, 16, ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER)?;

    let bound0 = (6.0f32 / (784.0 + 16.0)).sqrt();
    let bound1 = (6.0f32 / (16.0 + 16.0)).sqrt();
    let bound2 = (6.0f32 / (16.0 + 10.0)).sqrt();

    
    model.vars[w0].val.as_ref().unwrap().borrow_mut().fill_rand(&mut model.prng, -bound0, bound0);
    model.vars[w1].val.as_ref().unwrap().borrow_mut().fill_rand(&mut model.prng, -bound1, bound1);
    model.vars[w2].val.as_ref().unwrap().borrow_mut().fill_rand(&mut model.prng, -bound2, bound2);

    let b0 = model.mv_create(16, 1, ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER)?;
    let b1 = model.mv_create(16, 1, ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER)?;
    let b2 = model.mv_create(10, 1, ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER)?;

    let z0_a = model.mv_matmul(w0, input, ModelVarFlags::NONE)?;
    let z0_b = model.mv_add(z0_a, b0, ModelVarFlags::NONE)?;
    let a0 = model.mv_relu(z0_b, ModelVarFlags::NONE)?;

    let z1_a = model.mv_matmul(w1, a0, ModelVarFlags::NONE)?;
    let z1_b = model.mv_add(z1_a, b1, ModelVarFlags::NONE)?;
    let z1_c = model.mv_relu(z1_b, ModelVarFlags::NONE)?;
    let a1 = model.mv_add(a0, z1_c, ModelVarFlags::NONE)?;

    let z2_a = model.mv_matmul(w2, a1, ModelVarFlags::NONE)?;
    let z2_b = model.mv_add(z2_a, b2, ModelVarFlags::NONE)?;
    let output = model.mv_softmax(z2_b, ModelVarFlags::OUTPUT)?;

    let y = model.mv_create(10, 1, ModelVarFlags::DESIRED_OUTPUT)?;

    
    let _cost = model.mv_cross_entropy(y, output, ModelVarFlags::COST)?;

    Ok(())
}

fn main() -> Result<(), matrix::MatrixError> {
    let train_images = Matrix::load(60000, 784, "train_images.mat")?;
    let test_images = Matrix::load(10000, 784, "test_images.mat")?;
    let mut train_labels = Matrix::create(60000, 10)?;
    let mut test_labels = Matrix::create(10000, 10)?;

    {
        let train_labels_file = Matrix::load(60000, 1, "train_labels.mat")?;
        let test_labels_file = Matrix::load(10000, 1, "test_labels.mat")?;

        for i in 0..60000usize {
            let num = train_labels_file.data[i] as usize;
            train_labels.data[i * 10 + num] = 1.0;
        }

        for i in 0..10000usize {
            let num = test_labels_file.data[i] as usize;
            test_labels.data[i * 10 + num] = 1.0;
        }
    }

    draw_mnist_digit(&test_images.data);
    for i in 0..10 {
        print!("{:.0} ", test_labels.data[i]);
    }
    print!("\n\n");

    
    let mut model = ModelContext::new(42);
    create_mnist_model(&mut model)?;
    model.compile();

    let input_idx = model.input.expect("no ModelVar was created with INPUT flag set");
    let output_idx = model.output.expect("no ModelVar was created with OUTPUT flag set");

    model.vars[input_idx].val.as_ref().unwrap().borrow_mut().data[..784]
        .copy_from_slice(&test_images.data[..784]);
    model.feedforward();

    print!("pre-training output: ");
    for i in 0..10 {
        print!("{:.2} ", model.vars[output_idx].val.as_ref().unwrap().borrow().data[i]);
    }
    println!();

    let training_desc = ModelTrainingDesc {
        train_images: &train_images,
        train_labels: &train_labels,
        test_images: &test_images,
        test_labels: &test_labels,

        epochs: 10,
        batch_size: 50,
        learning_rate: 0.01,
    };
    model.train(&training_desc);

    model.vars[input_idx].val.as_ref().unwrap().borrow_mut().data[..784]
        .copy_from_slice(&test_images.data[..784]);
    model.feedforward();

    print!("post-training output: ");
    for i in 0..10 {
        
        print!("{:.6} ", model.vars[output_idx].val.as_ref().unwrap().borrow().data[i]);
    }
    print!("\n\n");

    Ok(())
}