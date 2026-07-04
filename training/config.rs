use bullet_lib::{
    game::{inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount},
    nn::optimiser::{AdamW, AdamWParams},
    trainer::{save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings},
    value::{ValueTrainerBuilder, loader::SfBinpackLoader},
};
use sfbinpack::chess::{piecetype::PieceType, r#move::MoveType};

fn main() {
    const FT_SIZE: usize = 1024;
    const NUM_OUTPUT_BUCKETS: usize = 8;

    let args: Vec<String> = std::env::args().collect();
    let dataset_dir = get_arg(&args, "--dataset-dir", "/content/data");
    let superbatches: usize = get_arg(&args, "--superbatches", "150").parse().unwrap();
    let wdl_proportion: f32 = get_arg(&args, "--wdl", "0.07").parse().unwrap();
    let initial_lr: f32 = get_arg(&args, "--lr", "0.001").parse().unwrap();
    let final_lr: f32 = get_arg(&args, "--final-lr",
        &format!("{}", initial_lr * 0.3f32.powi(5))).parse().unwrap();
    let save_rate: usize = get_arg(&args, "--save-rate", "25").parse().unwrap();
    let start_sb: usize = get_arg(&args, "--start-superbatch", "1").parse().unwrap();
    let load_ckpt: Option<String> = get_arg_opt(&args, "--load-checkpoint");
    let load_weights: Option<String> = get_arg_opt(&args, "--load-weights");

    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
         0,  4,  8, 12,
         0,  4,  8, 12,
         1,  5,  9, 13,
         1,  5,  9, 13,
         2,  6, 10, 14,
         2,  6, 10, 14,
         3,  7, 11, 15,
         3,  7, 11, 15,
    ];
    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").round().quantise::<i16>(64),
            SavedFormat::id("l1b").round().quantise::<i32>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().power_error(target, 2.5))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, FT_SIZE);
            let l1 = builder.new_affine("l1", 2 * FT_SIZE, NUM_OUTPUT_BUCKETS);
            let stm = l0.forward(stm_inputs).screlu();
            let ntm = l0.forward(ntm_inputs).screlu();
            l1.forward(stm.concat(ntm)).select(output_buckets)
        });
    trainer.optimiser.set_params_for_weight("l0w",
        AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() });

    if let Some(ckpt) = &load_ckpt {
        println!("Loading full checkpoint (weights + AdamW state) from: {}", ckpt);
        trainer.optimiser.load_from_checkpoint(ckpt)
            .expect("Failed to load checkpoint");
    }

    if let Some(wpath) = &load_weights {
        println!("Loading weights only from: {}", wpath);
        trainer.optimiser.load_weights_from_file(wpath)
            .expect("Failed to load weights");
    }

    let schedule = TrainingSchedule {
        net_id: "v5-1024s-colab".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16384,
            batches_per_superbatch: 6104,
            start_superbatch: start_sb,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr,
            final_superbatch: superbatches },
        save_rate,
    };
    let settings = LocalSettings {
        threads: 4, batch_queue_size: 64,
        output_directory: "/content/drive/MyDrive/nnue_checkpoints",
        test_set: None,
    };
    let filter = |e: &sfbinpack::TrainingDataEntry| {
        let stm = e.pos.side_to_move();
        e.ply >= 16 && !e.pos.is_checked(stm)
            && e.score.unsigned_abs() <= 10000
            && e.mv.mtype() == MoveType::Normal
            && e.pos.piece_at(e.mv.to()).piece_type() == PieceType::None
    };
    let data_files: Vec<String> = std::fs::read_dir(&dataset_dir)
        .unwrap().filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.extension().map_or(false, |ext| ext == "binpack") {
                Some(p.to_string_lossy().to_string())
            } else { None }
        }).collect();
    assert!(!data_files.is_empty(), "No binpack in {}", dataset_dir);
    let refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();
    let dl = SfBinpackLoader::new_concat_multiple(&refs, 256, 4, filter);

    println!("=== v5 1024s SCReLU ===");
    println!("{}/{} SBs, WDL {}, LR {}→{}", superbatches,
        NUM_OUTPUT_BUCKETS, wdl_proportion, initial_lr, final_lr);
    trainer.run(&schedule, &settings, &dl);
}

fn get_arg(a: &[String], f: &str, d: &str) -> String {
    get_arg_opt(a, f).unwrap_or_else(|| d.to_string())
}

fn get_arg_opt(a: &[String], f: &str) -> Option<String> {
    a.iter().position(|s| s == f)
        .and_then(|i| a.get(i+1))
        .map(|s| s.to_string())
}