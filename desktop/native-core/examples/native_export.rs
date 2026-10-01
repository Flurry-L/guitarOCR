//! Development CLI: independent-reader interoperability checks, no Python invoked.
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        args.len() == 4,
        "native_export <score.json> <musicxml|gp5> <output>"
    );
    let input = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let result = match args[2].as_str() {
        "musicxml" => guitarocr_native_core::music_exports::write_musicxml(
            &input,
            std::path::Path::new(&args[3]),
        ),
        "gp5" => {
            guitarocr_native_core::music_exports::write_gp5(&input, std::path::Path::new(&args[3]))
        }
        _ => panic!("Unknown format"),
    };
    match result {
        Ok(report) => println!("{}", report),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1)
        }
    }
}
