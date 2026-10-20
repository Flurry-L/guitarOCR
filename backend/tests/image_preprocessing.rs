use guitarocr_backend::image_transforms::contain_rgb;
use image::{Rgb, RgbImage};
use sha2::{Digest, Sha256};

#[test]
fn contain_matches_pillow_pixels_and_centering() {
    // Pillow 12.3 RGB/Lanczos references, including half-pixel ties and narrow crops.
    for ((width, height, w, h), placement, digest) in [
        (
            (3, 2, 9, 9),
            (9, 6, 0, 1),
            "8f7c154286e49caf86a78ba43e959eb50abc2a7822cb0f983fb590e54fe74ebf",
        ),
        (
            (6, 3, 5, 5),
            (5, 2, 0, 1),
            "3ef6fe0c86c3ba38b4561b9f72664acd8207568d34f82a35681ec8a69f78f911",
        ),
        (
            (513, 191, 768, 256),
            (688, 256, 40, 0),
            "d5079b842bdc426c1a783c852d580483658cb4d11157e57a8607cf5038854234",
        ),
        (
            (23, 17, 7, 5),
            (7, 5, 0, 0),
            "3d5841e0b8e67b88ceb2d6dd5c9b7a7ec73fa412c28403751dea6f9c17e33870",
        ),
        (
            (1, 97, 11, 19),
            (1, 19, 5, 0),
            "0b95ac32aad693c71d34580215e0e9c9f73a5ffef85fc03125dc115f0c442d1d",
        ),
    ] {
        let image = RgbImage::from_fn(width, height, |x, y| {
            Rgb([
                ((x * 17 + y * 11) % 256) as u8,
                ((x * 37 + y * 3) % 256) as u8,
                ((x * 5 + y * 71) % 256) as u8,
            ])
        });
        let (canvas, fitted) = contain_rgb(&image, w, h);
        assert_eq!(
            (fitted.width, fitted.height, fitted.left, fitted.top),
            placement
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(canvas.as_raw())),
            digest,
            "{width}x{height} → {w}x{h}"
        );
    }
}
