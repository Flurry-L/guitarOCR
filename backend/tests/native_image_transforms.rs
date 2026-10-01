//! Small checked-in goldens generated once with OpenCV 4.13.0 and Pillow 12.3.0.
//! No model, native ONNX library, Python interpreter, or download is used by tests.
//! Source formula: RGB=(53*x + 97*y + 71*c + 13*x*y) % 256; grayscale
//! uses c=0 and repeats the resulting value in all three RGB channels.
use guitarocr_backend::auxiliary_onnx::{layout_input, signature_input};
use guitarocr_backend::image_transforms::{resize_rgb_cubic, resize_rgb_lanczos};
use image::{Rgb, RgbImage};

struct Fixture {
    name: &'static str,
    input: (u32, u32),
    output: (u32, u32),
    gray: bool,
    cubic: &'static str,
    lanczos: &'static str,
}

fn source(width: u32, height: u32, gray: bool) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        Rgb(std::array::from_fn(|c| {
            ((53 * x + 97 * y + if gray { 0 } else { 71 * c as u32 } + 13 * x * y) % 256) as u8
        }))
    })
}

fn unhex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

const FIXTURES: &[Fixture] = &[
    Fixture { name: "rgb_up", input: (5,4), output: (9,7), gray: false,
        cubic: "00387d04499a2267c74094ff5ebeff88f983b5e015d35736e40058146cbe2f86a759a88b8897a69c7cc381a28c70b062927684ac4ba14fa3ff6bc9b298f141d79938e42e75793c962179bd499de26fb3fab251a58c728a5d9e68768a87a26ab082938d6eb27e9a90b8bb75e3cf0046831c621b4b8f1c7bd561a5ec8bea8bbbeb3aea8363ff378861368d565f6d4c94467b7d67a9579e8385a56cb487a1aa57ca9f37076ace339b7a79d50ace810ae7155c7b2eba2783c961c95892f503",
        lanczos: "003476013e8f1d60c7388fff57c4ff86ff81bbe110dd4a2de8005c0f69bc2484a756a68d8d96ac9a7fc67ea78c72b05f95717eb0439e4f9bff63c9ba94ef45e19b30e42e75773399237cbb49a3e378b1f5c34da397718b5e9e69748b88a36ab182938b6eb2809b90c4bb71f5d2004887195c1f4d8c157ade61a5ec8cf28ab9e73bed7c60ff318d5c33934f5e6e4c9644827e5eab549b817eaf6ab387a1ae4dcaa02c0168dd28a4867adc0ae58500ee0f567717c92182cb59d55595fb00",
    },
    Fixture { name: "rgb_down", input: (9,7), output: (4,3), gray: false,
        cubic: "6ad280497c938bbe4440b39d54b73e426cc8e4338f76bd44836fc16875bc5b67b25c6e82",
        lanczos: "5a819f8f9a93837f87789d8584736c877889a398777d7c8f9d6796864599827388868a8c",
    },
    Fixture { name: "gray_up", input: (4,3), output: (7,5), gray: true,
        cubic: "000000050505252525424242656565949494b2b2b2171717353535636363949494a2a2a27f7f7f6161615b5b5b7373739c9c9cdcdcdcdddddd6c6c6c141414acacac8383835353537373739d9d9d8585856d6d6ddfdfdf8282820b0b0b0e0e0e5d5d5d9d9d9dc7c7c7",
        lanczos: "0000000202022020203838386060609a9a9ab8b8b81111112929296060609898989e9e9e7d7d7d6161615c5c5c6d6d6d9a9a9ae9e9e9dbdbdb6363630f0f0fb9b9b98a8a8a4f4f4f6a6a6a9a9a9a8787876f6f6fe3e3e38989890909090000005a5a5aa8a8a8cccccc",
    },
    Fixture { name: "gray_down", input: (8,6), output: (3,2), gray: true,
        cubic: "949494303030d7d7d7ebebeb747474ededed",
        lanczos: "666666909090898989939393888888929292",
    },
    Fixture { name: "single_column", input: (1,4), output: (7,2), gray: false,
        cubic: "2786cd2786cd2786cd2786cd2786cd2786cd2786cd7c2b727c2b727c2b727c2b727c2b727c2b727c2b72",
        lanczos: "4669b04669b04669b04669b04669b04669b04669b078488f78488f78488f78488f78488f78488f78488f",
    },
    Fixture { name: "single_row", input: (4,1), output: (2,7), gray: false,
        cubic: "165da489d097165da489d097165da489d097165da489d097165da489d097165da489d097165da489d097",
        lanczos: "1d64bc82c99a1d64bc82c99a1d64bc82c99a1d64bc82c99a1d64bc82c99a1d64bc82c99a1d64bc82c99a",
    },
    Fixture { name: "mixed_scale", input: (17,3), output: (5,9), gray: false,
        cubic: "3582ea501181aef52062a489374a954289d6562781a8ef365ea5883b4f9c71a498686f7c9ae17852a8854659aca7c34f7dc27788cfc645ab825366bfbacd298ff28071b8ff37ac7e5c76d590b64395da9b5aa1ff30aa795a84e6498e829299ba488fe52ea676518eef0c6cb98f60d53a81c82da2734996f60061c98f50df347bc22ca172479afa",
        lanczos: "4886ad707b7273855572986d7987915389a9727d78778a6279936d7e828b648fa2757f817e9378858b6c877a817b95987c848d889e9493806b9271758c948f888d9791a2a59a7167956f6f8a888c96979794999c9266628b79777e7a8fa19d91928c8683625e7c8785727092a7a18b8f817377615b7092906b6a94aba4888e7a68706059699997",
    },
    Fixture { name: "same_size", input: (3,2), output: (3,2), gray: false,
        cubic: "00478e357cc36ab1f861a8efa3ea31e52c73",
        lanczos: "00478e357cc36ab1f861a8efa3ea31e52c73",
    },
    Fixture { name: "single_pixel", input: (1,1), output: (5,3), gray: false,
        cubic: "00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e",
        lanczos: "00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e00478e",
    },
];

#[test]
fn pillow_lanczos_matches_rgb_and_gray_goldens_exactly() {
    for fixture in FIXTURES {
        let input = source(fixture.input.0, fixture.input.1, fixture.gray);
        let actual = resize_rgb_lanczos(&input, fixture.output.0, fixture.output.1);
        assert_eq!(actual.dimensions(), fixture.output, "{}", fixture.name);
        assert_eq!(
            actual.into_raw(),
            unhex(fixture.lanczos),
            "{}",
            fixture.name
        );
    }
}

#[test]
fn opencv_cubic_matches_goldens_with_at_most_one_byte_rounding_difference() {
    for fixture in FIXTURES {
        let input = source(fixture.input.0, fixture.input.1, fixture.gray);
        let actual = resize_rgb_cubic(&input, fixture.output.0, fixture.output.1);
        assert_eq!(actual.dimensions(), fixture.output, "{}", fixture.name);
        let expected = unhex(fixture.cubic);
        assert_eq!(actual.as_raw().len(), expected.len());
        for (index, (&a, &e)) in actual.as_raw().iter().zip(&expected).enumerate() {
            assert!(
                a.abs_diff(e) <= 1,
                "{}: byte {index}: {a} != {e}",
                fixture.name
            );
        }
    }
}

#[test]
fn same_size_and_constant_colors_are_preserved() {
    let input = source(7, 5, false);
    for resize in [resize_rgb_cubic, resize_rgb_lanczos] {
        assert_eq!(resize(&input, 7, 5), input);
        for color in [[0, 0, 0], [255, 255, 255], [17, 83, 201]] {
            let constant = RgbImage::from_pixel(1, 1, Rgb(color));
            assert!(resize(&constant, 9, 7)
                .pixels()
                .all(|pixel| pixel.0 == color));
        }
    }
}

#[test]
fn cubic_scalar_tie_rounds_up_at_final_quantization() {
    let input = RgbImage::from_fn(2, 1, |x, _| Rgb([x as u8; 3]));
    assert_eq!(resize_rgb_cubic(&input, 1, 1).get_pixel(0, 0).0, [1; 3]);
}

#[test]
fn empty_dimensions_fail_explicitly() {
    for resize in [resize_rgb_cubic, resize_rgb_lanczos] {
        for (sw, sh, dw, dh) in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
            let input = RgbImage::new(sw, sh);
            assert!(std::panic::catch_unwind(|| resize(&input, dw, dh)).is_err());
        }
    }
}

#[test]
fn normalized_model_inputs_are_planar_rgb_with_expected_ranges() {
    let color = [17, 83, 201];
    let input = RgbImage::from_pixel(1, 1, Rgb(color));
    let layout = layout_input(&input).unwrap();
    assert_eq!(layout.len(), 3 * 800 * 800);
    for (channel, values) in layout.chunks_exact(800 * 800).enumerate() {
        assert!(values
            .iter()
            .all(|&value| value == f32::from(color[channel]) / 255.0));
    }
    let signature = signature_input(&input).unwrap();
    assert_eq!(signature.len(), 2 * 3 * 512 * 192);
    for view in signature.chunks_exact(3 * 512 * 192) {
        for (channel, plane) in view.chunks_exact(512 * 192).enumerate() {
            let normalized = (f32::from(color[channel]) / 255.0 - 0.5) / 0.5;
            for row in plane.chunks_exact(512) {
                assert!(row[..192].iter().all(|&value| value == normalized));
                assert!(row[192..].iter().all(|&value| value == 1.0));
            }
        }
    }
    assert!(layout_input(&RgbImage::new(0, 1)).is_err());
    assert!(signature_input(&RgbImage::new(1, 0)).is_err());
}
