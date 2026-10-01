//! Rust-only building blocks for the desktop client. No Python runtime or server.
pub mod llama;
pub mod project;

pub mod music_exports;
pub mod score;

pub mod auxiliary_onnx;
pub mod image_boundary;

pub mod gp5_constraints;

pub mod layout_postprocess;

pub mod image_transforms;

pub mod staff_geometry;

pub mod score_grid;

pub mod score_structure;

pub mod recognition;

pub mod staff_classifier;

pub mod device_selection;

pub mod ottava_geometry;

pub mod pixel_refinement;
