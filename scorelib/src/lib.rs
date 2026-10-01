//! Shared score representation, musical rules and GP5/MusicXML conversion.
pub mod gp5_constraints;
pub mod music_exports;
#[cfg(feature = "python")]
mod python;
pub mod score;
