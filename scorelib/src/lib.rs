//! Shared score representation, musical rules and GP5/MusicXML conversion.
pub mod chords;
pub mod gp5_constraints;
pub mod music_exports;
pub mod observation;
#[cfg(feature = "python")]
mod python;
pub mod score;
pub mod visual_techniques;
