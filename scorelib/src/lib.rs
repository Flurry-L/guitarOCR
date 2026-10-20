//! Shared score representation, musical rules and GP5/MusicXML conversion.
pub mod chords;
pub mod gp5_constraints;
pub mod information;
pub mod instruments;
pub mod music_exports;
pub mod observation;
pub mod pitch;
#[cfg(feature = "python")]
mod python;
pub mod score;
pub mod types;
pub mod visual_techniques;
