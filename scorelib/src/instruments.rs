//! Musical instrument names and their General MIDI programs.

/// Printed instrument names take precedence over a conflicting model program.
pub fn program_from_visible_name(name: &str) -> Option<i64> {
    let lowered = name.to_lowercase();
    let tokens: Vec<&str> = lowered
        .split(|c: char| !c.is_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    let text = tokens.join(" ");
    let abbreviations = [
        ("pno", 0),
        ("el pno", 4),
        ("hpsi", 6),
        ("clav", 7),
        ("org", 16),
        ("acc", 21),
        ("harm", 22),
        ("chr", 52),
        ("syn chr", 54),
        ("vln", 40),
        ("vla", 41),
        ("vlc", 42),
        ("fl", 73),
        ("picc", 72),
        ("ob", 68),
        ("clar", 71),
        ("bn", 70),
        ("tpt", 56),
        ("tbn", 57),
        ("tba", 58),
        ("fhn", 60),
        ("hp", 46),
        ("sax", 65),
        ("saxophone", 65),
        ("vib", 11),
        ("rec", 74),
        ("whis", 78),
    ];
    if let Some((_, program)) = abbreviations.iter().find(|(abbr, _)| *abbr == text) {
        return Some(*program);
    }
    let padded = format!(" {text} ");
    for (names, program) in [
        (&["piano"][..], 0),
        (&["violin"][..], 40),
        (&["viola"][..], 41),
        (&["cello", "violoncello"][..], 42),
        (&["contrabass"][..], 43),
        (&["flute"][..], 73),
        (&["oboe"][..], 68),
        (&["clarinet"][..], 71),
        (&["bassoon"][..], 70),
        (&["english horn", "cor anglais"][..], 69),
        (&["soprano sax", "soprano saxophone"][..], 64),
        (&["alto sax", "alto saxophone"][..], 65),
        (&["tenor sax", "tenor saxophone"][..], 66),
        (&["baritone sax", "baritone saxophone"][..], 67),
        (&["trumpet"][..], 56),
        (&["french horn", "horn in f", "f horn"][..], 60),
    ] {
        if names
            .iter()
            .any(|name| padded.contains(&format!(" {name} ")))
        {
            return Some(program);
        }
    }
    None
}
