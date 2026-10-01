use super::*;
#[derive(Clone)]
pub(super) struct Diagram {
    pub base: i64,
    pub frets: Vec<i64>,
    pub fingers: Vec<i64>,
    pub barres: Vec<[i64; 3]>,
}
pub(super) fn parse(text: &str) -> R<Diagram> {
    let fs = text.split(':').collect::<Vec<_>>();
    if fs.len() != 4 {
        return Err("Invalid chord diagram fields".into());
    }
    let integer = |s: &str| {
        s.parse::<i64>()
            .map_err(|_| "Invalid diagram integer".to_owned())
    };
    let base = integer(fs[0])?;
    let frets = fs[1]
        .split('/')
        .map(|s| if s == "x" { Ok(-1) } else { integer(s) })
        .collect::<R<Vec<_>>>()?;
    let fingers = fs[2]
        .split('/')
        .map(|s| if s == "-" { Ok(-2) } else { integer(s) })
        .collect::<R<Vec<_>>>()?;
    if !(1..=12).contains(&frets.len())
        || frets.len() != fingers.len()
        || !(1..=36).contains(&base)
        || frets
            .iter()
            .any(|n| !(-1..=36).contains(n) || *n > 0 && *n < base)
        || fingers.iter().any(|n| *n != -2 && !(0..=4).contains(n))
    {
        return Err("Invalid chord diagram range".into());
    }
    let mut barres = vec![];
    if fs[3] != "-" {
        for b in fs[3].split(';') {
            let b = b.split('/').map(integer).collect::<R<Vec<_>>>()?;
            if b.len() != 3
                || b[0] < base
                || b[0] > 36
                || b[1] < 0
                || b[1] >= b[2]
                || b[2] >= frets.len() as i64
            {
                return Err("Invalid chord barre".into());
            }
            barres.push([b[0], b[1], b[2]]);
        }
    }
    Ok(Diagram {
        base,
        frets,
        fingers,
        barres,
    })
}
impl Diagram {
    pub fn project(&self, high_to_low: &[usize]) -> Self {
        let indexes = high_to_low
            .iter()
            .rev()
            .map(|i| self.frets.len() as i64 - 1 - *i as i64)
            .collect::<Vec<_>>();
        let at = |source: &[i64], i: i64, missing: i64| {
            if i >= 0 {
                source.get(i as usize).copied().unwrap_or(missing)
            } else {
                missing
            }
        };
        let mut barres = vec![];
        for b in &self.barres {
            let covered = indexes
                .iter()
                .enumerate()
                .filter(|(_, i)| b[1] <= **i && **i <= b[2])
                .map(|(j, _)| j as i64)
                .collect::<Vec<_>>();
            if covered.len() > 1 {
                barres.push([b[0], covered[0], *covered.last().unwrap()]);
            }
        }
        Self {
            base: self.base,
            frets: indexes.iter().map(|i| at(&self.frets, *i, -1)).collect(),
            fingers: indexes.iter().map(|i| at(&self.fingers, *i, -2)).collect(),
            barres,
        }
    }
}
