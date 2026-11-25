//! The worlds the chip passes through after it erupts.
//!
//! Each theme is a [`Stage`]: its own world around the centred chip, in its own visual style,
//! with the chip restyled to belong there. The camera and the chip's pose come from the
//! director and carry on across cuts, so each cut reads as the same chip changing skin.
//!
//! To add a theme: write a module here that builds a `Box<dyn Stage>`, and add it to [`all`].

use std::ops::Range;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Chip;
use crate::director::{INK, Phase};
use crate::shaders::ViewRef;
use crate::shaders::composite::Grade;
use crate::sound::world::WorldSound;

pub mod arctic_aurora;
pub mod ascii_terminal;
pub mod baroque_gold;
pub mod bio_cell;
pub mod black_hole;
pub mod blueprint;
pub mod bokeh_macro;
pub mod candy_land;
pub mod cave_wall;
pub mod clay_stopmotion;
pub mod crt_vector;
pub mod crystal_geode;
pub mod deep_nebula;
pub mod desert_dunes;
pub mod earth;
pub mod flipbook_bw;
pub mod forest;
pub mod glitch;
pub mod glyph_rain;
pub mod halftone;
pub mod kaleidoscope;
pub mod kit;
pub mod knitted_wool;
pub mod lava_lamp;
pub mod lowpoly_peaks;
pub mod marble_hall;
pub mod moonlit_graveyard;
pub mod night_city;
pub mod night_vision;
pub mod oil_meadow;
pub mod orbital_station;
pub mod overlay;
pub mod papercraft;
pub mod parchment_map;
pub mod pastel_vapour;
pub mod pcb_city;
pub mod pencil;
pub mod reef;
pub mod retro_8bit;
pub mod roman_mosaic;
pub mod savanna_silhouette;
pub mod snow_globe;
pub mod stained_glass;
pub mod steampunk_works;
pub mod sumi_e;
pub mod synthwave_grid;
pub mod thermal;
pub mod tilt_shift_diorama;
pub mod topographic;
pub mod toy_brick;
pub mod tropical_island;
pub mod ukiyo_e;
pub mod volcano_crater;
pub mod voxel_world;
pub mod watercolour;
pub mod wireframe_grid;
pub mod xray;
pub mod zen_garden;

/// What a stage needs to draw one frame.
pub struct Scene<'a> {
    /// The camera, with the frame's state (time, the chip's pose) in `view.frame`.
    pub view: ViewRef,
    pub chip: &'a Chip,
    /// A full-screen quad, for backdrops and screen-space passes (see `shaders::screen`).
    pub quad: &'a Model<ObjVertex>,
    /// Seconds since this stage came on screen.
    pub local_time: f32,
    /// The scene target's size in pixels, for stages that render targets of their own.
    pub extent: Extent,
}

/// Something that fills the screen around the chip: the studio, or a themed world.
pub trait Stage {
    /// Renders whatever the stage needs before the scene pass: shadow maps, reflections,
    /// textures of its own. Nothing by default.
    fn prepare(&mut self, _context: &mut RenderContext<'_>, _scene: &Scene<'_>) {}

    /// Draws the stage's world and the chip into the HDR scene pass. Opaque surfaces write their
    /// view distance to alpha (`shaders::scene_colour`); where nothing is drawn alpha is
    /// `shaders::SKY_DISTANCE`.
    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>);

    /// Restyles the finished HDR scene, for stages whose look is a screen-space effect: renders
    /// into a target of the stage's own and returns it, keeping view distance in alpha. `None`
    /// (the default) keeps the scene as it is.
    fn stylise(
        &mut self,
        _context: &mut RenderContext<'_>,
        _scene: &Scene<'_>,
        _image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        None
    }

    /// The colour of a montage word set over the stage's lower third: the captions' off-white by
    /// default. A stage whose ground is light gives a dark ink from its own palette.
    fn caption_ink(&self) -> Vec3 {
        Vec3::new(INK, INK, INK)
    }

    /// The stage's part in the soundtrack: by default the tune on short strings, with nothing of
    /// its own.
    fn sound(&self) -> WorldSound {
        WorldSound::default()
    }

    /// How the finished image is graded.
    fn grade(&self) -> Grade {
        Grade::default()
    }
}

/// A themed world and its name.
pub struct Theme {
    /// A short kebab-case name, for logs and the `--theme` option.
    pub name: &'static str,
    pub stage: Box<dyn Stage>,
}

/// Every theme, in running order.
pub fn all() -> Result<Vec<Theme>, Box<dyn std::error::Error>> {
    Ok(vec![
        Theme {
            name: "reef",
            stage: Box::new(reef::Reef::new()),
        },
        Theme {
            name: "tropical-island",
            stage: Box::new(tropical_island::TropicalIsland::new()),
        },
        Theme {
            name: "earth",
            stage: Box::new(earth::Earth::new()),
        },
        Theme {
            name: "volcano-crater",
            stage: Box::new(volcano_crater::VolcanoCrater::new()),
        },
        Theme {
            name: "voxel-world",
            stage: Box::new(voxel_world::VoxelWorld::new()),
        },
        Theme {
            name: "night-city",
            stage: Box::new(night_city::NightCity::new()),
        },
        Theme {
            name: "forest",
            stage: Box::new(forest::Forest::new()?),
        },
        Theme {
            name: "synthwave-grid",
            stage: Box::new(synthwave_grid::SynthwaveGrid::new()),
        },
        Theme {
            name: "toy-brick",
            stage: Box::new(toy_brick::ToyBrick::new()),
        },
        Theme {
            name: "stained-glass",
            stage: Box::new(stained_glass::StainedGlass::new()),
        },
        Theme {
            name: "steampunk-works",
            stage: Box::new(steampunk_works::SteampunkWorks::new()),
        },
        Theme {
            name: "black-hole",
            stage: Box::new(black_hole::BlackHole::new()),
        },
        Theme {
            name: "parchment-map",
            stage: Box::new(parchment_map::ParchmentMap::new()),
        },
        Theme {
            name: "snow-globe",
            stage: Box::new(snow_globe::SnowGlobe::new()),
        },
        Theme {
            name: "crt-vector",
            stage: Box::new(crt_vector::CrtVector::new()),
        },
        Theme {
            name: "blueprint",
            stage: Box::new(blueprint::Blueprint::new()),
        },
        Theme {
            name: "halftone",
            stage: Box::new(halftone::ComicHalftone::new()),
        },
        Theme {
            name: "baroque-gold",
            stage: Box::new(baroque_gold::BaroqueGold::new()),
        },
        Theme {
            name: "retro-8bit",
            stage: Box::new(retro_8bit::Retro8bit::new()),
        },
        Theme {
            name: "ukiyo-e",
            stage: Box::new(ukiyo_e::UkiyoE::new()),
        },
        Theme {
            name: "glyph-rain",
            stage: Box::new(glyph_rain::GlyphRain::new()),
        },
        Theme {
            name: "tilt-shift-diorama",
            stage: Box::new(tilt_shift_diorama::TiltShiftDiorama::new()),
        },
        Theme {
            name: "savanna-silhouette",
            stage: Box::new(savanna_silhouette::SavannaSilhouette::new()),
        },
        Theme {
            name: "arctic-aurora",
            stage: Box::new(arctic_aurora::ArcticAurora::new()),
        },
        Theme {
            name: "orbital-station",
            stage: Box::new(orbital_station::OrbitalStation::new()),
        },
        Theme {
            name: "desert-dunes",
            stage: Box::new(desert_dunes::DesertDunes::new()),
        },
        Theme {
            name: "papercraft",
            stage: Box::new(papercraft::Papercraft::new()),
        },
        Theme {
            name: "bio-cell",
            stage: Box::new(bio_cell::BioCell::new()),
        },
        Theme {
            name: "oil-meadow",
            stage: Box::new(oil_meadow::OilMeadow::new()),
        },
        Theme {
            name: "moonlit-graveyard",
            stage: Box::new(moonlit_graveyard::MoonlitGraveyard::new()),
        },
        Theme {
            name: "candy-land",
            stage: Box::new(candy_land::CandyLand::new()),
        },
        Theme {
            name: "cave-wall",
            stage: Box::new(cave_wall::CaveWall::new()),
        },
        Theme {
            name: "marble-hall",
            stage: Box::new(marble_hall::MarbleHall::new()),
        },
        Theme {
            name: "lowpoly-peaks",
            stage: Box::new(lowpoly_peaks::LowpolyPeaks::new()),
        },
        Theme {
            name: "thermal",
            stage: Box::new(thermal::Thermal::new()),
        },
        Theme {
            name: "clay-stopmotion",
            stage: Box::new(clay_stopmotion::ClayStopmotion::new()),
        },
        Theme {
            name: "pastel-vapour",
            stage: Box::new(pastel_vapour::PastelVapour::new()),
        },
        Theme {
            name: "xray",
            stage: Box::new(xray::Xray::new()),
        },
        Theme {
            name: "kaleidoscope",
            stage: Box::new(kaleidoscope::Kaleidoscope::new()),
        },
        Theme {
            name: "knitted-wool",
            stage: Box::new(knitted_wool::KnittedWool::new()),
        },
        Theme {
            name: "deep-nebula",
            stage: Box::new(deep_nebula::DeepNebula::new()),
        },
        Theme {
            name: "zen-garden",
            stage: Box::new(zen_garden::ZenGarden::new()),
        },
        Theme {
            name: "watercolour",
            stage: Box::new(watercolour::Watercolour::new()),
        },
        Theme {
            name: "night-vision",
            stage: Box::new(night_vision::NightVision::new()),
        },
        Theme {
            name: "pcb-city",
            stage: Box::new(pcb_city::PcbCity::new()),
        },
        Theme {
            name: "lava-lamp",
            stage: Box::new(lava_lamp::LavaLamp::new()),
        },
        Theme {
            name: "roman-mosaic",
            stage: Box::new(roman_mosaic::RomanMosaic::new()),
        },
        Theme {
            name: "flipbook-bw",
            stage: Box::new(flipbook_bw::FlipbookBw::new()),
        },
        Theme {
            name: "pencil",
            stage: Box::new(pencil::Pencil::new()),
        },
        Theme {
            name: "topographic",
            stage: Box::new(topographic::Topographic::new()),
        },
        Theme {
            name: "sumi-e",
            stage: Box::new(sumi_e::SumiE::new()),
        },
        Theme {
            name: "ascii-terminal",
            stage: Box::new(ascii_terminal::AsciiTerminal::new()),
        },
        Theme {
            name: "bokeh-macro",
            stage: Box::new(bokeh_macro::BokehMacro::new()),
        },
        Theme {
            name: "crystal-geode",
            stage: Box::new(crystal_geode::CrystalGeode::new()),
        },
        Theme {
            name: "glitch",
            stage: Box::new(glitch::Glitch::new()),
        },
        Theme {
            name: "wireframe-grid",
            stage: Box::new(wireframe_grid::WireframeGrid::new()),
        },
    ])
}

/// When each world is on screen during the worlds phase, as a run of cuts.
///
/// The main run shows every theme once in running order: the first stays longest, and each cut
/// comes sooner than the last until they settle at a pace that still reads. The finale then
/// re-picks worlds in cuts that shrink to a single frame, so they blur together.
pub struct Schedule {
    cuts: Vec<Cut>,
}

/// One cut: when it starts and the theme it shows.
struct Cut {
    start: f32,
    theme: usize,
}

impl Schedule {
    /// How long the first world stays on screen: earth's length when the user judged it the right
    /// longest.
    const FIRST: f32 = 2.2;
    /// The main run's shortest cut: a little longer than the shortest the user judged readable
    /// (0.3 s), as they asked.
    const FLOOR: f32 = 0.36;
    /// How long the finale lasts, at the end of the worlds phase.
    pub const FINALE: f32 = 3.25;
    /// The finale's first cut (provisional, set by eye).
    const FINALE_FIRST: f32 = 0.22;
    /// The finale's shortest cut: one frame at 60 fps.
    const FINALE_FLOOR: f32 = 1.0 / 60.0;
    /// Enough finale cuts to fill [`Self::FINALE`] at its pace.
    const FINALE_CUTS: usize = 64;
    /// The step between the finale's picks in running order; coprime with the theme count, so it
    /// visits every theme before repeating one.
    const FINALE_STRIDE: usize = 23;

    pub fn new(count: usize) -> Self {
        if count == 0 {
            return Self { cuts: Vec::new() };
        }
        let (start, end) = (Phase::Worlds.start(), Phase::Worlds.end());
        let main = shrinking(Self::FIRST, Self::FLOOR, count, end - start - Self::FINALE);
        let finale = shrinking(
            Self::FINALE_FIRST,
            Self::FINALE_FLOOR,
            Self::FINALE_CUTS,
            Self::FINALE,
        );
        let themes =
            (0..count).chain((0..Self::FINALE_CUTS).map(|i| (i * Self::FINALE_STRIDE + 5) % count));
        let cuts = main
            .into_iter()
            .chain(finale)
            .zip(themes)
            .scan(start, |time, (duration, theme)| {
                let cut = Cut {
                    start: *time,
                    theme,
                };
                *time += duration;
                Some(cut)
            })
            .collect();
        Self { cuts }
    }

    /// The cut on screen at `time`, and the seconds since it came on. Cuts are numbered from 0;
    /// the main run's cut `i` shows theme `i`.
    pub fn at(&self, time: f32) -> Option<(usize, f32)> {
        if !(Phase::Worlds.start()..Phase::Worlds.end()).contains(&time) {
            return None;
        }
        let index = self.cuts.iter().rposition(|cut| cut.start <= time)?;
        Some((index, time - self.cuts[index].start))
    }

    /// How many cuts there are.
    pub fn cut_count(&self) -> usize {
        self.cuts.len()
    }

    /// The theme the cut at `index` shows.
    ///
    /// # Panics
    ///
    /// Panics if there is no cut at `index`.
    pub fn theme(&self, index: usize) -> usize {
        self.cuts[index].theme
    }

    /// When the cut at `index` comes on and goes off.
    ///
    /// # Panics
    ///
    /// Panics if there is no cut at `index`.
    pub fn slot(&self, index: usize) -> Range<f32> {
        let end = self
            .cuts
            .get(index + 1)
            .map_or(Phase::Worlds.end(), |cut| cut.start);
        self.cuts[index].start..end
    }
}

/// `count` durations shrinking geometrically from `first` down to `floor`, then holding it, that
/// fill `length`, with the ratio found by bisection. Too few to need shrinking share `length`
/// equally.
fn shrinking(first: f32, floor: f32, count: usize, length: f32) -> Vec<f32> {
    if count as f32 * first <= length {
        return vec![length / count as f32; count];
    }
    let durations = |ratio: f32| -> Vec<f32> {
        (0..count)
            .map(|i| (first * ratio.powi(i as i32)).max(floor))
            .collect()
    };
    let (mut low, mut high) = (0.0_f32, 1.0_f32);
    for _ in 0..40 {
        let ratio = (low + high) * 0.5;
        if durations(ratio).iter().sum::<f32>() > length {
            high = ratio;
        } else {
            low = ratio;
        }
    }
    let mut result = durations(low);
    let stretch = length / result.iter().sum::<f32>();
    result.iter_mut().for_each(|d| *d *= stretch);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length(slot: Range<f32>) -> f32 {
        slot.end - slot.start
    }

    #[test]
    fn main_run_shows_every_theme_once_with_ever_shorter_cuts_down_to_the_floor() {
        let schedule = Schedule::new(56);
        let durations: Vec<f32> = (0..56).map(|i| length(schedule.slot(i))).collect();
        assert!((durations[0] - Schedule::FIRST).abs() < 0.05);
        assert!(durations.windows(2).all(|pair| pair[1] <= pair[0] + 1e-4));
        assert!(durations[55] >= Schedule::FLOOR * 0.95);
        assert!((0..56).all(|i| schedule.theme(i) == i));
        assert_eq!(schedule.at(Phase::Worlds.start()), Some((0, 0.0)));
        assert_eq!(schedule.at(Phase::Worlds.start() - 1.0), None);
    }

    #[test]
    fn finale_repicks_worlds_in_cuts_shrinking_to_a_frame_until_the_phase_ends() {
        let schedule = Schedule::new(56);
        let finale_start = Phase::Worlds.end() - Schedule::FINALE;
        let (first, _) = schedule.at(finale_start + 1e-3).unwrap();
        assert_eq!(first, 56);
        let (last, _) = schedule.at(Phase::Worlds.end() - 1e-3).unwrap();
        assert!(length(schedule.slot(last)) < 1.5 * Schedule::FINALE_FLOOR);
        assert!(((first + 1)..=last).all(|i| schedule.theme(i) != schedule.theme(i - 1)));
        assert!((first..=last).all(|i| schedule.theme(i) < 56));
        assert_eq!(schedule.slot(last).end, Phase::Worlds.end());
    }
}
