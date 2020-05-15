extern crate chrono;
extern crate fern;
extern crate image;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate rand;
extern crate rlua;
extern crate rusttype;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate serde_xml_rs;
extern crate specs;
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use std::fs::File;
use std::io::BufReader;

mod core;
mod error;
mod windowing;
pub mod graphics;
pub mod position;
pub mod script;
pub mod state;

pub use image::ImageFormat;

pub fn assemble() -> Result<core::ClankEngine, Box<dyn std::error::Error>> {
    let mut windowing = windowing::WindowSystem::new();

    let graphics_system = graphics::GraphicsSystem::new(&mut windowing)?;

    let mut engine = core::ClankEngine::new(windowing);

    let config: graphics::sprite::SpriteConfig =
        serde_json::from_reader(BufReader::new(File::open("resources/SpriteConfig.json")?))?;

    engine.register::<graphics::Graphics>();
    engine.register::<script::Script>();
    engine.register::<graphics::anim::Animation>();
    engine.register::<position::Position>();
    engine.register::<graphics::sprite::Sprite>();

    engine.insert(config);

    Ok(engine
        .register_system(graphics::sprite::SpriteSystem, "sprite", &[])
        .register_system(graphics::anim::AnimationSystem, "animation", &["sprite"])
        .register_system(graphics_system, "graphics", &["animation"]))
}

pub fn new() -> core::Clank {
    core::Clank::new()
}
