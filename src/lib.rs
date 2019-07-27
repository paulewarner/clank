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
extern crate serde;
extern crate serde_json;
extern crate specs;
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use std::fs::File;
use std::io::BufReader;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use winit::EventsLoop;

use specs::prelude::*;

pub mod anim;
mod core;
pub mod graphics;
pub mod position;
pub mod script;
pub mod sprite;

pub fn assemble<'a, 'b>() -> Result<core::ClankEngine<'a, 'b>, Box<std::error::Error>> {
    let world = World::new();
    let events_loop = EventsLoop::new();

    let swapchain_flag = Arc::new(AtomicBool::new(false));

    let graphics_system = graphics::GraphicsSystem::new(&events_loop, swapchain_flag.clone())?;

    let dispatcher = DispatcherBuilder::new();

    let mut engine = core::ClankEngine::new(world, dispatcher, swapchain_flag, events_loop);

    let config: sprite::SpriteConfig =
        serde_json::from_reader(BufReader::new(File::open("SpriteConfig.json")?))?;

    engine.register::<graphics::Graphics>();
    engine.register::<script::Script>();
    engine.register::<anim::Animation>();
    engine.register::<position::Position>();
    engine.register::<sprite::Sprite>();

    engine.insert(config);

    Ok(engine
        .register_system(sprite::SpriteSystem, "sprite", &[])
        .register_system(anim::AnimationSystem, "animation", &["sprite"])
        .register_system(graphics_system, "graphics", &["animation"]))
}

pub fn new() -> core::Clank {
    core::Clank::new()
}
