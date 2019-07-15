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
extern crate specs;
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use winit::EventsLoop;

use specs::prelude::*;

pub mod anim;
mod core;
pub mod graphics;
pub mod position;
pub mod script;

pub fn assemble<'a, 'b>() -> core::ClankEngine<'a, 'b> {
    let world = World::new();
    let events_loop = EventsLoop::new();

    let swapchain_flag = Arc::new(AtomicBool::new(false));

    let graphics_system = graphics::GraphicsSystem::new(&events_loop, swapchain_flag.clone())
        .map_err(|e| error!("Failed to initalize graphics subsystem: {}", e))
        .expect("Failed to create graphics subsystem");

    let dispatcher = DispatcherBuilder::new();

    let mut engine = core::ClankEngine::new(world, dispatcher, swapchain_flag, events_loop);

    engine.register::<graphics::Graphics>();
    engine.register::<script::Script>();
    engine.register::<anim::Animation>();
    engine.register::<position::Position>();

    engine
        .register_system(graphics_system, "graphics", &[])
        .register_system(anim::AnimationSystem, "animation", &["graphics"])
}

pub fn new() -> core::Clank {
    core::Clank::new()
}
