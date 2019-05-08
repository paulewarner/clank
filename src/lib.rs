extern crate chrono;
extern crate fern;
extern crate image;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate rand;
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

pub mod graphics;
mod map;
mod core;
pub mod script;

struct Combatant {
    hp: u32,
    ai: Arc<
        Fn(Entity, &Combatant, (&Entities, &ReadStorage<Combatant>), &mut Read<LazyUpdate>)
            + Send
            + Sync,
    >,
}

impl std::fmt::Debug for Combatant {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "Combatant {{ hp: {} }}", self.hp)
    }
}

impl Component for Combatant {
    type Storage = VecStorage<Self>;
}

struct CombatSystem;

impl<'a> System<'a> for CombatSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Combatant>,
        Read<'a, LazyUpdate>,
    );

    fn run(&mut self, (entities, combatants, mut updater): Self::SystemData) {
        for (entity, combatant) in (&entities, &combatants).join() {
            let f = combatant.ai.clone();
            f(entity, combatant, (&entities, &combatants), &mut updater);
        }
    }
}

pub fn assemble<'a, 'b>() -> core::ClankEngine<'a, 'b> {
    let world = World::new();
    let events_loop = EventsLoop::new();

    let swapchain_flag = Arc::new(AtomicBool::new(false));

    let graphics_system = graphics::GraphicsSystem::new(&events_loop, swapchain_flag.clone())
        .map_err(|e| error!("Failed to initalize graphics subsystem: {}", e))
        .expect("Failed to create graphics subsystem");

    let dispatcher = DispatcherBuilder::new();

    let mut engine = core::ClankEngine::new(
        world,
        dispatcher,
        swapchain_flag,
        events_loop
    );

    core::setup_logger().expect("Failed to setup logging");

    engine.register::<graphics::Graphics>();
    engine.register::<script::Script>();

    engine.register_system(graphics_system, "graphics", &[])
}

pub fn new() -> core::Clank {
    core::Clank::new()
}