extern crate rand;
extern crate specs;
#[macro_use]
extern crate log;
extern crate chrono;
extern crate fern;
extern crate image;
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::mpsc::channel;

use image::ImageFormat;

use winit::{Event, EventsLoop, WindowEvent};

use specs::prelude::*;

mod graphics;
mod map;
mod events;

fn setup_logger() -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record
                    .file()
                    .and_then(|x| record.line().map(|y| format!("{}:{}", x, y)))
                    .unwrap_or(String::from(record.target())),
                record.level(),
                message
            ))
        })
        .level(log::LevelFilter::Trace)
        .chain(std::io::stdout())
        .chain(fern::log_file("output.log")?)
        .apply()?;
    Ok(())
}

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

fn main() {
    setup_logger().expect("Failed to setup logging");

    let mut world = World::new();
    world.register::<Combatant>();
    world.register::<map::Presence>();
    world.register::<map::Move>();
    world.register::<map::Space>();
    world.register::<graphics::Graphics>();
    world.register::<events::EventHandler>();

    let mut events_loop = EventsLoop::new();

    let swapchain_flag = Arc::new(AtomicBool::new(false));

    let graphics_system = graphics::GraphicsSystem::new(&events_loop, swapchain_flag.clone())
        .map_err(|e| error!("Failed to initalize graphics subsystem: {}", e))
        .expect("Failed to create graphics subsystem");

    let (tx, ty) = channel();

    let event_system = events::EventHandlerSystem::new(ty);

    let mut dispatcher = DispatcherBuilder::new()
        .with(graphics_system, "graphics", &[])
        .with(CombatSystem, "combat", &[])
        .with(map::MapSystem, "map", &[])
        .with(event_system, "events", &[])
        .build();

    let person = world.create_entity().with(map::Presence).build();

    let empty_space = world.create_entity().with(map::Space::new(true)).build();

    world
        .create_entity()
        .with(map::Space::new_with_contents(true, person))
        .with(map::Move::to(empty_space))
        .with(
            graphics::Graphics::load_with_scale("image.png", ImageFormat::PNG, 300.0, 0.0, 1.0)
                .unwrap(),
        )
        .build();

    world
        .create_entity()
        .with(
            graphics::Graphics::load(
                "image2.png",
                ImageFormat::PNG,
                -200.0,
                -200.0,
            )
            .unwrap(),
        )
        .build();

    dispatcher.setup(&mut world.res);

    let mut done = false;

    loop {
        dispatcher.dispatch(&mut world.res);

        events_loop.poll_events(|ev| {
            match ev {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => done = true,
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => swapchain_flag.store(true, Ordering::Relaxed),
                _ => (),
            };
            match tx.send(ev) {
                Ok(()) => (),
                Err(e) => error!("Failed to send event {}", e)
            };
        });
        if done {
            return;
        }
    }
}
