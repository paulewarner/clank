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
#[macro_use]
extern crate lazy_static;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, Duration};
use std::thread::sleep;
use std::sync::mpsc::channel;
use std::sync::Arc;

use image::ImageFormat;

use winit::{Event, EventsLoop, WindowEvent};

use specs::prelude::*;

mod graphics;
mod map;
mod script;

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

const FPS_CAP: u128 = 60;
const SCREEN_TICKS_PER_FRAME: u128 = 1000 / FPS_CAP;

fn main() {
    setup_logger().expect("Failed to setup logging");

    let mut world = World::new();
    world.register::<Combatant>();
    world.register::<map::Presence>();
    world.register::<map::Move>();
    world.register::<map::Space>();
    world.register::<graphics::Graphics>();
    world.register::<script::Script>();

    let mut events_loop = EventsLoop::new();

    let swapchain_flag = Arc::new(AtomicBool::new(false));

    let graphics_system = graphics::GraphicsSystem::new(&events_loop, swapchain_flag.clone())
        .map_err(|e| error!("Failed to initalize graphics subsystem: {}", e))
        .expect("Failed to create graphics subsystem");

    let (tx, ty) = channel();

    let event_system = script::ScriptSystem::new(ty);

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
            graphics::Graphics::load_with_scale("image.png", ImageFormat::PNG, 100.0, 0.0, 1.0)
                .unwrap(),
        )
        .with(script::Script::new().with_update(|world, ent| {
            let mut storage = world.write_storage::<graphics::Graphics>();
            let image = storage.get_mut(ent).unwrap();
            let (x, y) = image.position();
            if x <= -300.0 {
                image.set_position((x + 5.0, y));
            } else {
                image.set_position((x - 5.0, y));
            }
        }))
        .build();

    world
        .create_entity()
        .with(graphics::Graphics::load("image2.png", ImageFormat::PNG, -200.0, -200.0).unwrap())
        .build();

    dispatcher.setup(&mut world.res);

    let mut done = false;

    let mut average_fps;
    let mut counted_frames = 0.0;

    let frame_start = Instant::now();

    loop {
        let last_frame = Instant::now();

        dispatcher.dispatch(&mut world.res);

        world.maintain();

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
                Err(e) => error!("Failed to send event {}", e),
            };
        });
        if done {
            return;
        }

        counted_frames += 1.0;
        average_fps = counted_frames / frame_start.elapsed().as_secs() as f64;
        trace!("current fps {}", average_fps);
        if last_frame.elapsed().as_millis() < SCREEN_TICKS_PER_FRAME {
            sleep(Duration::from_millis((SCREEN_TICKS_PER_FRAME - last_frame.elapsed().as_millis()) as u64));
        }
    }
}
