extern crate rand;
extern crate specs;
#[macro_use]
extern crate log;
extern crate chrono;
extern crate fern;

use std::sync::Arc;

use specs::prelude::*;

mod map;

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

    let mut dispatcher = DispatcherBuilder::new()
        .with(CombatSystem, "combat", &[])
        .with(map::MapSystem, "map", &[])
        .build();

    let person = world.create_entity().with(map::Presence).build();

    let empty_space = world.create_entity().with(map::Space::new(true)).build();

    world
        .create_entity()
        .with(map::Space::new_with_contents(true, person))
        .with(map::Move::to(empty_space))
        .build();

    dispatcher.setup(&mut world.res);
    dispatcher.dispatch(&mut world.res);
}
