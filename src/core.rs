use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, Instant};

use winit::{Event, EventsLoop, WindowEvent};

use specs::prelude::*;

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

pub struct GameObjectComponent<T: Send + Sync + 'static> {
    component: Arc<Mutex<T>>,
}

impl<T: Send + Sync + 'static> Component for GameObjectComponent<T> {
    type Storage = VecStorage<Self>;
}

impl<T: Send + Sync + 'static> GameObjectComponent<T> {
    pub fn new(component: Mutex<T>) -> GameObjectComponent<T> {
        GameObjectComponent {
            component: Arc::new(component),
        }
    }

    pub fn get(&self) -> Arc<Mutex<T>> {
        self.component.clone()
    }
}

pub struct Clank {
    pub components: HashMap<TypeId, Arc<Any + Send + Sync + 'static>>,
    entity: Option<Entity>,
}

impl PartialEq for Clank {
    fn eq(&self, other: &Clank) -> bool {
        self.entity == other.entity
    }
}

impl Eq for Clank {}

impl Clank {
    pub fn new() -> Clank {
        Clank {
            components: HashMap::new(),
            entity: None,
        }
    }

    pub fn with_component<T: Any + Send + Sync>(mut self, component: T) -> Clank {
        self.components
            .insert(TypeId::of::<T>(), Arc::new(Mutex::new(component)));
        self
    }

    pub fn get<'a, T: Send + Sync + 'static>(&'a mut self) -> Option<&'a Mutex<T>> {
        self.components
            .get(&TypeId::of::<T>())
            .and_then(|x| x.downcast_ref::<Mutex<T>>())
            .map(|x| x.borrow())
    }
}

const FPS_CAP: u128 = 60;
const SCREEN_TICKS_PER_FRAME: u128 = 1000 / FPS_CAP;

pub type ClankSetter =
    for<'g> Fn(EntityBuilder<'g>, Arc<Any + Send + Sync + 'static>) -> EntityBuilder<'g>
        + Send
        + Sync;
pub type ClankGetter = Fn(Clank, &World, Entity) -> Clank + Send + Sync;

pub struct ClankEngine<'a, 'b> {
    world: World,
    dispatcher: DispatcherBuilder<'a, 'b>,
    swapchain_flag: Arc<AtomicBool>,
    events_loop: EventsLoop,
    setters: HashMap<TypeId, Arc<ClankSetter>>,
    getters: HashMap<TypeId, Arc<ClankGetter>>,
}

impl<'a, 'b> ClankEngine<'a, 'b> {
    pub fn new(
        world: World,
        dispatcher: DispatcherBuilder<'a, 'b>,
        swapchain_flag: Arc<AtomicBool>,
        events_loop: EventsLoop,
    ) -> ClankEngine<'a, 'b> {
        ClankEngine {
            world,
            dispatcher,
            swapchain_flag,
            events_loop,
            setters: HashMap::new(),
            getters: HashMap::new(),
        }
    }

    pub fn run<F: for<'g> FnOnce(EngineHandle<'g>)>(mut self, init: F) {
        setup_logger().expect("Failed to setup logging");

        let (event_chan, receive_chan) = channel();

        let event_system = super::script::ScriptSystem::new(
            receive_chan,
            self.setters.clone(),
            self.getters.clone(),
        );

        let mut dispatcher = self.dispatcher.with(event_system, "events", &[]).build();

        dispatcher.setup(&mut self.world.res);

        let mut done = false;

        let mut average_fps;
        let mut counted_frames = 0.0;

        let frame_start = Instant::now();
        let swapchain_flag = self.swapchain_flag.clone();

        let handle = EngineHandle::new(&mut self.world, self.setters.clone(), self.getters.clone());

        init(handle);

        loop {
            let last_frame = Instant::now();

            dispatcher.dispatch(&mut self.world.res);

            self.world.maintain();

            &mut self.events_loop.poll_events(|ev| {
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
                match event_chan.send(ev) {
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
                sleep(Duration::from_millis(
                    (SCREEN_TICKS_PER_FRAME - last_frame.elapsed().as_millis()) as u64,
                ));
            }
        }
    }

    pub fn register<T: Send + Sync + 'static>(&mut self) {
        self.world.register::<GameObjectComponent<T>>();
        self.setters.insert(
            TypeId::of::<T>(),
            Arc::new(|builder, component| {
                let typed_component = match component.downcast::<Mutex<T>>() {
                    Ok(a) => a,
                    Err(e) => panic!("Failed downcast {:?}", e),
                };

                match Arc::try_unwrap(typed_component) {
                    Ok(obj) => builder.with(GameObjectComponent::new(obj)),
                    Err(_e) => panic!("Failed to unwrap component"),
                }
            }),
        );

        self.getters.insert(
            TypeId::of::<T>(),
            Arc::new(|mut clank, world, ent| {
                let storage = world.read_storage::<GameObjectComponent<T>>();
                let component = storage.get(ent);
                if let Some(comp) = component.map(|x| x.component.clone() as Arc<Mutex<T>>) {
                    clank.components.insert(TypeId::of::<T>(), comp);
                }
                clank
            }),
        );
    }

    pub fn register_system<T: for<'d> System<'d> + Send + 'static>(
        mut self,
        system: T,
        name: &str,
        deps: &[&str],
    ) -> Self {
        self.dispatcher = self.dispatcher.with(system, name, deps);
        self
    }
}

pub struct EngineHandle<'a> {
    world: &'a mut World,
    inserters: HashMap<TypeId, Arc<ClankSetter>>,
    getters: HashMap<TypeId, Arc<ClankGetter>>,
}

impl<'a> EngineHandle<'a> {
    pub fn new(
        world: &'a mut World,
        inserters: HashMap<TypeId, Arc<ClankSetter>>,
        getters: HashMap<TypeId, Arc<ClankGetter>>,
    ) -> EngineHandle<'a> {
        EngineHandle {
            world,
            inserters,
            getters,
        }
    }

    pub fn add(&mut self, mut c: Clank) {
        let mut ent = self.world.create_entity();

        for (ty, component) in c.components.drain() {
            ent = self.inserters.get(&ty).unwrap()(ent, component);
        }
        ent.build();
    }

    pub fn get(&self, entity: Entity) -> Clank {
        let mut clank = Clank::new();
        for getter in self.getters.values() {
            clank = getter(clank, self.world, entity);
        }
        clank
    }
}
