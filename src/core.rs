use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, Instant};

use super::script::Event;
use winit::{Event as WEvent, EventsLoop, WindowEvent};

use rlua::prelude::*;
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

pub trait MethodAdder<'lua, T: Scriptable> {
    fn add_method<S: ?Sized, A, R, F>(&mut self, name: &S, method: F)
    where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &T, A) -> LuaResult<R>;

    fn add_method_mut<S: ?Sized, A, R, F>(&mut self, name: &S, method: F)
    where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &mut T, A) -> LuaResult<R>;

    fn add_function<S: ?Sized, A, R, F>(&mut self, name: &S, function: F)
    where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Fn(LuaContext<'lua>, A) -> LuaResult<R>;

    fn add_function_mut<S: ?Sized, A, R, F>(&mut self, name: &S, function: F)
    where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + FnMut(LuaContext<'lua>, A) -> LuaResult<R>;

    fn add_meta_method<A, R, F>(&mut self, meta: LuaMetaMethod, method: F)
    where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &T, A) -> LuaResult<R>;

    fn add_meta_method_mut<A, R, F>(&mut self, meta: LuaMetaMethod, method: F)
    where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &mut T, A) -> LuaResult<R>;

    fn add_meta_function<A, R, F>(&mut self, meta: LuaMetaMethod, function: F)
    where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Fn(LuaContext<'lua>, A) -> LuaResult<R>;

    fn add_meta_function_mut<A, R, F>(&mut self, meta: LuaMetaMethod, function: F)
    where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + FnMut(LuaContext<'lua>, A) -> LuaResult<R>;
}

pub trait Scriptable: Sized + Send + Sync + 'static {
    fn add_methods<'lua, M: MethodAdder<'lua, Self>>(methods: &mut M);

    fn name() -> &'static str;
}

pub struct _MethodAdder<'a, 'lua, F: 'lua, T: Scriptable, M>
where
    M: LuaUserDataMethods<'lua, GameObjectComponent<T>>,
{
    methods: &'a mut M,
    phantom: std::marker::PhantomData<T>,
    phanto: std::marker::PhantomData<&'lua F>,
}

pub type MethodAdderImpl<'a, 'lua, T, M> = _MethodAdder<'a, 'lua, (), T, M>;

impl<'a, 'lua, T: Scriptable, M: LuaUserDataMethods<'lua, GameObjectComponent<T>>>
    MethodAdder<'lua, T> for MethodAdderImpl<'a, 'lua, T, M>
{
    fn add_method<S: ?Sized, A, R, F>(&mut self, name: &S, method: F)
    where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &T, A) -> LuaResult<R>,
    {
        self.methods.add_method(name, move |context, wrapper, arg| {
            let item = wrapper.component.lock().expect("Failed to lock mutex");
            method(context, &item, arg)
        })
    }

    fn add_method_mut<S: ?Sized, A, R, F>(&mut self, name: &S, method: F)
    where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &mut T, A) -> LuaResult<R>,
    {
        self.methods
            .add_method_mut(name, move |context, wrapper, arg| {
                let mut item = wrapper.component.lock().expect("Failed to lock mutex");
                method(context, &mut item, arg)
            })
    }

    fn add_function<S: ?Sized, A, R, F>(&mut self, name: &S, function: F)
    where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Fn(LuaContext<'lua>, A) -> LuaResult<R>,
    {
        self.methods.add_function(name, function)
    }

    fn add_function_mut<S: ?Sized, A, R, F>(&mut self, name: &S, function: F)
    where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + FnMut(LuaContext<'lua>, A) -> LuaResult<R>,
    {
        self.methods.add_function_mut(name, function)
    }

    fn add_meta_method<A, R, F>(&mut self, meta: LuaMetaMethod, method: F)
    where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &T, A) -> LuaResult<R>,
    {
        self.methods
            .add_meta_method(meta, move |context, wrapper, arg| {
                let item = wrapper.component.lock().expect("Failed to lock mutex");
                method(context, &item, arg)
            })
    }

    fn add_meta_method_mut<A, R, F>(&mut self, meta: LuaMetaMethod, method: F)
    where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &mut T, A) -> LuaResult<R>,
    {
        self.methods
            .add_meta_method_mut(meta, move |context, wrapper, arg| {
                let mut item = wrapper.component.lock().expect("Failed to lock mutex");
                method(context, &mut item, arg)
            })
    }

    fn add_meta_function<A, R, F>(&mut self, meta: LuaMetaMethod, function: F)
    where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Fn(LuaContext<'lua>, A) -> LuaResult<R>,
    {
        self.methods.add_meta_function(meta, function)
    }

    fn add_meta_function_mut<A, R, F>(&mut self, meta: LuaMetaMethod, function: F)
    where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + FnMut(LuaContext<'lua>, A) -> LuaResult<R>,
    {
        self.methods.add_meta_function_mut(meta, function)
    }
}

pub struct GameObjectComponent<T: Scriptable + Send + Sync + 'static> {
    component: Arc<Mutex<T>>,
}

// I have no idea why a manual implementation of this is required - LuaContext.create_userdata complains otherwise.
impl<T: Scriptable + Send + Sync> Clone for GameObjectComponent<T> {
    fn clone(&self) -> Self {
        GameObjectComponent {
            component: self.component.clone(),
        }
    }
}

impl<T: Scriptable + Send + Sync> LuaUserData for GameObjectComponent<T> {
    fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(methods: &mut M) {
        let mut method_adder = MethodAdderImpl {
            methods,
            phantom: std::marker::PhantomData,
            phanto: std::marker::PhantomData,
        };

        T::add_methods(&mut method_adder)
    }
}

impl<T: Scriptable + Send + Sync + 'static> Component for GameObjectComponent<T> {
    type Storage = VecStorage<Self>;
}

impl<T: Scriptable + Send + Sync + 'static> GameObjectComponent<T> {
    pub fn new(component: Mutex<T>) -> GameObjectComponent<T> {
        GameObjectComponent {
            component: Arc::new(component),
        }
    }

    pub fn get(&self) -> std::sync::MutexGuard<'_, T> {
        let s = self.component.lock().expect("Failed to lock mutex");
        s
    }
}

pub struct Clank {
    pub components: HashMap<TypeId, Arc<dyn Any + Send + Sync + 'static>>,
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

    pub fn with_component<T: Any + Scriptable + Send + Sync>(mut self, component: T) -> Clank {
        self.components
            .insert(TypeId::of::<T>(), Arc::new(Mutex::new(component)));
        self
    }

    pub fn get<'a, T: Send + Sync + 'static>(&'a mut self) -> Option<std::sync::MutexGuard<'a, T>> {
        self.components
            .get(&TypeId::of::<T>())
            .and_then(|x| x.downcast_ref::<Mutex<T>>())
            .map(|x| x.lock().expect("Failed to lock mutex"))
    }
}

const FPS_CAP: u128 = 60;
const SCREEN_TICKS_PER_FRAME: u128 = 1000 / FPS_CAP;

pub type ClankSetter = dyn for<'g> Fn(EntityBuilder<'g>, Arc<dyn Any + Send + Sync + 'static>) -> EntityBuilder<'g>
    + Send
    + Sync;
pub type ClankGetter = dyn Fn(Clank, &World, Entity) -> Clank + Send + Sync;

pub type ClankScriptGetter =
    dyn for<'lua> Fn(&World, Entity, LuaContext<'lua>) -> Option<LuaValue<'lua>> + Send + Sync;

pub struct ClankEngine<'a, 'b> {
    world: World,
    dispatcher: DispatcherBuilder<'a, 'b>,
    swapchain_flag: Arc<AtomicBool>,
    events_loop: EventsLoop,
    setters: HashMap<TypeId, Arc<ClankSetter>>,
    getters: HashMap<TypeId, Arc<ClankGetter>>,
    names: HashMap<&'static str, Arc<ClankScriptGetter>>,
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
            names: HashMap::new(),
        }
    }

    pub fn run<F: for<'g> FnOnce(EngineHandle<'g>)>(mut self, init: F) {
        setup_logger().expect("Failed to setup logging");

        let (event_chan, receive_chan) = channel();

        let event_system = super::script::ScriptSystem::new(
            receive_chan,
            self.setters.clone(),
            self.getters.clone(),
            self.names.clone(),
        );

        let mut dispatcher = self.dispatcher.with(event_system, "events", &[]).build();

        dispatcher.setup(&mut self.world);

        let mut done = false;

        let swapchain_flag = self.swapchain_flag.clone();

        let handle = EngineHandle::new(&mut self.world, self.setters.clone(), self.getters.clone());

        init(handle);

        loop {
            let last_frame = Instant::now();

            dispatcher.dispatch(&mut self.world);

            self.world.maintain();

            &mut self.events_loop.poll_events(|ev| {
                match ev {
                    WEvent::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => done = true,
                    WEvent::WindowEvent {
                        event: WindowEvent::Resized(_),
                        ..
                    } => swapchain_flag.store(true, Ordering::Relaxed),
                    _ => (),
                };
                match Event::try_from(ev).and_then(|x| {
                    event_chan
                        .send(x)
                        .map_err(|y| format!("Failed to send chan, {}", y))
                }) {
                    Ok(_e) => (),
                    Err(e) => error!("Failed to send event {}", e),
                };
            });
            if done {
                return;
            }

            if last_frame.elapsed().as_millis() < SCREEN_TICKS_PER_FRAME {
                sleep(Duration::from_millis(
                    (SCREEN_TICKS_PER_FRAME - last_frame.elapsed().as_millis()) as u64,
                ));
            }
        }
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, resource: T) {
        self.world.insert(resource);
    }

    pub fn register<T: Scriptable + Send + Sync + 'static>(&mut self) {
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

        self.names.insert(
            T::name(),
            Arc::new(|world, entity, context| {
                let storage = world.read_storage::<GameObjectComponent<T>>();

                storage
                    .get(entity)
                    .cloned()
                    .and_then(|x| context.create_userdata(x).ok())
                    .map(|x| LuaValue::UserData(x))
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
            if let Some(inserter) = self.inserters.get(&ty) {
                ent = inserter(ent, component);
            }
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

    pub fn fetch<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.world.get_mut()
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, resource: T) {
        self.world.insert(resource);
    }
}
