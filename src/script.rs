use std::any::TypeId;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use rlua::prelude::*;

use winit::event::{Event as WEvent, VirtualKeyCode, WindowEvent};

use specs::prelude::*;

use crate::core::{
    Clank, ClankGetter, ClankScriptGetter, ClankSetter, EngineHandle, GameObjectComponent,
    MethodAdder, Scriptable,
};

use crate::state::{ScriptFields, ScriptState};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ScriptRef {
    File {
        #[serde(rename = "src")]
        src: String,
    },
    Internal {
        #[serde(rename = "$value")]
        body: String,
    },
}

impl ScriptRef {
    fn create_update(&self) -> std::io::Result<Arc<UpdateScript>> {
        Ok(wrap_update_script(self.read_script()?))
    }

    fn create_handler(&self) -> std::io::Result<Arc<HandlerScript>> {
        Ok(wrap_handler_script(self.read_script()?))
    }

    fn read_script(&self) -> std::io::Result<Vec<u8>> {
        match self {
            ScriptRef::File { src } => {
                let mut reader = std::io::BufReader::new(std::fs::File::open(src)?);
                let mut v = Vec::new();
                reader.read_to_end(&mut v)?;
                Ok(v)
            }
            ScriptRef::Internal { body } => Ok(body.clone().into_bytes()),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ScriptedObject {
    version: usize,

    fields: ScriptFields,

    #[serde(rename = "update")]
    update: ScriptRef,

    handlers: std::collections::HashMap<EventType, ScriptRef>,
}

#[derive(PartialEq, Eq, Hash, Debug, Deserialize)]
pub enum EventType {
    ButtonPressed,
    ButtonReleased,
}

type UpdateScript =
    dyn for<'a> Fn(&'a mut ScriptSystem, &LazyUpdate, Entity, ScriptState) + Send + Sync;

fn wrap_update_script(script: Vec<u8>) -> Arc<UpdateScript> {
    let wrapped_script = Arc::new(script);
    Arc::new(move |system, lazy, entity, state| {
        let script = wrapped_script.clone();
        let lua_ptr = system.lua.clone();
        let names = system.names.clone();

        lazy.exec_mut(move |world| {
            let lua = lua_ptr.lock().expect("Failed to lock lua");
            lua.context(|context| {
                let chunk = context.load(script.as_ref());
                let globals = context.globals();
                globals
                    .set(
                        "self",
                        state
                            .to_table(&context)
                            .expect("Failed to convert to table"),
                    )
                    .expect("Failed to set value: `self`");
                for (name, getter) in names {
                    if let Some(component) = getter(world, entity, context) {
                        globals
                            .set(name, component)
                            .expect(&format!("Failed to set value: {}", name));
                    }
                }
                chunk.exec().expect("Failed to execute chunk");
            });
        });
    })
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum Event {
    ButtonPressed(String),
    ButtonReleased(String),
}

fn map_keycode(code: VirtualKeyCode) -> Result<&'static str, &'static str> {
    match code {
        VirtualKeyCode::A => Ok("a"),
        VirtualKeyCode::B => Ok("b"),
        VirtualKeyCode::C => Ok("c"),
        VirtualKeyCode::D => Ok("d"),
        VirtualKeyCode::E => Ok("e"),
        VirtualKeyCode::F => Ok("f"),
        VirtualKeyCode::G => Ok("g"),
        VirtualKeyCode::H => Ok("h"),
        VirtualKeyCode::J => Ok("j"),
        VirtualKeyCode::I => Ok("i"),
        VirtualKeyCode::K => Ok("k"),
        VirtualKeyCode::L => Ok("l"),
        VirtualKeyCode::M => Ok("m"),
        VirtualKeyCode::N => Ok("n"),
        VirtualKeyCode::O => Ok("o"),
        VirtualKeyCode::P => Ok("p"),
        VirtualKeyCode::Q => Ok("q"),
        VirtualKeyCode::R => Ok("r"),
        VirtualKeyCode::S => Ok("s"),
        VirtualKeyCode::T => Ok("t"),
        VirtualKeyCode::U => Ok("u"),
        VirtualKeyCode::V => Ok("v"),
        VirtualKeyCode::W => Ok("w"),
        VirtualKeyCode::X => Ok("x"),
        VirtualKeyCode::Y => Ok("y"),
        VirtualKeyCode::Z => Ok("z"),
        VirtualKeyCode::Down => Ok("down"),
        VirtualKeyCode::Up => Ok("up"),
        VirtualKeyCode::Left => Ok("left"),
        VirtualKeyCode::Right => Ok("right"),
        VirtualKeyCode::Key0 => Ok("0"),
        VirtualKeyCode::Key1 => Ok("1"),
        VirtualKeyCode::Key2 => Ok("2"),
        VirtualKeyCode::Key3 => Ok("3"),
        VirtualKeyCode::Key4 => Ok("4"),
        VirtualKeyCode::Key5 => Ok("5"),
        VirtualKeyCode::Key6 => Ok("6"),
        VirtualKeyCode::Key7 => Ok("7"),
        VirtualKeyCode::Key8 => Ok("8"),
        VirtualKeyCode::Key9 => Ok("9"),
        VirtualKeyCode::Space => Ok("space"),
        VirtualKeyCode::Tab => Ok("tab"),
        VirtualKeyCode::Capital => Ok("shift"),
        _ => Err("Cannot convert!"),
    }
}

impl<'a, T> std::convert::TryFrom<WEvent<'a, T>> for Event {
    type Error = String;

    fn try_from(ev: WEvent<'a, T>) -> Result<Event, String> {
        match ev {
            WEvent::WindowEvent {
                window_id: _,
                event:
                    WindowEvent::KeyboardInput {
                        device_id: _,
                        input,
                        is_synthetic: _,
                    },
            } => {
                let keycode = input.virtual_keycode.ok_or("No keycode present")?;
                match input.state {
                    winit::event::ElementState::Pressed => {
                        Ok(Event::ButtonPressed(map_keycode(keycode)?.to_owned()))
                    }
                    winit::event::ElementState::Released => {
                        Ok(Event::ButtonReleased(map_keycode(keycode)?.to_owned()))
                    }
                }
            }
            _ => Err("unsupported event type".to_owned()),
        }
    }
}

impl std::convert::From<Event> for EventType {
    fn from(ev: Event) -> EventType {
        match ev {
            Event::ButtonPressed(_) => EventType::ButtonPressed,
            Event::ButtonReleased(_) => EventType::ButtonReleased,
        }
    }
}

impl<'lua> ToLua<'lua> for Event {
    fn to_lua(self, context: LuaContext<'lua>) -> Result<LuaValue<'lua>, LuaError> {
        let ev = context.create_table()?;
        match self {
            Event::ButtonPressed(s) => {
                ev.set("type", "ButtonPressed")?;
                ev.set("button", s)?;
            }
            Event::ButtonReleased(s) => {
                ev.set("type", "ButtonReleased")?;
                ev.set("button", s)?;
            }
        }
        Ok(LuaValue::Table(ev))
    }
}

type HandlerScript =
    dyn for<'a> Fn(&'a mut ScriptSystem, &LazyUpdate, Entity, ScriptState, Event) + Send + Sync;

fn wrap_handler_script(script: Vec<u8>) -> Arc<HandlerScript> {
    let wrapped_script = Arc::new(script);
    Arc::new(move |system, lazy, entity, state, event| {
        let script = wrapped_script.clone();
        let lua_ptr = system.lua.clone();
        let names = system.names.clone();

        lazy.exec_mut(move |world| {
            let lua = lua_ptr.lock().expect("Failed to lock lua");
            lua.context(|context| {
                let chunk = context.load(script.as_ref());
                let globals = context.globals();
                globals
                    .set(
                        "self",
                        state
                            .to_table(&context)
                            .expect("Failed to convert to table"),
                    )
                    .expect("Failed to set value: `self`");
                globals
                    .set("event", event)
                    .expect("Failed to set value `event`");
                for (name, getter) in names {
                    if let Some(component) = getter(world, entity, context) {
                        globals
                            .set(name, component)
                            .expect(&format!("Failed to set value: {}", name));
                    }
                }
                chunk.exec().expect("Failed to execute chunk");
            });
        });
    })
}

pub struct Script {
    update: Arc<UpdateScript>,
    handlers: HashMap<EventType, Arc<HandlerScript>>,
    state: ScriptState,
}

impl Component for Script {
    type Storage = VecStorage<Self>;
}

impl Script {
    fn should_run(&self) -> bool {
        true
    }

    pub fn new() -> ScriptBuilder {
        ScriptBuilder {
            update: Arc::new(|_x, _y, _z, _s| {}),
            handlers: HashMap::new(),
            state: ScriptFields::new(vec![]),
        }
    }
}

pub struct ScriptBuilder {
    update: Arc<UpdateScript>,
    handlers: HashMap<EventType, Arc<HandlerScript>>,
    state: ScriptFields,
}

impl ScriptBuilder {
    pub fn with_native_update<F: Fn(EngineHandle, Clank, ScriptState) + Send + Sync + 'static>(
        mut self,
        f: F,
    ) -> ScriptBuilder {
        let update = Arc::new(f);
        self.update = Arc::new(move |system, lazy, entity, state| {
            let update_copy = update.clone();
            let setters = system.setters.clone();
            let getters = system.getters.clone();
            lazy.exec_mut(move |world| {
                let handle = EngineHandle::new(world, setters, getters);
                let clank = handle.get(entity);
                update_copy(handle, clank, state);
            });
        });
        self
    }

    pub fn with_script_update<P: AsRef<std::path::Path>>(mut self, path: P) -> ScriptBuilder {
        let script = load_file(path).expect("Failed to read file");
        self.update = wrap_update_script(script);
        self
    }

    pub fn with_script_file<P: AsRef<std::path::Path>>(
        mut self,
        fp: P,
    ) -> Result<ScriptBuilder, Box<dyn std::error::Error>> {
        let scripted_object: ScriptedObject =
            serde_xml_rs::from_reader(std::io::BufReader::new(std::fs::File::open(fp)?))?;
        self.update = scripted_object.update.create_update()?;
        self.handlers = scripted_object
            .handlers
            .into_iter()
            .map(|(k, v)| (k, v.create_handler().expect("Failed to read file")))
            .collect();
        self.state = scripted_object.fields;
        Ok(self)
    }

    pub fn with_handler<F: Fn(EngineHandle, Clank, ScriptState, Event) + Send + Sync + 'static>(
        mut self,
        input: EventType,
        f: F,
    ) -> ScriptBuilder {
        let script = Arc::new(f);
        self.handlers.insert(
            input,
            Arc::new(move |system, lazy, entity, state, event| {
                let script_copy = script.clone();
                let inserters = system.setters.clone();
                let getters = system.getters.clone();
                lazy.exec_mut(move |world| {
                    let handle = EngineHandle::new(world, inserters, getters);
                    let clank = handle.get(entity);
                    script_copy(handle, clank, state, event);
                });
            }),
        );
        self
    }

    pub fn build(self) -> Script {
        Script {
            update: self.update,
            handlers: self.handlers,
            state: ScriptState::new(self.state),
        }
    }
}

fn load_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut vec = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut vec)?;
    Ok(vec)
}

impl Scriptable for Script {
    fn add_methods<'lua, M: MethodAdder<'lua, Self>>(_methods: &mut M) {}

    fn name() -> &'static str {
        "events"
    }
}

pub struct ScriptSystem {
    chan: Receiver<Event>,
    setters: HashMap<TypeId, Arc<ClankSetter>>,
    getters: HashMap<TypeId, Arc<ClankGetter>>,
    names: HashMap<&'static str, Arc<ClankScriptGetter>>,
    lua: Arc<Mutex<Lua>>,
}

impl<'a> System<'a> for ScriptSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, GameObjectComponent<Script>>,
        specs::Read<'a, LazyUpdate>,
    );

    fn run(&mut self, data: Self::SystemData) {
        self.run_updates(&data);
        self.run_handlers(&data);
    }
}

impl ScriptSystem {
    pub fn new(
        chan: Receiver<Event>,
        setters: HashMap<TypeId, Arc<ClankSetter>>,
        getters: HashMap<TypeId, Arc<ClankGetter>>,
        names: HashMap<&'static str, Arc<ClankScriptGetter>>,
    ) -> ScriptSystem {
        ScriptSystem {
            chan,
            setters,
            getters,
            names,
            lua: Arc::new(Mutex::new(Lua::new())),
        }
    }

    fn run_updates(&mut self, (entities, scripts, lazy): &<ScriptSystem as System>::SystemData) {
        for (entity, script_obj) in (entities, scripts).join() {
            let script = script_obj.get();
            if script.should_run() {
                let state = script.state.clone();
                (script.update)(self, lazy, entity, state);
            }
        }
    }

    fn run_handlers(&mut self, (entities, scripts, lazy): &<ScriptSystem as System>::SystemData) {
        while let Ok(winit_event) = self.chan.try_recv() {
            match Event::try_from(winit_event) {
                Ok(event) => {
                    for (entity, script_obj) in (entities, scripts).join() {
                        let script = script_obj.get();
                        script
                            .handlers
                            .get(&EventType::from(event.clone()))
                            .cloned()
                            .map(|s| {
                                let state = script.state.clone();
                                s(self, lazy, entity, state, event.clone());
                            });
                    }
                }
                Err(e) => {
                    debug!("Failed to convert window event for handling processing with the following error: {:?}", e);
                }
            }
        }
    }
}
