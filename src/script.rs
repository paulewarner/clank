use std::any::TypeId;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use rlua::prelude::*;

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

impl std::convert::From<crate::windowing::InputEvent> for EventType {
    fn from(ev: crate::windowing::InputEvent) -> EventType {
        match ev {
            crate::windowing::InputEvent::ButtonPressed(_) => EventType::ButtonPressed,
            crate::windowing::InputEvent::ButtonReleased(_) => EventType::ButtonReleased,
        }
    }
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

type HandlerScript = dyn for<'a> Fn(&'a mut ScriptSystem, &LazyUpdate, Entity, ScriptState, crate::windowing::InputEvent)
    + Send
    + Sync;

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

    pub fn with_handler<
        F: Fn(EngineHandle, Clank, ScriptState, crate::windowing::InputEvent) + Send + Sync + 'static,
    >(
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
    chan: Receiver<crate::windowing::InputEvent>,
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
        chan: Receiver<crate::windowing::InputEvent>,
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
        while let Ok(event) = self.chan.try_recv() {
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
    }
}
