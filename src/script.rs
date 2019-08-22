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

use winit::{Event, KeyboardInput, VirtualKeyCode, WindowEvent};

use specs::prelude::*;

use super::core::{
    Clank, ClankGetter, ClankScriptGetter, ClankSetter, EngineHandle, GameObjectComponent,
    MethodAdder, Scriptable,
};

type UpdateScript = dyn for<'a> Fn(&'a mut ScriptSystem, &LazyUpdate, Entity) + Send + Sync;

// type HandlerScript = for<'a> Fn(&'a mut ScriptSystem, &LazyUpdate, Entity) + Send + Sync;

pub struct Script {
    update: Arc<UpdateScript>,
    handlers: HashMap<
        (winit::ElementState, VirtualKeyCode),
        Arc<dyn Fn(EngineHandle, Clank, KeyboardInput) + Send + Sync>,
    >,
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
            update: Arc::new(|_x, _y, _z| {}),
            handlers: HashMap::new(),
        }
    }
}

pub struct ScriptBuilder {
    update: Arc<UpdateScript>,
    handlers: HashMap<
        (winit::ElementState, VirtualKeyCode),
        Arc<dyn Fn(EngineHandle, Clank, KeyboardInput) + Send + Sync>,
    >,
}

impl ScriptBuilder {
    pub fn with_native_update<F: Fn(EngineHandle, Clank) + Send + Sync + 'static>(
        mut self,
        f: F,
    ) -> ScriptBuilder {
        let update = Arc::new(f);
        self.update = Arc::new(move |system, lazy, entity| {
            let update_copy = update.clone();
            let setters = system.setters.clone();
            let getters = system.getters.clone();
            lazy.exec_mut(move |world| {
                let handle = EngineHandle::new(world, setters, getters);
                let clank = handle.get(entity);
                update_copy(handle, clank);
            });
        });
        self
    }

    pub fn with_script_update<P: AsRef<std::path::Path>>(mut self, path: P) -> ScriptBuilder {
        let script = Arc::new(load_file(path).expect("Failed to read file"));
        self.update = Arc::new(move |system, lazy, entity| {
            let script = script.clone();
            let lua_ptr = system.lua.clone();
            let names = system.names.clone();

            lazy.exec_mut(move |world| {
                let lua = lua_ptr.lock().expect("Failed to lock lua");
                lua.context(|context| {
                    let chunk = context.load(script.as_ref());
                    let globals = context.globals();
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
        });
        self
    }

    pub fn with_handler<F: Fn(EngineHandle, Clank, KeyboardInput) + Send + Sync + 'static>(
        mut self,
        input: (winit::ElementState, VirtualKeyCode),
        f: F,
    ) -> ScriptBuilder {
        self.handlers.insert(input, Arc::new(f));
        self
    }

    pub fn build(self) -> Script {
        Script {
            update: self.update,
            handlers: self.handlers,
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
                (script.update)(self, lazy, entity);
            }
        }
    }

    fn run_handlers(&mut self, (entities, scripts, lazy): &<ScriptSystem as System>::SystemData) {
        while let Ok(ev) = self.chan.try_recv() {
            match ev {
                Event::WindowEvent {
                    window_id: _,
                    event:
                        WindowEvent::KeyboardInput {
                            device_id: _,
                            input,
                        },
                } => {
                    for (entity, script_obj) in (entities, scripts).join() {
                        let input = input.clone();
                        let script = script_obj.get();

                        if let Some(keycode) = input.virtual_keycode {
                            script
                                .handlers
                                .get(&(input.state, keycode))
                                .cloned()
                                .map(|script| {
                                    let inserters = self.setters.clone();
                                    let getters = self.getters.clone();
                                    lazy.exec_mut(move |world| {
                                        let handle = EngineHandle::new(world, inserters, getters);
                                        let clank = handle.get(entity);
                                        script(handle, clank, input);
                                    })
                                });
                        }
                    }
                }
                _ => (),
            }
        }
    }
}
