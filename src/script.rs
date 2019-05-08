use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use winit::{Event, KeyboardInput, VirtualKeyCode, WindowEvent};

use specs::prelude::*;

use super::core::{Clank, EngineHandle, GameObjectComponent, ClankGetter, ClankSetter};

pub struct Script {
    update: Arc<Fn(EngineHandle, Clank) + Send + Sync>,
    handlers: HashMap<
        (winit::ElementState, VirtualKeyCode),
        Arc<Fn(EngineHandle, Clank, KeyboardInput) + Send + Sync>,
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
            update: Arc::new(|_x, _y| {}),
            handlers: HashMap::new(),
        }
    }
}

pub struct ScriptBuilder {
    update: Arc<Fn(EngineHandle, Clank) + Send + Sync>,
    handlers: HashMap<
        (winit::ElementState, VirtualKeyCode),
        Arc<Fn(EngineHandle, Clank, KeyboardInput) + Send + Sync>,
    >,
}

impl ScriptBuilder {
    pub fn with_update<F: Fn(EngineHandle, Clank) + Send + Sync + 'static>(
        mut self,
        f: F,
    ) -> ScriptBuilder {
        self.update = Arc::new(f);
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

pub struct ScriptSystem {
    chan: Receiver<Event>,
    setters: HashMap<TypeId, Arc<ClankSetter>>,
    getters: HashMap<TypeId, Arc<ClankGetter>>,
}

impl<'a> System<'a> for ScriptSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, GameObjectComponent<Script>>,
        Read<'a, LazyUpdate>,
    );

    fn run(&mut self, data: Self::SystemData) {
        self.run_updates(&data);
        self.run_handlers(&data);
    }
}

impl ScriptSystem {
    pub fn new(chan: Receiver<Event>, setters: HashMap<TypeId, Arc<ClankSetter>>, getters: HashMap<TypeId, Arc<ClankGetter>>) -> ScriptSystem {
        ScriptSystem {
            chan,
            setters,
            getters,
        }
    }

    fn run_updates(&mut self, (entities, scripts, lazy): &<ScriptSystem as System>::SystemData) {
        for (ent, script) in (entities, scripts).join() {
            let ptr = script.get();
            let script_ptr = ptr.lock().unwrap();
            if script_ptr.should_run() {
                let update = script_ptr.update.clone();
                let setters = self.setters.clone();
                let getters = self.getters.clone();
                lazy.exec_mut(move |world| {
                    let handle = EngineHandle::new(world, setters, getters);
                    let clank = handle.get(ent);
                    update(handle, clank);
                });
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
                    for (entity, script) in (entities, scripts).join() {
                        let input = input.clone();
                        let ptr = script.get();
                        let script_ptr = ptr.lock().unwrap();

                        if let Some(keycode) = input.virtual_keycode {
                            script_ptr
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
