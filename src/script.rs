use std::sync::Arc;
use std::collections::HashMap;
use std::sync::mpsc::Receiver;

use winit::{Event, KeyboardInput, WindowEvent, VirtualKeyCode};

use specs::prelude::*;

pub struct Script {
    update: Arc<Fn(&mut World, Entity) + Send + Sync>,
    handlers: HashMap<(winit::ElementState, VirtualKeyCode), Arc<Fn(&mut World, Entity, KeyboardInput) + Send + Sync>>
}

impl Component for Script {
    type Storage = VecStorage<Self>;
}

impl Script {
    fn should_run(&self) -> bool {
        true
    }

    pub fn new() -> Script {
        Script {
            update: Arc::new(|_x, _y| {}),
            handlers: HashMap::new()
        }
    }

    pub fn with_update<F: Fn(&mut World, Entity) + Send + Sync + 'static>(mut self, f: F) -> Script {
        self.update = Arc::new(f);
        self
    }

    pub fn with_handler<F: Fn(&mut World, Entity, KeyboardInput) + Send + Sync + 'static>(mut self, input: (winit::ElementState, VirtualKeyCode), f: F) -> Script {
        self.handlers.insert(input, Arc::new(f));
        self
    }
}

pub struct ScriptSystem {
    chan: Receiver<Event>
}

impl<'a> System<'a> for ScriptSystem {
    type SystemData = (Entities<'a>, ReadStorage<'a, Script>, Read<'a, LazyUpdate>);

    fn run(&mut self, data: Self::SystemData) {
        self.run_updates(&data);
        self.run_handlers(&data);
    }
}

impl ScriptSystem {

    pub fn new(chan: Receiver<Event>) -> ScriptSystem {
        ScriptSystem {
            chan
        }
    }

    fn run_updates(&mut self, (entities, scripts, lazy): &<ScriptSystem as System>::SystemData) {
        for (ent, script) in (entities, scripts).join() {
            if script.should_run() {
                let update = script.update.clone();
                lazy.exec_mut(move |world| {
                    update(world, ent);
                });
            }
        }
    }

    fn run_handlers(&mut self, (entities, scripts, lazy): &<ScriptSystem as System>::SystemData) {
        while let Ok(ev) = self.chan.try_recv() {
            match ev {
                Event::WindowEvent{window_id: _, event: WindowEvent::KeyboardInput{device_id: _, input}} => {
                    for (entity, script) in (entities, scripts).join() {
                        let input = input.clone();

                        if let Some(keycode) = input.virtual_keycode {
                            script.handlers.get(&(input.state, keycode))
                                .cloned()
                                .map(|script| lazy.exec_mut(move |world| {
                                    script(world, entity, input);
                                }));
                        }
                    }
                },
                _ => ()
            }
        }
    }
}