use std::collections::HashMap;
use std::sync::mpsc::Receiver;

use specs::prelude::*;
use winit::{Event, KeyboardInput, WindowEvent, VirtualKeyCode};

pub struct EventHandler {
    handlers: HashMap<(winit::ElementState, Option<VirtualKeyCode>), Box<Fn(Entity, KeyboardInput) + Send + Sync>>,
}

impl Component for EventHandler {
    type Storage = VecStorage<Self>;
}

impl EventHandler {
    pub fn new() -> EventHandler {
        EventHandler {
            handlers: HashMap::new()
        }
    }

    pub fn new_handler<F: Fn(Entity, KeyboardInput) + Send + Sync + 'static>(key: (winit::ElementState, Option<VirtualKeyCode>), handler: F) -> EventHandler {
        let mut handlers = HashMap::new();
        handlers.insert(key, Box::new(handler) as Box<Fn(Entity, KeyboardInput) + Send + Sync>);
        EventHandler {
            handlers
        }
    }
}

pub struct EventHandlerSystem {
    chan: Receiver<Event>
}

impl EventHandlerSystem {
    pub fn new(chan: Receiver<Event>) -> EventHandlerSystem {
        EventHandlerSystem {
            chan: chan
        }
    }
}

impl<'a> System<'a> for EventHandlerSystem {
    type SystemData = (Entities<'a>, ReadStorage<'a, EventHandler>);

    fn run(&mut self, (entities, handlers): Self::SystemData) {
        while let Ok(ev) = self.chan.try_recv() {
            match ev {
                Event::WindowEvent{window_id: _, event: WindowEvent::KeyboardInput{device_id: _, input}} => {
                    for (entity, event) in (&entities, &handlers).join() {
                        event.handlers.get(&(input.state, input.virtual_keycode))
                            .map(|x| x.as_ref())
                            .unwrap_or(&|_x, _y| {})(entity, input);
                    }
                },
                _ => ()
            }
        }
    }
}