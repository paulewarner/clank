use std::sync::Mutex;
use std::time::{Duration, Instant};

use rlua::prelude::*;
use specs::prelude::*;

use super::core::GameObjectComponent;
use super::core::MethodAdder;
use super::core::Scriptable;
use super::graphics::Graphics;

pub struct AnimationBuilder {
    frames: Vec<Frame>,
}

impl AnimationBuilder {
    pub fn add_frame(mut self, duration: Duration, image: Graphics) -> Self {
        self.frames.push(Frame {
            duration,
            image: GameObjectComponent::new(Mutex::new(image)),
        });
        self
    }

    pub fn build(self) -> Animation {
        Animation {
            frames: self.frames,
            start_time: None,
        }
    }
}

pub struct Frame {
    duration: Duration,
    image: GameObjectComponent<Graphics>,
}

pub struct Animation {
    frames: Vec<Frame>,
    start_time: Option<Instant>,
}

impl Animation {
    fn choose_frame(&mut self, now: Instant) -> Option<GameObjectComponent<Graphics>> {
        let total_duration = self.total_duration();
        let elapsed_time = now
            .duration_since(self.start_time.unwrap_or(now))
            .as_nanos();
        let mut period = match total_duration.as_nanos() {
            0 => 0,
            any => elapsed_time % any,
        };
        self.start_time = Some(self.start_time.unwrap_or(now));
        for frame in &self.frames {
            let frame_period = frame.duration.as_nanos();
            if period < frame_period {
                return Some(frame.image.clone());
            }
            period -= frame_period;
        }
        None
    }

    pub fn new() -> AnimationBuilder {
        AnimationBuilder { frames: Vec::new() }
    }

    pub fn total_duration(&self) -> Duration {
        self.frames.iter().map(|x| x.duration).sum()
    }
}

impl Scriptable for Animation {
    fn add_methods<'a, 'lua, M: LuaUserDataMethods<'lua, GameObjectComponent<Self>>>(
        _methods: &'a mut MethodAdder<'a, 'lua, Self, M>,
    ) {

    }

    fn name() -> &'static str {
        "anim"
    }
}

pub struct AnimationSystem;

impl<'a> specs::System<'a> for AnimationSystem {
    type SystemData = (
        WriteStorage<'a, GameObjectComponent<Animation>>,
        Entities<'a>,
        Read<'a, LazyUpdate>,
    );

    fn run(&mut self, (mut anims, entities, lazy_update): Self::SystemData) {
        let now = Instant::now();
        for (anim_obj, entity) in (&mut anims, &entities).join() {
            match anim_obj.get().lock() {
                Ok(mut animation) => match animation.choose_frame(now) {
                    Some(frame) => lazy_update.insert(entity, frame),
                    None => lazy_update.remove::<GameObjectComponent<Graphics>>(entity),
                },
                Err(e) => error!("Failed to lock mutex: {}", e),
            }
        }
    }
}
