use std::sync::Mutex;
use std::time::{Duration, Instant};

use specs::prelude::*;

use crate::core::{GameObjectComponent, MethodAdder, Scriptable};
use crate::graphics::Graphics;

pub struct AnimationBuilder {
    frames: Vec<Frame>,
    id: Option<usize>,
}

impl AnimationBuilder {
    pub fn add_frame(mut self, duration: Duration, image: Graphics) -> Self {
        self.frames.push(Frame {
            duration,
            image: GameObjectComponent::new(Mutex::new(image)),
        });
        self
    }

    pub fn id(mut self, id: usize) -> Self {
        self.id = Some(id);
        self
    }

    pub fn build(self) -> Animation {
        Animation {
            frames: self.frames,
            start_time: None,
            id: self.id,
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
    id: Option<usize>,
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
        AnimationBuilder {
            frames: Vec::new(),
            id: None,
        }
    }

    pub fn total_duration(&self) -> Duration {
        self.frames.iter().map(|x| x.duration).sum()
    }

    pub fn id(&self) -> Option<usize> {
        self.id
    }
}

impl Scriptable for Animation {
    fn add_methods<'lua, M: MethodAdder<'lua, Self>>(_methods: &mut M) {}

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
            let mut animation = anim_obj.get();
            match animation.choose_frame(now) {
                Some(frame) => lazy_update.insert(entity, frame),
                None => lazy_update.remove::<GameObjectComponent<Graphics>>(entity),
            }
        }
    }
}
