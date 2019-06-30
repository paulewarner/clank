use std::time::{Instant, Duration};

use specs::prelude::*;
use rlua::prelude::*;

use super::graphics::Graphics;
use super::core::GameObjectComponent;
use super::core::Scriptable;
use super::core::MethodAdder;

struct Frame {
    time: Duration,
    image: GameObjectComponent<Graphics>,
}

struct Animation {
    frames: Vec<Frame>,
    start_time: Option<Instant>
}

impl Animation {
    fn choose_frame(&mut self, now: Instant) -> Option<GameObjectComponent<Graphics>> {
        let total_duration = self.total_duration();
        let elapsed_time = now.duration_since(self.start_time.unwrap_or(Instant::now())).as_nanos();
        let mut period = elapsed_time % total_duration.as_nanos();
        self.start_time = Some(self.start_time.unwrap_or(Instant::now()));
        for frame in &self.frames {
            let frame_period = frame.time.as_nanos();
            if frame_period < period {
                return Some(frame.image.clone());
            }
            period -= frame_period;
        }
        None
    }

    fn total_duration(&self) -> Duration {
        self.frames.iter()
            .map(|x| x.time)
            .sum()
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

struct AnimationSystem;

impl<'a> specs::System<'a> for AnimationSystem {
    type SystemData = (WriteStorage<'a, GameObjectComponent<Animation>>, Entities<'a>, Read<'a, LazyUpdate>);

    fn run(&mut self, (mut anims, entities, lazy_update): Self::SystemData) {
        let now = Instant::now();
        for (anim_obj, entity) in (&mut anims, &entities).join() {
            match anim_obj.get().lock() {
                Ok(mut animation) => {
                    match animation.choose_frame(now) {
                        Some(frame) => lazy_update.insert(entity, frame),
                        None => lazy_update.remove::<GameObjectComponent<Graphics>>(entity)
                    }
                },
                Err(e) => error!("Failed to lock mutex: {}", e)
            }
        }
    }
}