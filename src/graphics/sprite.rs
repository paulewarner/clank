use serde::Deserialize;
use specs::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use crate::core::{GameObjectComponent, MethodAdder, Scriptable};
use crate::graphics;
use crate::graphics::anim;
use crate::graphics::imagewrapper::{Image, ImageFormat};

pub struct SpriteSystem;

#[derive(Debug)]
struct SpriteTypeNotFoundError {
    sprite_type: String,
}

impl std::fmt::Display for SpriteTypeNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "No sprite type [{}] could be found!", self.sprite_type)
    }
}

impl std::error::Error for SpriteTypeNotFoundError {}

impl SpriteSystem {
    pub fn new() -> SpriteSystem {
        SpriteSystem {}
    }
}

impl<'a> System<'a> for SpriteSystem {
    type SystemData = (
        Read<'a, LazyUpdate>,
        Entities<'a>,
        ReadStorage<'a, GameObjectComponent<Sprite>>,
        ReadStorage<'a, GameObjectComponent<anim::Animation>>,
    );

    fn run(&mut self, (lazy, entities, sprites, animations): Self::SystemData) {
        for (entity, sprite_obj) in (&entities, &sprites).join() {
            let animation_id = animations.get(entity).map(|x| x.get().id());
            let sprite = sprite_obj.get();
            match sprite.animations.get(&sprite.current) {
                Some(new_animation) => match animation_id {
                    Some(id) => {
                        if id != new_animation.get().id() {
                            lazy.insert(entity, new_animation.clone());
                        }
                    }
                    None => lazy.insert(entity, new_animation.clone()),
                },
                None => {
                    lazy.remove::<GameObjectComponent<anim::Animation>>(entity);
                    lazy.remove::<GameObjectComponent<graphics::Graphics>>(entity);
                }
            }
        }
    }
}

#[derive(Deserialize)]
pub struct SpriteConfig {
    types: HashMap<String, SpriteTypeInfo>,
}

impl SpriteConfig {
    fn create_sprite_inner<P: AsRef<std::path::Path>, S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        path: P,
        format: ImageFormat,
        sprite_type: S1,
        default: S2,
        scale: Option<f32>,
    ) -> Result<Sprite, Box<dyn std::error::Error>> {
        let sprite_sheet = Image::load_with_ext(path, format)?;
        Ok(self
            .types
            .get(sprite_type.as_ref())
            .map(|sprite_type_info| {
                sprite_type_info.create_sprite(
                    sprite_sheet,
                    String::from(default.as_ref()),
                    scale.unwrap_or(sprite_type_info.default_scale),
                )
            })
            .transpose()?
            .ok_or(Box::new(SpriteTypeNotFoundError {
                sprite_type: String::from(sprite_type.as_ref()),
            }))?)
    }

    pub fn create_sprite_with_scale<P: AsRef<std::path::Path>, S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        path: P,
        format: image::ImageFormat,
        sprite_type: S1,
        default: S2,
        scale: f32,
    ) -> Result<Sprite, Box<dyn std::error::Error>> {
        self.create_sprite_inner(path, format, sprite_type, default, Some(scale))
    }

    pub fn create_sprite<P: AsRef<std::path::Path>, S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        path: P,
        format: image::ImageFormat,
        sprite_type: S1,
        default: S2,
    ) -> Result<Sprite, Box<dyn std::error::Error>> {
        self.create_sprite_inner(path, format, sprite_type, default, None)
    }
}

#[derive(Deserialize)]
struct SpriteTypeInfo {
    sprite_width: u32,
    sprite_height: u32,
    default_scale: f32,
    animation_info: HashMap<String, Vec<AnimationInfo>>,
}

impl SpriteTypeInfo {
    fn create_sprite(
        &self,
        mut sprite_sheet: Image,
        default: String,
        scale: f32,
    ) -> Result<Sprite, Box<dyn std::error::Error>> {
        let animations: HashMap<String, GameObjectComponent<anim::Animation>> = self
            .animation_info
            .clone()
            .drain()
            .enumerate()
            .map(|(id, (key, animation))| {
                let mut builder = anim::Animation::new();
                for frame in animation {
                    builder = builder.add_frame(
                        frame.frame_run,
                        graphics::Graphics::new()
                            .image(sprite_sheet.get_image_by_index(
                                (self.sprite_width, self.sprite_height),
                                frame.sprite_number,
                            ))
                            .scale(scale)
                            .build()?,
                    );
                }
                Ok((
                    key,
                    GameObjectComponent::new(Mutex::new(builder.id(id).build())),
                ))
            })
            .collect::<Result<
                HashMap<String, GameObjectComponent<anim::Animation>>,
                Box<dyn std::error::Error>,
            >>()?;

        Ok(Sprite {
            animations,
            current: default,
        })
    }
}

#[derive(Clone, Deserialize)]
struct AnimationInfo {
    frame_run: Duration,
    sprite_number: u32,
}

pub struct Sprite {
    current: String,
    animations: HashMap<String, GameObjectComponent<anim::Animation>>,
}

impl Scriptable for Sprite {
    fn name() -> &'static str {
        "sprite"
    }

    fn add_methods<'lua, M: MethodAdder<'lua, Self>>(_methods: &mut M) {}
}
