use image::{DynamicImage, GenericImageView};
use serde::Deserialize;
use specs::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Mutex;
use std::time::Duration;

use crate::core::{GameObjectComponent, MethodAdder, Scriptable};
use crate::graphics;
use crate::graphics::anim;

pub struct SpriteSystem;

#[derive(Debug)]
struct NoSpriteTypeFound {
    sprite_type: String,
}

impl std::fmt::Display for NoSpriteTypeFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "No sprite type [{}] could be found!", self.sprite_type)
    }
}

impl std::error::Error for NoSpriteTypeFound {}

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
            trace!("Finished!");
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
        format: image::ImageFormat,
        sprite_type: S1,
        default: S2,
        scale: Option<f32>,
    ) -> Result<Sprite, Box<dyn Error>> {
        let sprite_sheet = image::load(BufReader::new(File::open(path)?), format)?;
        self.types
            .get(sprite_type.as_ref())
            .map(|sprite_type_info| {
                sprite_type_info.create_sprite(
                    sprite_sheet,
                    String::from(default.as_ref()),
                    scale.unwrap_or(sprite_type_info.default_scale),
                )
            })
            .ok_or(Box::new(NoSpriteTypeFound {
                sprite_type: String::from(sprite_type.as_ref()),
            }))
    }

    pub fn create_sprite_with_scale<P: AsRef<std::path::Path>, S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        path: P,
        format: image::ImageFormat,
        sprite_type: S1,
        default: S2,
        scale: f32,
    ) -> Result<Sprite, Box<dyn Error>> {
        self.create_sprite_inner(path, format, sprite_type, default, Some(scale))
    }

    pub fn create_sprite<P: AsRef<std::path::Path>, S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        path: P,
        format: image::ImageFormat,
        sprite_type: S1,
        default: S2,
    ) -> Result<Sprite, Box<dyn Error>> {
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
    fn create_sprite(&self, mut sprite_sheet: DynamicImage, default: String, scale: f32) -> Sprite {
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
                            .image(self.get_image_by_index(&mut sprite_sheet, frame.sprite_number))
                            .scale(scale)
                            .build()
                            .unwrap(),
                    );
                }
                (
                    key,
                    GameObjectComponent::new(Mutex::new(builder.id(id).build())),
                )
            })
            .collect();

        Sprite {
            animations,
            current: default,
        }
    }

    fn get_image_by_index(&self, sprite_sheet: &mut DynamicImage, index: u32) -> DynamicImage {
        let (width, _) = sprite_sheet.dimensions();
        let sheet_width = width / self.sprite_width;
        let (x, y) = (
            (index % sheet_width) * self.sprite_width,
            (index / sheet_width) * self.sprite_height,
        );
        sprite_sheet.crop(x, y, self.sprite_width, self.sprite_height)
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
