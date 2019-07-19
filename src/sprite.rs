use image::{DynamicImage, GenericImageView};
use serde::Deserialize;
use specs::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Mutex;
use std::time::Duration;

use super::anim;
use super::core::{GameObjectComponent, MyMethods, Scriptable};
use super::graphics;

pub struct SpriteSystem {
    config: SpriteConfig,
}

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
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Result<SpriteSystem, Box<Error>> {
        Ok(SpriteSystem {
            config: serde_json::from_reader(BufReader::new(File::open(path)?))?,
        })
    }

    pub fn create_sprite<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        format: image::ImageFormat,
        sprite_type: String,
        default: String,
    ) -> Result<Sprite, Box<Error>> {
        let sprite_sheet = image::load(BufReader::new(File::open(path)?), format)?;
        self.config
            .types
            .get(&sprite_type)
            .map(|sprite_type_info| sprite_type_info.create_sprite(sprite_sheet, default))
            .ok_or(Box::new(NoSpriteTypeFound { sprite_type }))
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
            let sprite = sprite_obj.get();
            match animations.get(entity) {
                Some(animation) => {
                    if Some(animation.get().id())
                        != sprite.animations.get(&sprite.current).map(|x| x.get().id())
                    {
                        lazy.insert(
                            entity,
                            sprite.animations.get(&sprite.current).unwrap().clone(),
                        );
                    }
                }
                None => {
                    lazy.remove::<GameObjectComponent<anim::Animation>>(entity);
                    lazy.remove::<GameObjectComponent<graphics::Graphics>>(entity);
                }
            }
        }
    }
}

#[derive(Deserialize)]
struct SpriteConfig {
    types: HashMap<String, SpriteTypeInfo>,
}

#[derive(Deserialize)]
struct SpriteTypeInfo {
    sprite_width: u32,
    sprite_height: u32,
    animation_info: HashMap<String, Vec<AnimationInfo>>,
}

impl SpriteTypeInfo {
    fn create_sprite(&self, mut sprite_sheet: DynamicImage, default: String) -> Sprite {
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
                        graphics::Graphics::from_image(
                            self.get_image_by_index(&mut sprite_sheet, frame.sprite_number),
                        ),
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
        let (width, height) = sprite_sheet.dimensions();
        let (x, y) = (
            (index % self.sprite_width) * width,
            (index % self.sprite_height) * height,
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

    fn add_methods<'lua, M: MyMethods<'lua, Self>>(_methods: &mut M) {}
}
