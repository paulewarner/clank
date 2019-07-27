fn main() {
    clank::assemble()
        .expect("Failed to initialize clank")
        .run(|mut world| {
            let sprite_builder = world.fetch::<clank::sprite::SpriteConfig>().unwrap();
            let sprite = sprite_builder
                .create_sprite("image3.png", image::PNG, "character", "idle_forward")
                .unwrap();
            let first = clank::new()
                .with_component(sprite)
                .with_component(clank::position::Position::new(0.0, 0.0))
                .with_component(
                    clank::script::Script::new()
                        .with_native_update(|_engine, mut clank| {
                            let mut position = clank.get::<clank::position::Position>().unwrap();
                            let (x, y) = position.get();
                            position.set((x + 1.0, y + 1.0));
                        })
                        .build(),
                );
            world.add(first);
        });
}
