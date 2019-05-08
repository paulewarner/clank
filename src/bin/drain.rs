fn main() {
    clank::assemble().run(|mut world| {
        let first = clank::new()
            .with_component(
                clank::graphics::Graphics::load_with_scale(
                    "image.png",
                    clank::graphics::ImageFormat::PNG,
                    100.0,
                    0.0,
                    1.0,
                )
                .unwrap(),
            )
            .with_component(
                clank::script::Script::new()
                    .with_update(|_world, mut clank| {
                        let mut image = clank
                            .get::<clank::graphics::Graphics>()
                            .and_then(|x| x.lock().ok())
                            .unwrap();
                        let (x, y) = image.position();
                        if x > -1200.0 {
                            image.set_position((x - 5.0, y));
                        }
                    })
                    .build(),
            );

        world.add(first);
    });
}
