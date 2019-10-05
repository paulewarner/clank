fn main() {
    clank::assemble()
        .expect("Failed to initialize clank")
        .run(|mut handle| {
            let clank = clank::new()
                .with_component(
                    clank::graphics::Graphics::new()
                        .load_image("image.png", image::ImageFormat::PNG).unwrap()
                        .texture_position(0.0, 25.0)
                        .texture_size(150.0, 150.0)
                        .build(),
                )
                .with_component(clank::position::Position::new(0.0, 0.0))
                .with_component(
                    clank::script::Script::new()
                        .with_script_update("scripts/test.lua")
                        .build(),
                );
            handle.add(clank);
        });
}
