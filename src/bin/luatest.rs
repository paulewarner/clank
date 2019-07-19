fn main() {
    clank::assemble().run(|mut handle| {
        let clank = clank::new()
            .with_component(
                clank::graphics::Graphics::load_with_crop(
                    "image.png",
                    image::ImageFormat::PNG,
                    0,
                    25,
                    150,
                    150,
                )
                .unwrap(),
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
