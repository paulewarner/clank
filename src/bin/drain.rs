use image::ImageFormat;

fn main() {
    clank::assemble().run(|mut world| {
        let first = clank::new()
            .with_component(
                clank::graphics::Graphics::load_with_crop(
                    "image.png",
                    ImageFormat::PNG,
                    0,
                    25,
                    150,
                    150,
                )
                .unwrap(),
            )
            .with_component(clank::position::Position::new(0.0, 0.0));
        world.add(first);
    });
}
