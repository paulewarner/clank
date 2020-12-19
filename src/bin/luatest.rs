fn main() {
    clank::assemble()
        .expect("Failed to initialize clank")
        .run(|mut handle| {
            let clank = clank::new()
                .with_component(
                    clank::graphics::Graphics::new()
                        .load_image("image.png", image::ImageFormat::Png)
                        .build()
                        .unwrap(),
                )
                .with_component(clank::position::Position::new(0.0, 0.0));
            handle.add(clank);
        });
}
