fn main() {
    clank::assemble()
        .expect("Failed to initialize clank")
        .run(|mut world| {
            let anim = clank::anim::Animation::new()
                .add_frame(std::time::Duration::from_secs(1), clank::graphics::Graphics::new().load_image("image.png", image::PNG).unwrap().build())
                .build();
            let first = clank::new()
                .with_component(anim)
                .with_component(clank::position::Position::new(0.0, 0.0));
            world.add(first);
        });
}
