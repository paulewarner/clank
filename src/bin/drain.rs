fn main() {
    clank::assemble()
        .expect("Failed to initialize clank")
        .run(|mut world| {
            let window_builder = world
                .fetch::<clank::graphics::window::WindowBuilderSystem>()
                .expect("Failed to get windowing");

            let first = clank::new()
                .with_component(
                    window_builder
                        .create_window()
                        .text("This text")
                        .build()
                        .expect("failed to build window"),
                )
                .with_component(clank::position::Position::new(0.0, 0.0));
            world.add(first);
        });
}
