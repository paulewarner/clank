fn main() {
    clank::assemble().run(|mut world| {
        let first = clank::new()
            .with_component(
                clank::anim::Animation::new()
                .add_frame(
                    std::time::Duration::from_secs(1),
                    clank::graphics::Graphics::load_text_with_font_path("One".to_owned(), "Go-Medium.ttf", (0, 0, 0), 128.0)
                    .unwrap())
                .add_frame(
                    std::time::Duration::from_secs(1),
                    clank::graphics::Graphics::load_text_with_font_path("Two".to_owned(), "Go-Medium.ttf", (0, 0, 0), 128.0)
                    .unwrap()
                )
                .build()
            )
            .with_component(clank::position::Position::new(0.0, 0.0))
            .with_component(
                clank::script::Script::new()
                    .with_native_update(|_world, mut clank| {
                        let mut position = clank
                            .get::<clank::position::Position>()
                            .and_then(|x| x.lock().ok())
                            .unwrap();
                        let (x, y) = position.get();
                        if x > -1200.0 {
                            position.set((x - 5.0, y));
                        }
                    })
                    .build(),
            );
        world.add(first);
    });
}
