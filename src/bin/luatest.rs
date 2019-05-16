fn main() {
    clank::assemble().run(|mut handle| {
        let clank = clank::new().with_component(
            clank::script::Script::new()
                .with_script_update("scripts/test.lua")
                .build(),
        );
        handle.add(clank);
    });
}
