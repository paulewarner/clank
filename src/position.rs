use rlua::prelude::*;
use super::core::{GameObjectComponent, Scriptable, MethodAdder};

pub struct Position(f64, f64);

impl Scriptable for Position {

    fn add_methods<'a, 'lua, M: LuaUserDataMethods<'lua, GameObjectComponent<Self>>>(
        methods: &'a mut MethodAdder<'a, 'lua, Self, M>,
    ) {
        methods.add_method_mut("set", |_context, this, (x, y)| {
            Ok(this.set((x, y)))
        });

        methods.add_method("get_position", |_context, this, ()| {
            Ok(this.get())
        });
    }

    fn name() -> &'static str {
        "position"
    }
}

impl Position {
    pub fn get(&self) -> (f64, f64) {
        (self.0, self.1)
    }

    pub fn set(&mut self, (x, y): (f64, f64)) {
        self.0 = x;
        self.1 = y;
    }

    pub fn new(x: f64, y: f64) -> Position {
        Position(x, y)
    }
}